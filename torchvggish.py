import numpy as np
import torch
import torch.nn as nn
import vggish_params
from torch import hub
import os
import sys
import warnings
import errno
from urllib.parse import urlparse


VGGISH_WEIGHTS = (
    "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish-10086976.pth"
)
PCA_PARAMS = (
    "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish_pca_params-4d878af3.npz"
)


class VGGishParams:
    """
    These should not be changed. They have been added into this file for convenience.
    """

    NUM_FRAMES = (96,)  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.

    # Hyperparameters used in feature and example generation.
    SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96  # with zero overlap.

    # Parameters used for embedding postprocessing.
    PCA_EIGEN_VECTORS_NAME = "pca_eigen_vectors"
    PCA_MEANS_NAME = "pca_means"
    QUANTIZE_MIN_VAL = -2.0
    QUANTIZE_MAX_VAL = +2.0


"""
VGGish
Input: 96x64 1-channel spectrogram
Output:  128 Embedding 
"""
class VGGish(nn.Module):  
    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(  1,  64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.embeddings = nn.Sequential(
            nn.Linear(512*4*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, VGGishParams.EMBEDDING_SIZE),
            nn.ReLU(inplace=True))
                    
    def forward(self, x):
        x = self.features(x)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0),-1)
        x = self.embeddings(x)
        return x


class TorchPostprocessor(object):
    """Post-processes VGGish embeddings. Returns a non-quantized version to preserve
    the gradient in pytorch.

    The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations.
    """

    def __init__(self, pca_params_npz_path):
        """Constructs a postprocessor.

        Args:
          pca_params_npz_path: Path to a NumPy-format .npz file that
            contains the PCA parameters used in postprocessing.
        """
        params = np.load(pca_params_npz_path)
        self._pca_matrix = torch.as_tensor(params[vggish_params.PCA_EIGEN_VECTORS_NAME])
        # Load means into a column vector for easier broadcasting later.
        self._pca_means = torch.as_tensor(
            params[vggish_params.PCA_MEANS_NAME].reshape(-1, 1)
        )

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed
          and quantized version of the input.
        """

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(
            self._pca_matrix, (embeddings_batch.t() - self._pca_means)
        ).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL) * (
            255.0 / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
        )

        return quantized_embeddings


def vggish():
    """
    VGGish is a PyTorch implementation of Tensorflow's VGGish architecture used to create embeddings
    for Audioset. It produces a 128-d embedding of a 96ms slice of audio. Always comes pretrained.
    """
    model = VGGish()
    model.load_state_dict(hub.load_state_dict_from_url(VGGISH_WEIGHTS), strict=True)
    return model


def load_params_from_url(url, param_dir=None, progress=True):
    r"""
    Loads the PCA params using the syntax from https://github.com/pytorch/pytorch/blob/master/torch/hub.py,
    except doesn't serialize using torch.load, simply provides files as numpy format.
    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if param_dir is None:
        torch_home = hub._get_torch_home()
        param_dir = os.path.join(torch_home, "params")

    try:
        os.makedirs(param_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(param_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = hub.HASH_REGEX.search(filename).group(1)
        hub._download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return cached_file
