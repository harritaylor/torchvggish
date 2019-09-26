import numpy as np
import torch
import torch.nn as nn
from torch import hub
import os
import sys
import warnings
import errno
from urllib.parse import urlparse

# TODO: Upload weights and PCA to project release

VGGISH_WEIGHTS = "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish-10086976.pth"
PCA_PARAMS = "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish_pca_params-4d878af3.npz"


class VGG(nn.Module):
    def __init__(self, features, pca=True):
        super(VGG, self).__init__()
        self.pca = pca
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True)
        )
        self.pproc = Postprocessor()

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to remain compatible
        # with the original VGGish model
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        x = self.embeddings(x)
        x = self.pproc.postprocess(x) if self.pca else x
        return x


class Postprocessor(object):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations.
    """

    def __init__(self, pca_params_npz_path=None):
        """Constructs a postprocessor.

        Args:
          pca_params_npz_path: Path to a NumPy-format .npz file that
            contains the PCA parameters used in postprocessing.
        """
        if pca_params_npz_path is not None:
            params = np.load(pca_params_npz_path)
        else:
            params = load_params_from_url(PCA_PARAMS)
        self._pca_matrix = torch.as_tensor(params['pca_eigen_vectors']).float()
        self._pca_means = torch.as_tensor(params['pca_means'].reshape(-1, 1)).float()

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        pca_applied = torch.mm(self._pca_matrix,
                               (embeddings_batch.t() - self._pca_means)).t()
        clipped_embeddings = torch.clamp(pca_applied, -2.0, +2.0)
        quantized_embeddings = torch.round((clipped_embeddings - -2.0)
                                           * (255.0 / (+2.0 - -2.0)))
        return quantized_embeddings


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M']:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


def vggish():
    """
    VGGish is a PyTorch implementation of Tensorflow's VGGish architecture used to create embeddings
    for Audioset. It produces a 128-d embedding of a 96ms slice of audio. Always comes pretrained.
    """
    model = _vgg()
    state_dict = hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True)
    model.load_state_dict(state_dict)
    return model


def load_params_from_url(url, param_dir=None, progress=True):
    r"""
    Loads the PCA params using the syntax from
    https://github.com/pytorch/pytorch/blob/master/torch/hub.py,
    except doesn't serialize using torch.load, simply provides files as numpy format.
    """
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
    return np.load(cached_file)
