import torch
import torch.nn as nn
from torch import hub

from . import vggish_params

VGGISH_WEIGHTS = (
    "https://github.com/harritaylor/torchvggish/"
    "releases/download/v0.1/vggish-10086976.pth"
)
PCA_PARAMS = (
    "https://github.com/harritaylor/torchvggish/"
    "releases/download/v0.1/vggish_pca_params-970ea276.pth"
)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        x = self.embeddings(x)
        return x


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self, pretrained=True, progress=True):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        if pretrained:
            self.init_params_pth_url(PCA_PARAMS, progress=progress)

    def init_params_pth_url(self, pca_params_dict_url, progress=True):
        params = hub.load_state_dict_from_url(pca_params_dict_url, progress=progress)
        self.pca_matrix = torch.as_tensor(
            params[vggish_params.PCA_EIGEN_VECTORS_NAME]
        ).float()
        self.pca_means = torch.as_tensor(
            params[vggish_params.PCA_MEANS_NAME].reshape(-1, 1)
        ).float()
        self.pca_matrix = nn.Parameter(self.pca_matrix, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)
        assert self.pca_matrix.shape == (
            vggish_params.EMBEDDING_SIZE,
            vggish_params.EMBEDDING_SIZE,
        ), "Bad PCA matrix shape: %r" % (self.pca_matrix.shape,)
        assert self.pca_means.shape == (
            vggish_params.EMBEDDING_SIZE,
            1,
        ), "Bad PCA means shape: %r" % (self.pca_means.shape,)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_matrix, (embeddings_batch.T - self.pca_means)).T

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


def vggish(postprocess=True, pretrained=True, progress=True):
    """
    VGGish is a PyTorch port of Tensorflow's VGGish architecture
    used to create embeddings for Audioset. It produces a 128-d
    embedding of a 96ms slice of audio.
    """
    model = _vgg()
    if pretrained:
        state_dict = hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=progress)
        model.load_state_dict(state_dict)

    if postprocess:
        model = nn.Sequential(model, Postprocessor(pretrained, progress))

    return model
