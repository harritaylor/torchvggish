import torch
import torch.nn as nn
from torch import hub

VGGISH_WEIGHTS = "https://github.com/harritaylor/torchvggish/" \
                 "releases/download/v0.1/vggish-10086976.pth"
PCA_PARAMS = "https://github.com/harritaylor/torchvggish/" \
             "releases/download/v0.1/vggish_pca_params-970ea276.pth"


class VGG(nn.Module):
    def __init__(self, features, postprocess):
        super(VGG, self).__init__()
        self.postprocess = postprocess
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True),
        )
        if postprocess:
            self.pproc = Postprocessor()

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        x = self.embeddings(x)
        x = self.pproc.postprocess(x) if self.postprocess else x
        return x


class Postprocessor(object):
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

    def __init__(self):
        """Constructs a postprocessor."""
        params = hub.load_state_dict_from_url(PCA_PARAMS)
        self._pca_matrix = torch.as_tensor(params["pca_eigen_vectors"]).float()
        self._pca_means = torch.as_tensor(params["pca_means"].reshape(-1, 1)).float()

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        pca_applied = torch.mm(
            self._pca_matrix, (embeddings_batch.t() - self._pca_means)
        ).t()
        clipped_embeddings = torch.clamp(pca_applied, -2.0, +2.0)
        quantized_embeddings = torch.round(
            (clipped_embeddings - -2.0) * (255.0 / (+2.0 - -2.0))
        )
        return torch.squeeze(quantized_embeddings)


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


def _vgg(postprocess=False):
    return VGG(make_layers(), postprocess)


def vggish(postprocess=True):
    """
    VGGish is a PyTorch port of Tensorflow's VGGish architecture
    used to create embeddings for Audioset. It produces a 128-d
    embedding of a 96ms slice of audio.
    """
    model = _vgg(postprocess)
    state_dict = hub.load_state_dict_from_url(VGGISH_WEIGHTS, progress=True)
    model.load_state_dict(state_dict)
    return model
