import torch.nn as nn
import torch.utils.model_zoo as model_zoo

WEIGHT_URL = "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish-e3b372a4.pth"

class VGGish(nn.Module):
    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.embeddings = nn.Sequential(
            nn.Linear(512*24, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 128)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.embeddings(x)
        return x


def vggish(pretrained=True):
    """
    VGGish is a PyTorch implementation of Tensorflow's VGGish architecture used to create embeddings
    for Audioset. It produces a 128-d embedding of a 96ms slice of audio.
    """
    model = VGGish()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(WEIGHT_URL))
    return model

# Test to make sure everything has loaded
if __name__ == '__main__':
    model = vggish()
    print("Everything loaded successfully. VGGish model architecture:")
    print(model)
