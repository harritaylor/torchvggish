
# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications copyright (c) 2019 Harrison Taylor taylorh23@cardiff.ac.uk

""""Defines the 'VGGish' model used to generate AudioSet embedding features.
The public AudioSet release (https://research.google.com/audioset/download.html)
includes 128-D features extracted from the embedding layer of a VGG-like model
that was trained on a large Google-internal YouTube dataset. Here we provide
a TF-Slim definition of the same model, without any dependences on libraries
internal to Google. We call it 'VGGish'.
Note that we only define the model up to the embedding layer, which is the
penultimate layer before the final classifier layer. We also provide various
hyperparameter values (in vggish_params.py) that were used to train this model
internally." - The TensorFlow Authors

This is a PyTorch implementation of the same model, using weights
extracted from the TF-Slim implementation of VGGish. For comparison,
here is TF-Slim's VGG definition:
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py

"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

WEIGHT_URL = "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish_features-7a250f89.pt"


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


def vggish(pretrained=True):
    """
    VGGish is a PyTorch implementation of Tensorflow's VGGish architecture used to create embeddings
    for Audioset. It produces a 128-d embedding of a 96ms slice of audio.
    """
    model = VGG()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(WEIGHT_URL))
    return model

# Test to make sure everything has loaded
if __name__ == '__main__':
    model = vggish().features
    print("Everything loaded successfully. VGGish model architecture:")
    print(model)
