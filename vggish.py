"""
Created by hlt (Harri Taylor, taylorh23@cardiff.ac.uk), 6/2/2019
PyTorch implementation of the VGGish model taken from the tensorflow models
repository. Gratitude is extended to the AudioSet project authors for development
of the VGGish model.
(https://github.com/tensorflow/models/blob/master/research/audioset/vggish_slim.py)
"""

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

"""Defines the 'VGGish' model used to generate AudioSet embedding features.
The public AudioSet release (https://research.google.com/audioset/download.html)
includes 128-D features extracted from the embedding layer of a VGG-like model
that was trained on a large Google-internal YouTube dataset. Here we provide
a TF-Slim definition of the same model, without any dependences on libraries
internal to Google. We call it 'VGGish'.
Note that we only define the model up to the embedding layer, which is the
penultimate layer before the final classifier layer. We also provide various
hyperparameter values (in vggish_params.py) that were used to train this model
internally.
For comparison, here is TF-Slim's VGG definition:
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


VGGISH_WEIGHTS_URL = "https://users.cs.cf.ac.uk/taylorh23/pytorch/models/vggish11-8c8f01aa.pth"


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.fc1 = nn.Linear(in_features=12288, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool3(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)


def vggish(pretrained=True):
    """
    VGGish is a PyTorch implementation of Tensorflow's VGGish architecture used to create embeddings
    for Audioset. It produces a 128-d embedding of a 96ms slice of audio.
    """
    model = VGG()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(VGGISH_WEIGHTS_URL))
    return model
