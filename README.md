# Torch VGGish
A PyTorch port of [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset)<sup>1</sup>, 
a feature embedding frontend for audio classification models. The weights are ported directly from the tensorflow model, so embeddings created using `torchvggish` will be identical.

## Quick start
There are two options: you can install the last stable version from pypi, or clone this repo and install.
```shell script
# optional: create virtual env
cd torchvggish && python3 -m venv .env
source activate .env/bin/activate

pip install -i https://test.pypi.org/simple/ torchvggish==0.1

# OR get the latest version
git clone git@github.com:harritaylor/torchvggish.git
pip install -r requirements.txt
```
## Usage
Barebones example of creating embeddings from an `example` wav file:
```python
from torchvggish import vggish, vggish_input

# Initialise model and download weights
embedding_model = vggish()
embedding_model.eval()

example = vggish_input.wavfile_to_examples("example.wav")
embeddings = embedding_model.forward(example)
```

<hr>
1.  S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\
    in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\
    Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
    

