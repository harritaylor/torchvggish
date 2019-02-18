# PyTorch-VGGish

Citation: https://ai.google/research/pubs/pub45611

Resources: http://users.cs.cf.ac.uk/TaylorH23/vggish/vggish_pca_params.npz
As well as the .pt weights of VGGish

## Todo
- [x] https://storage.googleapis.com/audioset/vggish_model.ckpt
- [x] https://storage.googleapis.com/audioset/vggish_pca_params.npz
- [x] https://github.com/tensorflow/models/tree/master/research/audioset
- [ ] Run example embedding code with provided WAV files
- [ ] Read AudioSet paper to figure out what the code is doing
- [ ] Initialise VGGish model and extract weights
- [ ] Clone VGGish architecture into PyTorch format

## VGGish Architecture

| Layer        | Filters | Kernel | Stride | Padding |
|:-------------|:--------|:-------|:-------|:--------|
| 1. Conv2d    | 64      | 3      | 1      | SAME    |
| 2. MaxPool2d | -       | 2      | 2      | SAME    |
| 3. Conv2d    | 128     | 3      | 1      | SAME    |
| 4. MaxPool2d | -       | 2      | 2      | SAME    |
| 5. Conv2d    | 256     | 3      | 1      | SAME    |
| 6. Conv2d    | 256     | 3      | 1      | SAME    |
| 7. MaxPool2d | -       | 2      | 2      | SAME    |
| 8. Conv2d    | 512     | 3      | 1      | SAME    |
| 9. Conv2d    | 512     | 3      | 1      | SAME    |
| 10. MaxPool2d| -       | 2      | 2      | SAME    |
| 11. Flatten  | -       | -      | -      | -       |
| 12. Linear   | 4096    | -      | -      | -       |
| 13. Linear   | 4096    | -      | -      | -       |
| 14. Linear   | E_SIZE  | -      | -      | -       |
