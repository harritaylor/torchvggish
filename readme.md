# PyTorch-VGGish
Citation: https://ai.google/research/pubs/pub45611

To get started simply run the file `vggish.py`. This will download the
weights to your home dir. If you would like to verify the conversion
process, take a look at the `VGGish-TF_to_PT.ipynb`.
## VGGish Architecture
|---------------|---------|--------|--------|---------|
| Layer         | Filters | Kernel | Stride | Padding |
|---------------|---------|--------|--------|---------|
| 1. Conv2d     | 64      | 3      | 1      | SAME    |
| 2. MaxPool2d  | -       | 2      | 2      | SAME    |
| 3. Conv2d     | 128     | 3      | 1      | SAME    |
| 4. MaxPool2d  | -       | 2      | 2      | SAME    |
| 5. Conv2d     | 256     | 3      | 1      | SAME    |
| 6. Conv2d     | 256     | 3      | 1      | SAME    |
| 7. MaxPool2d  | -       | 2      | 2      | SAME    |
| 8. Conv2d     | 512     | 3      | 1      | SAME    |
| 9. Conv2d     | 512     | 3      | 1      | SAME    |
| 10. MaxPool2d | -       | 2      | 2      | SAME    |
| 11. Flatten   | -       | -      | -      | -       |
| 12. Linear    | 4096    | -      | -      | -       |
| 13. Linear    | 4096    | -      | -      | -       |
| 14. Linear    | 128     | -      | -      | -       |
|---------------|---------|--------|--------|---------|

