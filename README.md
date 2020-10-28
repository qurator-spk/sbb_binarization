# Binarization

> Binarization for document images

# Examples

<img src="https://user-images.githubusercontent.com/952378/63592437-e433e400-c5b1-11e9-9c2d-889c6e93d748.jpg" width="200"><img src="https://user-images.githubusercontent.com/952378/63592435-e433e400-c5b1-11e9-88e4-3e441b61fa67.jpg" width="200"><img src="https://user-images.githubusercontent.com/952378/63592440-e4cc7a80-c5b1-11e9-8964-2cd1b22c87be.jpg" width="240"><img src="https://user-images.githubusercontent.com/952378/63592438-e4cc7a80-c5b1-11e9-86dc-a9e9f8555422.jpg" width="240">

## Introduction

This tool performs document image binarization (i.e. transform colour/grayscale
to black-and-white pixels) for OCR using multiple trained models.

## Installation

Clone the repository, enter it and run

`pip install .`

### Models

Pre-trained models can be downloaded from here:   

https://qurator-data.de/sbb_binarization/

## Usage

```sh
sbb_binarize \
  --patches \
  -m <directory with models> \
  <input image> \
  <output image>
```

**Note** In virtually all cases, the `--patches` flag will improve results.
