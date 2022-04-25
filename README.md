# Binarization

> Binarization for document images

## Examples

<img src="https://user-images.githubusercontent.com/952378/63592437-e433e400-c5b1-11e9-9c2d-889c6e93d748.jpg" width="180"><img src="https://user-images.githubusercontent.com/952378/63592435-e433e400-c5b1-11e9-88e4-3e441b61fa67.jpg" width="180"><img src="https://user-images.githubusercontent.com/952378/63592440-e4cc7a80-c5b1-11e9-8964-2cd1b22c87be.jpg" width="220"><img src="https://user-images.githubusercontent.com/952378/63592438-e4cc7a80-c5b1-11e9-86dc-a9e9f8555422.jpg" width="220">

## Introduction

This tool performs document image binarization using trained models. The method is based on [Calvo-Zaragoza and Gallego, 2018](https://arxiv.org/abs/1706.10241).

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
  -m <path to directory containing model files> \
  <input image> \
  <output image>
```

**Note** In virtually all cases, applying the `--patches` flag will improve the quality of results.

Example

```sh
sbb_binarize --patches -m /path/to/models/ myimage.tif myimage-bin.tif
```

To use the [OCR-D](https://ocr-d.de/) interface:
```sh
ocrd-sbb-binarize --overwrite -I INPUT_FILE_GRP -O OCR-D-IMG-BIN -P model "/var/lib/sbb_binarization"
```
