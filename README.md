# Binarization

> Binarization for document images

[![pip release](https://img.shields.io/pypi/v/sbb-binarization.svg)](https://pypi.org/project/sbb-binarization/)
[![CircleCI test](https://circleci.com/gh/qurator-spk/sbb_binarization.svg?style=shield)](https://circleci.com/gh/qurator-spk/sbb_binarization)
[![GHActions Tests](https://github.com/qurator-spk/sbb_binarization/actions/workflows/test.yml/badge.svg)](https://github.com/qurator-spk/sbb_binarization/actions/workflows/test.yml)

## Examples

<img src="https://user-images.githubusercontent.com/952378/63592437-e433e400-c5b1-11e9-9c2d-889c6e93d748.jpg" width="180"><img src="https://user-images.githubusercontent.com/952378/63592435-e433e400-c5b1-11e9-88e4-3e441b61fa67.jpg" width="180"><img src="https://user-images.githubusercontent.com/952378/63592440-e4cc7a80-c5b1-11e9-8964-2cd1b22c87be.jpg" width="220"><img src="https://user-images.githubusercontent.com/952378/63592438-e4cc7a80-c5b1-11e9-86dc-a9e9f8555422.jpg" width="220">

## Introduction

This tool performs document image binarization using a trained ResNet50-UNet model. 

## Installation

Clone the repository, enter it and run

`pip install .`

### Models

Pre-trained models in HDF5 format can be downloaded from here:

https://qurator-data.de/sbb_binarization/

We also provide models in Tensorflow SavedModel format via Huggingface and Github release assets:

https://huggingface.co/SBB/sbb_binarization
https://github.com/qurator-spk/sbb_binarization/releases

With [OCR-D](https://ocr-d.de/), you can use the [Resource Manager](Tensorflow SavedModel) to deploy models, e.g.

    ocrd resmgr download ocrd-sbb-binarize "*"


## Usage

```sh
sbb_binarize \
  -m <path to directory containing model files \
  <input image> \
  <output image>
```

Images containing a lot of border noise (black pixels) should be cropped beforehand to improve the quality of results.

### Example


    sbb_binarize -m /path/to/model/ myimage.tif myimage-bin.tif


To use the [OCR-D](https://ocr-d.de/en/spec/cli) interface:

    ocrd-sbb-binarize -I INPUT_FILE_GRP -O OCR-D-IMG-BIN -P model default


## Testing

For simple smoke tests, the following will
- download models
- download test data
- run the OCR-D wrapper (on page and region level):
    
        make model
        make test
    
