# sbb_binarization

> Document Image Binarization using pre-trained models

[![pip release](https://img.shields.io/pypi/v/sbb-binarization.svg)](https://pypi.org/project/sbb-binarization/)
[![CircleCI test](https://circleci.com/gh/qurator-spk/sbb_binarization.svg?style=shield)](https://circleci.com/gh/qurator-spk/sbb_binarization)
[![GHActions Tests](https://github.com/qurator-spk/sbb_binarization/actions/workflows/test.yml/badge.svg)](https://github.com/qurator-spk/sbb_binarization/actions/workflows/test.yml)

## Examples

<img src="https://user-images.githubusercontent.com/952378/63592437-e433e400-c5b1-11e9-9c2d-889c6e93d748.jpg" width="180"><img src="https://user-images.githubusercontent.com/952378/63592435-e433e400-c5b1-11e9-88e4-3e441b61fa67.jpg" width="180"><img src="https://user-images.githubusercontent.com/952378/63592440-e4cc7a80-c5b1-11e9-8964-2cd1b22c87be.jpg" width="220"><img src="https://user-images.githubusercontent.com/952378/63592438-e4cc7a80-c5b1-11e9-86dc-a9e9f8555422.jpg" width="220">

## Installation

Python versions `3.7-3.10` are currently supported.

You can either install via 

```
pip install sbb-binarization
```

or clone the repository, enter it and install (editable) with

```
git clone git@github.com:qurator-spk/sbb_binarization.git
cd sbb_binarization; pip install -e .
```

### Models

Pre-trained models can be downloaded from the locations below. We also provide the models and [model card](https://huggingface.co/SBB/sbb_binarization) on 🤗 

| Version   |      Format      |  Download |
|----------|:-------------:|-------|
| 2021-03-09 |  `SavedModel` | https://github.com/qurator-spk/sbb_binarization/releases/download/v0.0.11/saved_model_2021_03_09.zip |
| 2021-03-09 |  `HDF5` | https://qurator-data.de/sbb_binarization/2021-03-09/models.tar.gz |
| 2020-01-16 | `SavedModel` | https://github.com/qurator-spk/sbb_binarization/releases/download/v0.0.11/saved_model_2020_01_16.zip |
| 2020-01-16 |  `HDF5` | https://qurator-data.de/sbb_binarization/2020-01-16/models.tar.gz |

With [OCR-D](https://ocr-d.de/), you can use the [Resource Manager](https://ocr-d.de/en/models) to deploy models, e.g.

    ocrd resmgr download ocrd-sbb-binarize "*"


## Usage

```sh
sbb_binarize \
  -m <path to directory containing model files> \
  <input image> \
  <output image>
```

**Note:** the output image MUST use either `.tif` or `.png` as file extension to produce a binary image. Input images can also be JPEG.

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
    
        make models
        make test
    
