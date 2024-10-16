# sbb_binarization

> Document Image Binarization

[![pip release](https://img.shields.io/pypi/v/sbb-binarization.svg)](https://pypi.org/project/sbb-binarization/)
[![GHActions CI](https://github.com/qurator-spk/sbb_binarization/actions/workflows/test.yml/badge.svg)](https://github.com/qurator-spk/sbb_binarization/actions/workflows/test.yml)
[![GHActions CD](https://github.com/qurator-spk/sbb_binarization/actions/workflows/docker-image.yml/badge.svg)](https://github.com/qurator-spk/sbb_binarization/actions/workflows/docker-image.yml)

<img src="https://user-images.githubusercontent.com/952378/63592437-e433e400-c5b1-11e9-9c2d-889c6e93d748.jpg" width="180"><img src="https://user-images.githubusercontent.com/952378/63592435-e433e400-c5b1-11e9-88e4-3e441b61fa67.jpg" width="180"><img src="https://user-images.githubusercontent.com/952378/63592440-e4cc7a80-c5b1-11e9-8964-2cd1b22c87be.jpg" width="220"><img src="https://user-images.githubusercontent.com/952378/63592438-e4cc7a80-c5b1-11e9-86dc-a9e9f8555422.jpg" width="220">

## Installation

Python `3.8-3.11` with Tensorflow `<2.13` are currently supported. While newer versions might also work, we currently don't test this.

You can either install from PyPI via 

    pip install sbb-binarization


or clone the repository, enter it and install (editable) with

    git clone git@github.com:qurator-spk/sbb_binarization.git
    cd sbb_binarization; pip install -e .


Alternatively, download the prebuilt image from Dockerhub:

    docker pull ocrd/sbb_binarization


### Models

Pre-trained models can be downloaded from the locations below. We also provide models and [model cards](https://huggingface.co/SBB/sbb_binarization) on 🤗 

| Version    |      Format   |  Download                                                                                            |
|------------|:-------------:|------------------------------------------------------------------------------------------------------|
| 2021-03-09 |  `SavedModel` | https://github.com/qurator-spk/sbb_binarization/releases/download/v0.0.11/saved_model_2021_03_09.zip |
| 2021-03-09 |  `HDF5` | https://qurator-data.de/sbb_binarization/2021-03-09/models.tar.gz                                          |
| 2020-01-16 | `SavedModel` | https://github.com/qurator-spk/sbb_binarization/releases/download/v0.0.11/saved_model_2020_01_16.zip  |
| 2020-01-16 |  `HDF5` | https://qurator-data.de/sbb_binarization/2020-01-16/models.tar.gz                                          |

With [OCR-D](https://ocr-d.de/), you can also use the [Resource Manager](https://ocr-d.de/en/models), e.g.

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
    
