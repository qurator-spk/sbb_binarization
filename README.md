# Binarization

> Binarization for document images

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
  -m <directory with models> \
  -i <image file> \
  -p <set to true to let the model see the image divided into patches> \
  -s <directory where the results will be saved>`
```
