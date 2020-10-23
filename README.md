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
  --patches \
  -m <directory with models> \
  <input image> \
  <output image>
```

**Note** In virtually all cases, the `--patches` flag will improve results.
