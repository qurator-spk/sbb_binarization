# Binarization
> Binarization for document images

## Introduction
This tool performs document image binarization for OCR using a trained model.

## Installation
`./make`

### Models
Pretrained models can be downloaded from here:   
https://qurator-data.de/sbb_binarization/

## Usage 
`sbb_binarize -m <directory of models> -i <image> 
-p <set to true in order to let the model see image in patches> 
-s <provide a directory with a given ouput name and format. The result will be saved here.>`
