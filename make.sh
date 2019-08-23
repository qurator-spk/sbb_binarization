all: build install

build:
       python3 setup.py build

install:
       python3 setup.py install --user
       cp sbb_binarize/sbb_binarize.py ~/bin/sbb_binarize
       chmod +x ~/bin/sbb_binarize
