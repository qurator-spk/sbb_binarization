"""
sbb_binarize CLI
"""

from argparse import ArgumentParser

from .sbb_binarize import SbbBinarizer

def main():
    parser = ArgumentParser()

    parser.add_argument('-i', '--image', default=None, help='image.')
    parser.add_argument('-p', '--patches', default=False, help='by setting this parameter to true you let the model to see the image in patches.')
    parser.add_argument('-s', '--save', default=False, help='save prediction with a given name here. The name and format should be given (outputname.tif).')
    parser.add_argument('-m', '--model', default=None, help='models directory.')

    options = parser.parse_args()

    binarizer = SbbBinarizer(
            image_path=options.image,
            model=options.model,
            patches=options.patches,
            save=options.save
    )
    binarizer.run()

if __name__ == "__main__":
    main()
