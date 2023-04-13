"""
sbb_binarize CLI
"""

from click import command, option, argument, version_option, types
from .sbb_binarize import SbbBinarizer

@command()
@version_option(package_name="sbb-binarization")
@option('--model-dir', '-m', type=types.Path(exists=True, file_okay=False), required=True, help='directory containing models for prediction')
@argument('input_image')
@argument('output_image')
def main(model_dir, input_image, output_image):
    SbbBinarizer(model_dir).run(image_path=input_image, save=output_image)
