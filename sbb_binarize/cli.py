"""
sbb_binarize CLI
"""

from click import command, option, argument, version_option, types

from .sbb_binarize import SbbBinarizer


@command()
@version_option()
@option('--model-dir', '-m', type=types.Path(exists=True, file_okay=False), required=True, help='directory containing models for prediction')
@argument('input_image')
@argument('output_image')
def main(model_dir, input_image, output_image):
    binarizer = SbbBinarizer()
    binarizer.load_model(model_dir)
    binarizer.binarize_image_file(image_path=input_image, save_path=output_image)
