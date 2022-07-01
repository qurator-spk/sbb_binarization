"""
sbb_binarize CLI
"""
import click
from click import command, option, argument, version_option, types
from .sbb_binarize import SbbBinarizer

@command()
@version_option()
@option('--patches/--no-patches', default=True, help='by enabling this parameter you let the model to see the image in patches.')
@option('--model-dir', '-m', type=click.Path(exists=True, file_okay=False), required=True, help='directory containing models for prediction')
@argument('input_image')
@argument('output_image')
def main(patches, model_dir, input_image, output_image):
    SbbBinarizer(model_dir).run(image_path=input_image, use_patches=patches, save=output_image)
