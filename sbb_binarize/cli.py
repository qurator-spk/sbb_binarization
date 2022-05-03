"""
sbb_binarize CLI
"""

from click import command, option, argument, version_option, types
from .sbb_binarize import SbbBinarizer
import click

@command()
@version_option()
@option('--patches/--no-patches', default=True, help='by enabling this parameter you let the model to see the image in patches.')
@option('--model-dir', '-m', type=click.Path(exists=True, file_okay=False), required=True, help='directory containing models for prediction')
@argument('input_image')
@argument('output_image')
@click.option(
    "--dir_in",
    "-di",
    help="directory of images",
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--dir_out",
    "-do",
    help="directory where the binarized images will be written",
    type=click.Path(exists=True, file_okay=False),
)

def main(patches, model_dir, input_image, output_image, dir_in, dir_out):
    if not dir_out and (dir_in):
        print("Error: You used -di but did not set -do")
        sys.exit(1)
    elif dir_out and not (dir_in):
        print("Error: You used -do to write out binarized images but have not set -di")
        sys.exit(1)
    SbbBinarizer(model_dir).run(image_path=input_image, use_patches=patches, save=output_image, dir_in=dir_in, dir_out=dir_out)
