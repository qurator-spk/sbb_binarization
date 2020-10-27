from os import environ
from os.path import join
from pathlib import Path
from pkg_resources import resource_string
from json import loads

from PIL import Image
import numpy as np
import cv2
from click import command

from ocrd_utils import (
    getLogger,
    assert_file_grp_cardinality,
    make_file_id,
    MIMETYPE_PAGE
)
from ocrd import Processor
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import AlternativeImageType, to_xml
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .sbb_binarize import SbbBinarizer

OCRD_TOOL = loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))
TOOL = 'ocrd-sbb-binarize'

def cv2pil(img):
    return Image.fromarray(img.astype('uint8'))

def pil2cv(img):
    # from ocrd/workspace.py
    color_conversion = cv2.COLOR_GRAY2BGR if img.mode in ('1', 'L') else  cv2.COLOR_RGB2BGR
    pil_as_np_array = np.array(img).astype('uint8') if img.mode == '1' else np.array(img)
    return cv2.cvtColor(pil_as_np_array, color_conversion)

class SbbBinarizeProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        if not(kwargs.get('show_help', None) or kwargs.get('dump_json', None) or kwargs.get('show_version')):
            if not 'model' in kwargs['parameter']:
                raise ValueError("'model' parameter is required")
            model_path = Path(kwargs['parameter']['model'])
            if not model_path.is_absolute():
                if 'SBB_BINARIZE_DATA' in environ:
                    model_path = Path(environ['SBB_BINARIZE_DATA']).joinpath(model_path)
                model_path = model_path.resolve()
            if not model_path.is_dir():
                raise FileNotFoundError("Does not exist or is not a directory: %s" % model_path)
            kwargs['parameter']['model'] = str(model_path)
        super().__init__(*args, **kwargs)

    def process(self):
        """
        Binarize with sbb_binarization
        """
        LOG = getLogger('processor.SbbBinarize')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        oplevel = self.parameter['operation_level']
        model_path = self.parameter['model'] # pylint: disable=attribute-defined-outside-init
        binarizer = SbbBinarizer(model_dir=model_path, logger=LOG)

        for n, input_file in enumerate(self.input_files):
            file_id = make_file_id(input_file, self.output_file_grp)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            pcgts.set_pcGtsId(file_id)
            page = pcgts.get_Page()

            if oplevel == 'page':
                LOG.info("Binarizing on 'page' level in page '%s'", page_id)
                page_image, page_xywh, _ = self.workspace.image_from_page(page, page_id, feature_filter='binarized')
                bin_image = cv2pil(binarizer.run(image=pil2cv(page_image), use_patches=True))
                # update METS (add the image file):
                bin_image_path = self.workspace.save_image_file(bin_image,
                        file_id + '.IMG-BIN',
                        page_id=input_file.pageId,
                        file_grp=self.output_file_grp)
                page.add_AlternativeImage(AlternativeImageType(filename=bin_image_path, comments='%s,binarized' % page_xywh['features']))

            elif oplevel == 'region':
                regions = page.get_AllRegions(['Text', 'Table'], depth=1)
                if not regions:
                    LOG.warning("Page '%s' contains no text/table regions", page_id)
                for region in regions:
                    region_image, region_xywh = self.workspace.image_from_segment(region, page_image, page_xywh, feature_filter='binarized')
                    region_image_bin = cv2pil(binarizer.run(image=pil2cv(region_image), use_patches=True))
                    region_image_bin_path = self.workspace.save_image_file(
                            region_image_bin,
                            "%s_%s.IMG-BIN" % (file_id, region.id),
                            page_id=input_file.pageId,
                            file_grp=self.output_file_grp)
                    region.add_AlternativeImage(
                        AlternativeImageType(filename=region_image_bin_path, comments='%s,binarized' % region_xywh['features']))

            elif oplevel == 'line':
                region_line_tuples = [(r.id, r.get_TextLine()) for r in page.get_AllRegions(['Text'], depth=0)]
                if not region_line_tuples:
                    LOG.warning("Page '%s' contains no text lines", page_id)
                for region_id, line in region_line_tuples:
                    line_image, line_xywh = self.workspace.image_from_segment(line, page_image, page_xywh, feature_filter='binarized')
                    line_image_bin = cv2pil(binarizer.run(image=pil2cv(line_image), use_patches=True))
                    line_image_bin_path = self.workspace.save_image_file(
                            line_image_bin,
                            "%s_%s_%s.IMG-BIN" % (file_id, region_id, line.id),
                            page_id=input_file.pageId,
                            file_grp=self.output_file_grp)
                    line.add_AlternativeImage(
                        AlternativeImageType(filename=line_image_bin_path, comments='%s,binarized' % line_xywh['features']))

            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts))

@command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(SbbBinarizeProcessor, *args, **kwargs)
