# TODO: AlternativeImage 'binarized' comment should be additive

import os.path
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
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return Image.fromarray(color_coverted)

def pil2cv(img):
    # from ocrd/workspace.py
    color_conversion = cv2.COLOR_GRAY2BGR if img.mode in ('1', 'L') else  cv2.COLOR_RGB2BGR
    pil_as_np_array = np.array(img).astype('uint8') if img.mode == '1' else np.array(img)
    return cv2.cvtColor(pil_as_np_array, color_conversion)

class SbbBinarizeProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)

    def _run_binarizer(self, img):
        return cv2pil(
                SbbBinarizer(
                    image=pil2cv(img),
                    model=self.model_path,
                    patches=self.use_patches,
                    save=None).run())

    def process(self):
        """
        Binarize with sbb_binarization
        """
        LOG = getLogger('processor.SbbBinarize')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        oplevel = self.parameter['operation_level']
        self.use_patches = self.parameter['patches'] # pylint: disable=attribute-defined-outside-init
        self.model_path = self.parameter['model'] # pylint: disable=attribute-defined-outside-init

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
                bin_image = self._run_binarizer(page_image)
                # update METS (add the image file):
                bin_image_path = self.workspace.save_image_file(bin_image,
                        file_id + '.IMG-BIN',
                        page_id=input_file.pageId,
                        file_grp=self.output_file_grp)
                page.add_AlternativeImage(AlternativeImageType(filename=bin_image_path, comment=page_xywh['features']+",binarized"))

            else:
                regions = page.get_AllRegions(['Text', 'Table'])
                if not regions:
                    LOG.warning("Page '%s' contains no text/table regions", page_id)

                for region in regions:
                    region_image, region_xywh = self.workspace.image_from_segment(region, page_image, page_xywh, feature_filter='binarized')

                    if oplevel == 'region':
                        region_image_bin = self._run_binarizer(region_image)
                        region_image_bin_path = self.workspace.save_image_file(
                                region_image_bin,
                                "%s_%s.IMG-BIN" % (file_id, region.id),
                                page_id=input_file.pageId,
                                file_grp=self.output_file_grp)
                        region.add_AlternativeImage(
                            AlternativeImageType(filename=region_image_bin_path, comments=region_xywh['features']+',binarized'))

                    elif oplevel == 'line':
                        lines = region.get_TextLine()
                        if not lines:
                            LOG.warning("Page '%s' region '%s' contains no text lines", page_id, region.id)
                        for line in lines:
                            line_image, line_xywh = self.workspace.image_from_segment(line, page_image, page_xywh, feature_filter='binarized')
                            line_image_bin = self._run_binarizer(line_image)
                            line_image_bin_path = self.workspace.save_image_file(
                                    line_image_bin,
                                    "%s_%s_%s.IMG-BIN" % (file_id, region.id, line.id),
                                    page_id=input_file.pageId,
                                    file_grp=self.output_file_grp)
                            line.add_AlternativeImage(
                                AlternativeImageType(filename=line_image_bin_path, comments=line_xywh['features']+',binarized'))

            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts))

@command()
@ocrd_cli_options
def cli(*args, **kwargs):
    return ocrd_cli_wrap_processor(SbbBinarizeProcessor, *args, **kwargs)