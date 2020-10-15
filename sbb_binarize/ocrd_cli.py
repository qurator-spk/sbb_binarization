import os.path
from pkg_resources import resource_string
from json import loads

from ocrd_utils import (
    getLogger,
    assert_file_grp_cardinality,
    make_file_id,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file
from ocrd_models.ocrd_page import (
    MetadataItemType,
    LabelsType, LabelType,
    AlternativeImageType,
    TextRegionType,
    to_xml
)
from ocrd import Processor

from .sbb_binarize import SbbBinarizer

OCRD_TOOL = loads(resource_string(__name__, 'ocrd-tool.json').decode('utf8'))
TOOL = 'ocrd-sbb-binarize'

class SbbBinarizeProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)

    def process(self):
        """
        Binarize with sbb_binarization
        """
        LOG = getLogger('processor.SbbBinarize')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        oplevel = self.parameter['operation_level']
        use_patches = self.parameter['patches']
        model_path = self.parameter['model']

        for n, input_file in enumerate(self.input_files):
            file_id = make_file_id(input_file, self.output_file_grp)
            page_id = input_file.pageId or input_file.ID
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()

            if oplevel == 'page':
                LOG.info("Binarizing on 'page' level in page '%s'", page_id)
                page_image, page_xywh, _ = self.workspace.image_from_page(page, page_id)
                binarizer = SbbBinarizer(image=page_image, model=model_path, patches=use_patches, save=None)
                bin_image = binarizer.run()
                # update METS (add the image file):
                bin_image_path = self.workspace.save_image_file(bin_image,
                        file_id + '.IMG-BIN',
                        page_id=page_id,
                        file_grp=self.output_file_grp)
                page.add_AlternativeImage(filename=bin_image_path, comment="binarized")
            else:
                raise NotImplementedError("Binarization below page level not implemented yet")

            file_id = make_file_id(input_file, self.output_file_grp)
            pcgts.set_pcGtsId(file_id)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts))
