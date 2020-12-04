# Directory to store models
MODEL_DIR = $(PWD)/models

# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install  Install with pip"
	@echo "    models   Downloads the pre-trained models from qurator-data.de"
	@echo "    test     Run tests"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    MODEL_DIR  Directory to store models"

# END-EVAL

# Install with pip
install:
	pip install .

# Downloads the pre-trained models from qurator-data.de
models: $(MODEL_DIR)/model1_bin.h5

$(MODEL_DIR)/model1_bin.h5: models.tar.gz
	tar xf models.tar.gz

models.tar.gz:
	wget 'https://qurator-data.de/sbb_binarization/models.tar.gz'

# Run tests
test: model
	cd repo/assets/data/kant_aufklaerung_1784/data; ocrd-sbb-binarize -I OCR-D-IMG -O BIN -P model $(MODEL_DIR)
	cd repo/assets/data/kant_aufklaerung_1784-page-region/data; ocrd-sbb-binarize -I OCR-D-IMG -O BIN -P model $(MODEL_DIR) -P level-of-operation region
