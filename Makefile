# BEGIN-EVAL makefile-parser --make-help Makefile

help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install  Install with pip"
	@echo "    model    Downloads the pre-trained models from qurator-data.de"
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
.PHONY: model
model:
	ocrd resmgr download --allow-uninstalled --location cwd ocrd-sbb-binarize default

# Run tests
test: model
	cd repo/assets/data/kant_aufklaerung_1784/data; ocrd-sbb-binarize -I OCR-D-IMG -O BIN -P model default
	cd repo/assets/data/kant_aufklaerung_1784-page-region/data; ocrd-sbb-binarize -I OCR-D-IMG -O BIN -P model default -P operation_level region
