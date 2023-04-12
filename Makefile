# BEGIN-EVAL makefile-parser --make-help Makefile

.PHONY: help install
help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    install  Install with pip"
	@echo "    models   Downloads the pre-trained models from qurator-data.de"
	@echo "    test     Run tests"
	@echo "    clean    Remove copies/results in test/assets"
	@echo ""
	@echo "  Variables"
	@echo ""

# END-EVAL

# Install with pip
install:
	pip install .

# Downloads the pre-trained models from qurator-data.de
.PHONY: models
models:
	ocrd resmgr download ocrd-sbb-binarize "*"

repo/assets:
	git submodule update --init

# Setup test data
test/assets: repo/assets
	@mkdir -p $@
	cp -r -t $@ repo/assets/data/*

# Run tests
.PHONY: test
test: test/assets models
	ocrd-sbb-binarize -m test/assets/kant_aufklaerung_1784/data/mets.xml -I OCR-D-IMG -O BIN -P model default
	ocrd-sbb-binarize -m test/assets/kant_aufklaerung_1784/data/mets.xml -I OCR-D-IMG -O BIN2 -P model default-2021-03-09
	ocrd-sbb-binarize -m test/assets/kant_aufklaerung_1784-page-region/data/mets.xml -g phys_0001 -I OCR-D-GT-SEG-REGION -O BIN -P model default -P operation_level region
	ocrd-sbb-binarize -m test/assets/kant_aufklaerung_1784-page-region/data/mets.xml -g phys_0001 -I OCR-D-GT-SEG-REGION -O BIN2 -P model default-2021-03-09 -P operation_level region

.PHONY: clean
clean:
	-$(RM) -fr test/assets
