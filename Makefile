# BEGIN-EVAL makefile-parser --make-help Makefile

DOCKER_BASE_IMAGE = docker.io/ocrd/core:v2.69.0
DOCKER_TAG = ocrd/sbb_binarization

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
	@echo "    docker  Build a Docker image $(DOCKER_TAG) from $(DOCKER_BASE_IMAGE)"
	@echo ""
	@echo "  Variables"
	@echo ""
	@echo "    PYTHON"
	@echo "    DOCKER_TAG    Docker image tag of result for the docker target"

# END-EVAL

# Install with pip
install:
	pip install -U setuptools pip
	pip install .

# Downloads the pre-trained models from qurator-data.de
.PHONY: models
models:
	ocrd resmgr download ocrd-sbb-binarize "*"

repo/assets/data:
	git submodule update --init

# Setup test data
test/assets: repo/assets/data
	@mkdir -p $@
	cp -r -t $@ $</*

# Run tests
.PHONY: test
test: test/assets models
	ocrd-sbb-binarize -m test/assets/kant_aufklaerung_1784/data/mets.xml -I OCR-D-IMG -O BIN -P model default
	ocrd-sbb-binarize -m test/assets/kant_aufklaerung_1784/data/mets.xml -I OCR-D-IMG -O BIN2 -P model default-2021-03-09
	ocrd-sbb-binarize -m test/assets/kant_aufklaerung_1784-page-region/data/mets.xml -g phys_0001 -I OCR-D-GT-SEG-REGION -O BIN -P model default -P operation_level region
	ocrd-sbb-binarize -m test/assets/kant_aufklaerung_1784-page-region/data/mets.xml -g phys_0001 -I OCR-D-GT-SEG-REGION -O BIN2 -P model default-2021-03-09 -P operation_level region

.PHONY: docker
docker:
	docker build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .


.PHONY: clean
clean:
	-$(RM) -fr test/assets
