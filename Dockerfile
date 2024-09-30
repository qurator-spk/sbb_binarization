ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/kontakt" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/qurator-spk/sbb_binarization" \
    org.label-schema.build-date=$BUILD_DATE

WORKDIR /build/sbb_binarization
COPY setup.py .
COPY ocrd-tool.json .
COPY sbb_binarize ./sbb_binarize
COPY requirements.txt .
COPY README.md .
COPY Makefile .
RUN make install
RUN rm -rf /build/sbb_binarization

WORKDIR /data
VOLUME ["/data"]
