#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from json import load
from setuptools import setup, find_packages

with open('./ocrd-tool.json', 'r') as f:
    version = load(f)['version']

install_requires = open('requirements.txt').read().split('\n')

setup(
    name='sbb_binarization',
    version=version,
    description='Pixelwise binarization with selectional auto-encoders in Keras',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Vahid Rezanezhad',
    url='https://github.com/qurator-spk/sbb_binarization',
    license='Apache License 2.0',
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    package_data={'': ['*.json', '*.yml', '*.yaml']},
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'sbb_binarize=sbb_binarize.cli:main',
            'ocrd-sbb-binarize=sbb_binarize.ocrd_cli:cli',
        ]
    },
)
