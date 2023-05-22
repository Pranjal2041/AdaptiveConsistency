import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setup(
    name = "AdaptiveConsistency",
    version = "1.0.0",
    description = ("Library for running AdapativeConsistency based Inference on large language models."),
    license = "MIT",
    packages=find_packages(),
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
)
