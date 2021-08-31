#!/usr/bin/env python3
from setuptools import find_packages, setup

long_description = open("README.md", "r", encoding="utf-8").read()

setup(
    name="torchvggish",
    version="v0.1.1",
    description="Pytorch port of Google Research's VGGish model used for extracting audio features.",
    author="Harri Taylor",
    author_email="harritaylor@protonmail.com",
    url="https://github.com/harritaylor/torchvggish",
    license="Apache-2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=['torch', 'numpy', 'resampy', 'soundfile'],
    extras_require={
    },
)

