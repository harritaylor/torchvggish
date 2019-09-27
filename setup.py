import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchvggish",
    version="0.1",
    author="Harri Taylor",
    author_email="taylorh23@cardiff.ac.uk",
    description="A Pytorch port of Tensorflow's VGGish embedding model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harritaylor/torchvggish",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6',
)