#!/usr/bin/env python

from setuptools import setup, find_packages

# with open("README.md", "r") as fh:
#     long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="musev",  # used in pip install
    version="1.0.0",
    author="anchorxia, zkangchen",
    author_email="anchorxia@tencent.com, zkangchen@tencent.com",
    description="Package about human video creation",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/TMElyralab/MuseV",
    # include_package_data=True, # please edit MANIFEST.in
    # packages=find_packages(),  # used in import
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
)
