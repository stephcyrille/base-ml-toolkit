#!/usr/bin/env python

import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
  long_description = fh.read()

setuptools.setup(
  version='1.0.0',
  name='mlbasetoolkit',
  author='Stephane Mebenga',
  author_email='stephcyril,sc@gmail.com',
  description='This is a ML toolkit that aims to help you when you have a ML heavy project',
  long_description = long_description,
  url='https://github.com/stephcyrille/base-ml-toolkit',
  classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  package_dir = {"": "src"},
  packages = setuptools.find_packages(where="src"),
  python_requires = ">=3.8"
)