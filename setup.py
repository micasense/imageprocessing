#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Parse the version from the main __init__.py
with open('micasense/__init__.py') as f:
    for line in f:
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

setup(name='micasense',
      version=version,
      description=u"Micasense Image Processing",
      author=u"MicaSense, Inc.",
      author_email='github@micasense.com',
      url='https://github.com/micasense/imageprocessing',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'requests',
          'numpy',
          'opencv-python',
          'gdal',
          'pysolar',
          'matplotlib',
          'scikit-image',
          'packaging',
          'pyexiftool<=0.4.13',
          'pytz',
          'pyzbar',
          'tqdm'
      ])
