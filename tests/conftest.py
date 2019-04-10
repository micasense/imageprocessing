#!/usr/bin/env python
# coding: utf-8
"""
Test image class

Copyright 2019 MicaSense, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import pytest
import os, glob

import micasense.capture as capture
import micasense.image as image
import micasense.metadata as metadata

@pytest.fixture()
def files_dir():
    return os.path.join('data', '0000SET', '000')

@pytest.fixture()
def altum_files_dir():
    return os.path.join('data', 'ALTUM1SET', '000')

@pytest.fixture()
def file_list(files_dir):
    return glob.glob(os.path.join(files_dir, 'IMG_0000_*.tif'))

@pytest.fixture()
def non_panel_rededge_file_list(files_dir):
    return glob.glob(os.path.join(files_dir, 'IMG_0001_*.tif'))

@pytest.fixture()
def bad_file_list(files_dir):
    file1 = os.path.join(files_dir, 'IMG_0000_1.tif')
    file2 = os.path.join(files_dir, 'IMG_0001_1.tif')
    return [file1, file2]

@pytest.fixture()
def panel_altum_file_list(altum_files_dir):
    return glob.glob(os.path.join(altum_files_dir, 'IMG_0000_*.tif'))

@pytest.fixture()
def panel_altum_capture(panel_altum_file_list):
    imgs = [image.Image(fle) for fle in panel_altum_file_list]
    return capture.Capture(imgs)

@pytest.fixture()
def non_panel_altum_file_list(altum_files_dir):
    return glob.glob(os.path.join(altum_files_dir, 'IMG_0008_*.tif'))

@pytest.fixture()
def non_panel_altum_capture(non_panel_altum_file_list):
    imgs = [image.Image(fle) for fle in non_panel_altum_file_list]
    return capture.Capture(imgs)

@pytest.fixture()
def panel_image_name():
    image_path = os.path.join('data', '0000SET', '000')
    return os.path.join(image_path, 'IMG_0000_1.tif')

@pytest.fixture()
def panel_image_name_red():
    image_path = os.path.join('data', '0000SET', '000')
    return os.path.join(image_path, 'IMG_0000_2.tif')

@pytest.fixture()
def flight_image_name():
    image_path = os.path.join('data', '0000SET', '000')
    return os.path.join(image_path, 'IMG_0001_1.tif')

@pytest.fixture()
def altum_panel_image_name(altum_files_dir):
    return os.path.join(altum_files_dir, 'IMG_0000_1.tif')

@pytest.fixture()
def altum_lwir_image_name(altum_files_dir):
    return os.path.join(altum_files_dir, 'IMG_0000_6.tif')

@pytest.fixture()
def altum_flight_image_name(altum_files_dir):
    return os.path.join(altum_files_dir, 'IMG_0008_1.tif')

@pytest.fixture()
def img(files_dir):
    return image.Image(os.path.join(files_dir,'IMG_0000_1.tif'))

@pytest.fixture()
def img2(files_dir):
    return image.Image(os.path.join(files_dir,'IMG_0000_2.tif'))

@pytest.fixture()
def panel_altum_file_name(altum_files_dir):
    return os.path.join(altum_files_dir, 'IMG_0000_1.tif')

@pytest.fixture()
def panel_altum_image(panel_altum_file_name):
    return image.Image(panel_altum_file_name)

@pytest.fixture()
def altum_flight_image(altum_flight_image_name):
    return image.Image(altum_flight_image_name)

@pytest.fixture()
def non_existant_file_name(altum_files_dir):
    return os.path.join(altum_files_dir, 'NOFILE.tif')

@pytest.fixture()
def altum_lwir_image(altum_files_dir):
    return image.Image(os.path.join(altum_files_dir, 'IMG_0000_6.tif'))

@pytest.fixture()
def meta():
    image_path = os.path.join('data', '0000SET', '000')
    return metadata.Metadata(os.path.join(image_path, 'IMG_0000_1.tif'))

@pytest.fixture()
def meta_v3():
    image_path = os.path.join('data', '0001SET', '000')
    return metadata.Metadata(os.path.join(image_path, 'IMG_0002_4.tif'))

@pytest.fixture()
def meta_bad_exposure():
    image_path = os.path.join('data', '0001SET', '000')
    return metadata.Metadata(os.path.join(image_path, 'IMG_0003_1.tif'))

@pytest.fixture()
def meta_altum_dls2(altum_flight_image_name):
    return metadata.Metadata(altum_flight_image_name)

@pytest.fixture()
def bad_dls2_horiz_irr_image():
    image_path = os.path.join('data', 'ALTUM0SET', '000')
    return image.Image(os.path.join(image_path, 'IMG_0000_1.tif'))
