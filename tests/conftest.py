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

import glob
from pathlib import Path

import pytest

import micasense.capture as capture
import micasense.image as image
import micasense.metadata as metadata


@pytest.fixture()
def rededge_files_dir():
    return Path(__file__).parent.parent / 'data' / 'REDEDGE-MX'


@pytest.fixture()
def panel_rededge_file_list(rededge_files_dir: Path):
    return glob.glob(str(rededge_files_dir / 'IMG_0001_*.tif'))


@pytest.fixture()
def non_panel_rededge_file_list(rededge_files_dir: Path):
    return glob.glob(str(rededge_files_dir / 'IMG_0020_*.tif'))


@pytest.fixture()
def bad_file_list(rededge_files_dir: Path):
    file1 = str(rededge_files_dir / 'IMG_0020_1.tif')
    file2 = str(rededge_files_dir / 'IMG_0001_1.tif')
    return [file1, file2]


@pytest.fixture()
def panel_rededge_capture(panel_rededge_file_list):
    return capture.Capture.from_filelist(panel_rededge_file_list)


@pytest.fixture()
def non_panel_rededge_capture(non_panel_rededge_file_list):
    return capture.Capture.from_filelist(non_panel_rededge_file_list)


@pytest.fixture()
def img(rededge_files_dir: Path):
    return image.Image(str(rededge_files_dir / 'IMG_0001_1.tif'))


@pytest.fixture()
def img2(rededge_files_dir: Path):
    return image.Image(str(rededge_files_dir / 'IMG_0001_2.tif'))


@pytest.fixture()
def ten_band_files_dir():
    return Path(__file__).parent.parent / 'data' / 'REDEDGE-MX-DUAL'


@pytest.fixture()
def panel_10band_rededge_file_list(ten_band_files_dir: Path):
    return glob.glob(str(ten_band_files_dir / 'IMG_0000_*.tif'))


@pytest.fixture()
def flight_10band_rededge_file_list(ten_band_files_dir: Path):
    return glob.glob(str(ten_band_files_dir / 'IMG_0431_*.tif'))


@pytest.fixture()
def panel_10band_rededge_capture(panel_10band_rededge_file_list):
    return capture.Capture.from_filelist(panel_10band_rededge_file_list)


@pytest.fixture()
def flight_10band_rededge_capture(flight_10band_rededge_file_list):
    return capture.Capture.from_filelist(flight_10band_rededge_file_list)


@pytest.fixture()
def panel_image_name(rededge_files_dir: Path):
    return str(rededge_files_dir / 'IMG_0001_1.tif')


@pytest.fixture()
def panel_image_name_red(rededge_files_dir: Path):
    return str(rededge_files_dir / 'IMG_0001_2.tif')


@pytest.fixture()
def flight_image_name(rededge_files_dir: Path):
    return str(rededge_files_dir / 'IMG_0020_1.tif')


@pytest.fixture()
def altum_files_dir():
    return Path(__file__).parent.parent / 'data' / 'ALTUM'


@pytest.fixture()
def panel_altum_file_list(altum_files_dir):
    return glob.glob(str(altum_files_dir / 'IMG_0000_*.tif'))


@pytest.fixture()
def panel_altum_capture(panel_altum_file_list):
    imgs = [image.Image(fle) for fle in panel_altum_file_list]
    return capture.Capture(imgs)


@pytest.fixture()
def non_panel_altum_file_list(altum_files_dir: Path):
    return glob.glob(str(altum_files_dir / 'IMG_0021_*.tif'))


@pytest.fixture()
def non_panel_altum_capture(non_panel_altum_file_list):
    imgs = [image.Image(fle) for fle in non_panel_altum_file_list]
    return capture.Capture(imgs)


@pytest.fixture()
def altum_panel_image_name(altum_files_dir: Path):
    return str(altum_files_dir / 'IMG_0000_1.tif')


@pytest.fixture()
def altum_lwir_image_name(altum_files_dir: Path):
    return str(altum_files_dir / 'IMG_0000_6.tif')


@pytest.fixture()
def altum_flight_image_name(altum_files_dir: Path):
    return str(altum_files_dir / 'IMG_0021_1.tif')


@pytest.fixture()
def panel_altum_file_name(altum_files_dir):
    return str(altum_files_dir / 'IMG_0000_1.tif')


@pytest.fixture()
def panel_altum_image(panel_altum_file_name):
    return image.Image(panel_altum_file_name)


@pytest.fixture()
def altum_flight_image(altum_flight_image_name):
    return image.Image(altum_flight_image_name)


@pytest.fixture()
def non_existant_file_name(altum_files_dir: Path):
    return str(altum_files_dir / 'NOFILE.tif')


@pytest.fixture()
def altum_lwir_image(altum_files_dir: Path):
    return image.Image(str(altum_files_dir / 'IMG_0000_6.tif'))


@pytest.fixture()
def meta(rededge_files_dir: Path):
    return metadata.Metadata(str(rededge_files_dir / 'IMG_0001_1.tif'))


@pytest.fixture()
def meta_bad_exposure(rededge_files_dir: Path):
    return metadata.Metadata(str(rededge_files_dir / 'IMG_0020_1.tif'))


@pytest.fixture()
def meta_altum_dls2(altum_flight_image_name):
    return metadata.Metadata(altum_flight_image_name)
