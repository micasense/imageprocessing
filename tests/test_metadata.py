#!/usr/bin/env python
# coding: utf-8
"""
Test metadata class

Copyright 2017 MicaSense, Inc.

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

import micasense.metadata as metadata

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


def test_load_image_metadata(meta):
    assert meta is not None

def test_band_index(meta):
    assert meta.band_index() == 0

def test_camera_make(meta):
    assert meta.camera_make() == 'MicaSense'

def test_camera_model(meta):
    assert meta.camera_model() == 'RedEdge'

def test_flight_id(meta):
    assert meta.flight_id() == 'NtLNbVIdowuCaWYbg3ck'

def test_capture_id(meta):
    assert meta.capture_id() == '5v25BtsZg3BQBhVH7Iaz'

def test_black_level(meta):
    assert meta.black_level() == 4800.0

def test_focal_length_mm(meta):
    assert meta.focal_length_mm() == pytest.approx(5.43509341)

def test_focal_length_mm_v3(meta_v3):
    assert meta_v3.focal_length_mm() == pytest.approx(5.45221099)

def test_fp_resolution(meta):
    assert meta.focal_plane_resolution_px_per_mm() == pytest.approx([266.666667,266.666667])

def test_fp_resolution_v3(meta_v3):
    assert meta_v3.focal_plane_resolution_px_per_mm() == pytest.approx([266.666667,266.666667])

def test_utc_time(meta):
    utc_time =  meta.utc_time()
    assert utc_time is not None
    assert utc_time.strftime('%Y-%m-%d %H:%M:%S.%f') == '2017-10-19 20:40:39.200174'

def test_utc_time_v3(meta_v3):
    utc_time =  meta_v3.utc_time()
    assert utc_time is not None
    assert utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:] == '2018-04-10 10:52:30.866550'

def test_position(meta):
    assert meta.position() == pytest.approx((36.576096, -119.4352689, 101.861))

def test_dls_present(meta):
    assert meta.dls_present() == True

def test_metadata_size(meta):
    assert meta.size('XMP:RadiometricCalibration') == 3

def test_center_wavelength(meta):
    assert meta.center_wavelength() == 475

def test_vignette_center(meta):
    assert meta.vignette_center() == pytest.approx([676.703, 480.445], abs=0.001)

def test_vignette_polynomial(meta):
    expected_poly = [-3.188190987533484e-05, 1.1380741452056501e-07, -2.7776829778142425e-09, 9.981184981301047e-12, -1.4703936738578638e-14, 7.334097230810222e-18]
    assert meta.vignette_polynomial() == pytest.approx(expected_poly, rel=0.001)

def test_principal_point_mm(meta):
    assert meta.principal_point() == pytest.approx([2.35363, 1.79947])

def test_distortion_parameters(meta):
    expected_params = [-0.09679655532374383, 0.14041893470790068, -0.022980842634993275, 0.0002758383774216635, 0.0006600729536460939]
    assert meta.distortion_parameters() == pytest.approx(expected_params, rel=0.001)

def test_bits_per_pixel(meta):
    assert meta.bits_per_pixel() == 16

def test_dark_pixels(meta):
    assert meta.dark_pixels() == pytest.approx(5071.5)

def test_gain(meta):
    assert meta.gain() == 1

def test_firmware_version(meta):
    assert meta.firmware_version() == "v2.1.2-34-g05e37eb-local"

def test_firmware_version_v3(meta_v3):
    assert meta_v3.firmware_version() == "v3.3.0"

def test_dls_irradiance(meta):
    assert meta.dls_irradiance() == pytest.approx(1.0848, abs=0.0001)

def test_dls_pose(meta):
    assert meta.dls_pose() == pytest.approx((-3.070, -0.188, -0.013), abs=0.001)

def test_good_exposure(meta):
    assert meta.exposure() == pytest.approx(0.0004725)

def test_good_exposure_v3(meta_v3):
    assert meta_v3.exposure() == pytest.approx(0.00171)

def test_bad_exposure_time(meta_bad_exposure):
    assert meta_bad_exposure.exposure() == pytest.approx(247e-6, abs=1e-3)