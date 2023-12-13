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


def test_load_image_metadata(meta):
    assert meta is not None


def test_band_index(meta):
    assert meta.band_index() == 0


def test_camera_make(meta):
    assert meta.camera_make() == 'MicaSense'


def test_camera_model(meta):
    assert meta.camera_model() == 'RedEdge-M'


def test_flight_id(meta):
    assert meta.flight_id() == 'ePdxBBmSitgkTdpwZiM9'


def test_capture_id(meta):
    assert meta.capture_id() == 'Rb0pibHa08uHJwrTjf8Y'


def test_black_level(meta):
    assert meta.black_level() == 4800.0


def test_focal_length_mm(meta):
    assert meta.focal_length_mm() == pytest.approx(5.40602081)


def test_fp_resolution(meta):
    assert meta.focal_plane_resolution_px_per_mm() == pytest.approx([266.666667, 266.666667])


def test_utc_time(meta):
    utc_time = meta.utc_time()
    assert utc_time is not None
    assert utc_time.strftime('%Y-%m-%d %H:%M:%S.%f') == '2022-04-06 18:50:25.983430'


def test_position(meta):
    assert meta.position() == pytest.approx((47.7036143, -122.1414373, 6.728))


def test_dls_present(meta):
    assert meta.dls_present() == True


def test_metadata_size(meta):
    assert meta.size('XMP:RadiometricCalibration') == 3


def test_center_wavelength(meta):
    assert meta.center_wavelength() == 475


def test_vignette_center(meta):
    assert meta.vignette_center() == pytest.approx([623.6301, 470.2927], abs=0.001)


def test_vignette_polynomial(meta):
    expected_poly = [1.001285e-06, 5.61421e-07, -5.962064e-09, 1.862037e-11, -1.4703936738578638e-14,
                     7.334097230810222e-18]
    assert meta.vignette_polynomial() == pytest.approx(expected_poly, rel=0.001)


def test_principal_point_mm(meta):
    assert meta.principal_point() == pytest.approx([2.46526, 1.79271])


def test_distortion_parameters(meta):
    expected_params = [-0.1058375, 0.2199191, -0.2010044, 0.0007368542, -0.0004963633]
    assert meta.distortion_parameters() == pytest.approx(expected_params, rel=0.001)


def test_bits_per_pixel(meta):
    assert meta.bits_per_pixel() == 16


def test_dark_pixels(meta):
    assert meta.dark_pixels() == pytest.approx(5045.25)


def test_gain(meta):
    assert meta.gain() == 1


def test_firmware_version(meta):
    assert meta.firmware_version() == "v7.5.0-beta6"


def test_dls_irradiance(meta):
    assert meta.spectral_irradiance() == pytest.approx(0.8821, abs=0.0001)


def test_dls_pose(meta):
    assert meta.dls_pose() == pytest.approx((-2.0091497634122724, 0.018554597483870183, 0.031269217556393974),
                                            abs=0.001)


def test_good_exposure(meta):
    assert meta.exposure() == pytest.approx(0.000135)


def test_bad_exposure_time(meta_bad_exposure):
    assert meta_bad_exposure.exposure() == pytest.approx(247e-6, abs=1e-3)


def test_dls_present_dls2(meta_altum_dls2):
    assert meta_altum_dls2.dls_present() == True


def test_dls2_scale_factor(meta_altum_dls2):
    assert meta_altum_dls2.irradiance_scale_factor() == pytest.approx(0.01)


def test_horizontal_irradiance_valid_altum(meta_altum_dls2):
    assert meta_altum_dls2.horizontal_irradiance_valid() == True
