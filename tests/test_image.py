#!/usr/bin/env python
# coding: utf-8
"""
Test image class

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

import numpy as np
import pytest

import micasense.image as image
import micasense.panel as panel


def test_load_image_metadata(img):
    assert img.meta is not None
    assert img.meta.band_index() == 0
    assert img.meta.camera_make() == 'MicaSense'
    assert img.meta.camera_model() == 'RedEdge-M'


def test_less_than(img, img2):
    assert img < img2


def test_greater_than(img, img2):
    assert img2 > img


def test_equal(img, img2):
    assert img == img


def test_not_equal(img, img2):
    assert img != img2


def test_load_image_raw(img):
    assert img.raw() is not None


def test_clear_image_data(img):
    assert img.undistorted(img.radiance()) is not None
    img.clear_image_data()
    assert img._Image__raw_image is None
    assert img._Image__intensity_image is None
    assert img._Image__radiance_image is None
    assert img._Image__reflectance_image is None
    assert img._Image__reflectance_irradiance is None
    assert img._Image__undistorted_source is None
    assert img._Image__undistorted_image is None


def test_reflectance(img):
    pan = panel.Panel(img)
    panel_reflectance = 0.50
    panel_irradiance = pan.irradiance_mean(panel_reflectance)
    reflectance_img = img.reflectance(panel_irradiance)
    ref_mean, _, _, _ = pan.region_stats(reflectance_img,
                                         pan.panel_corners())
    assert ref_mean == pytest.approx(panel_reflectance, 1e-2)


def test_size(img):
    assert img.size() == (1280, 960)


def test_pp_px(img):
    assert img.principal_point_px() == pytest.approx((657.402, 478.056), abs=0.1)


def test_cv2_camera_matrix(img):
    test_mat = [[1441.60555, 0, 657.402667],
                [0, 1441.60555, 478.056001],
                [0, 0, 1]]
    for idx, row in enumerate(img.cv2_camera_matrix()):
        assert row == pytest.approx(test_mat[idx], abs=0.1)


def test_altum_panel_image(panel_altum_image):
    assert panel_altum_image.size() == (2064, 1544)
    assert panel_altum_image.meta.camera_make() == "MicaSense"
    assert panel_altum_image.meta.camera_model() == "Altum"
    assert panel_altum_image.auto_calibration_image == True


def test_altum_flight_image(altum_flight_image):
    assert altum_flight_image.meta.camera_make() == "MicaSense"
    assert altum_flight_image.meta.camera_model() == "Altum"
    assert altum_flight_image.auto_calibration_image == False


def test_image_not_file(non_existant_file_name):
    with pytest.raises(OSError):
        image.Image(non_existant_file_name)


def test_altum_lwir_image(altum_lwir_image):
    assert altum_lwir_image.meta.band_name() == 'LWIR'
    assert altum_lwir_image.size() == (160, 120)
    assert altum_lwir_image.auto_calibration_image == False


def test_altum_image_horizontal_irradiance(altum_flight_image):
    assert altum_flight_image.dls_present
    solar_el = altum_flight_image.solar_elevation
    direct_irr = altum_flight_image.direct_irradiance
    scattered_irr = altum_flight_image.scattered_irradiance
    good_horiz_irradiance = direct_irr * np.sin(solar_el) + scattered_irr
    assert altum_flight_image.horizontal_irradiance == pytest.approx(good_horiz_irradiance, 1e-3)
