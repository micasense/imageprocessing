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

import pytest
import os, glob
import math
import numpy as np

import micasense.image as image
import micasense.panel as panel

@pytest.fixture()
def img():
    image_path = os.path.join('data','0000SET','000',)
    return image.Image(os.path.join(image_path,'IMG_0000_1.tif'))

@pytest.fixture()
def img2():
    image_path = os.path.join('data','0000SET','000',)
    return image.Image(os.path.join(image_path,'IMG_0000_2.tif'))

def test_load_image_metadata(img):
    assert img.meta is not None
    assert img.meta.band_index() == 0
    assert img.meta.camera_make() == 'MicaSense'
    assert img.meta.camera_model() == 'RedEdge'

def test_less_than(img,img2):
    assert img < img2

def test_greater_than(img,img2):
    assert img2 > img

def test_equal(img,img2):
    assert img == img

def test_not_equal(img,img2):
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
    panel_reflectance = 0.5
    panel_irradiance = pan.irradiance_mean(panel_reflectance)
    reflectance_img = img.reflectance(panel_irradiance)
    ref_mean, _, _, _ = pan.region_stats(reflectance_img,
                                         pan.panel_corners())
    assert ref_mean == pytest.approx(panel_reflectance, 1e-4)

def test_size(img):
    assert img.size() == (1280,960)

def test_pp_px(img):
    assert img.principal_point_px() == pytest.approx((627.6, 479.9), abs=0.1)

def test_cv2_camera_matrix(img):
    test_mat = [[1449.4,    0.0, 627.6],
                [   0.0, 1449.4, 479.9],
                [   0.0,    0.0,   1.0]]
    for idx, row in enumerate(img.cv2_camera_matrix()):
        assert row == pytest.approx(test_mat[idx], abs=0.1)
