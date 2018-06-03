#!/usr/bin/env python
# coding: utf-8
"""
Test dls class

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
import numpy as np
import math

import micasense.dls as dls
import micasense.image as image

@pytest.fixture()
def img():
    image_path = os.path.join('data','0000SET','000',)
    return image.Image(os.path.join(image_path,'IMG_0000_1.tif'))

def test_sun_angle(img):
    if dls.havePysolar:
        sun_angle = dls.compute_sun_angle((img.latitude, img.longitude, img.altitude),
                                          (img.dls_yaw, img.dls_pitch, img.dls_roll),
                                          img.utc_time,
                                          np.array([0,0,-1]))
        assert sun_angle[0] == pytest.approx([-0.711, -0.247, -0.659], abs=0.001)
        assert sun_angle[1] == pytest.approx([-1.87482468e-01,  1.82720334e-05, -9.82267949e-01], abs=0.001)
        assert sun_angle[2] == pytest.approx(0.6754, abs=0.001)
        assert sun_angle[3] == pytest.approx(0.7193, abs=0.001)
        assert sun_angle[4] == pytest.approx(-0.334, abs=0.001)
    else:
        assert True

def test_fresnel():
    assert dls.fresnel(0.00) == pytest.approx(0.9416, abs=0.001)
    assert dls.fresnel(0.01) == pytest.approx(0.9416, abs=0.001)
    assert dls.fresnel(0.50) == pytest.approx(0.940, abs=0.001)
    assert dls.fresnel(0.99) == pytest.approx(0.903, abs=0.001)
    assert dls.fresnel(1.00) == pytest.approx(0.901, abs=0.001)

def test_get_orientation_zenith():
    pose = (math.radians(0),math.radians(0), math.radians(0))
    orientation = [0,0,-1]
    ned = dls.get_orientation(pose, orientation)
    assert ned == pytest.approx([0,0,-1])

def test_get_orientation_north():
    pose = (math.radians(0),math.radians(-90), math.radians(0))
    orientation = [0,0,-1]
    ned = dls.get_orientation(pose, orientation)
    assert ned == pytest.approx([1,0,0])

def test_get_orientation_east():
    pose = (math.radians(90),math.radians(-90), math.radians(0))
    orientation = [0,0,-1]
    ned = dls.get_orientation(pose, orientation)
    assert ned == pytest.approx([0,1,0])

def test_get_orientation_south():
    pose = (math.radians(0),math.radians(90), math.radians(0))
    orientation = [0,0,-1]
    ned = dls.get_orientation(pose, orientation)
    assert ned == pytest.approx([-1,0,0])

def test_get_orientation_south2():
    pose = (math.radians(180),math.radians(-90), math.radians(0))
    orientation = [0,0,-1]
    ned = dls.get_orientation(pose, orientation)
    assert ned == pytest.approx([-1,0,0])

def test_get_orientation_west():
    pose = (math.radians(-90),math.radians(-90), math.radians(0))
    orientation = [0,0,-1]
    ned = dls.get_orientation(pose, orientation)
    assert ned == pytest.approx([0,-1,0])