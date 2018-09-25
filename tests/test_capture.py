#!/usr/bin/env python
# coding: utf-8
"""
Test capture class

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

import micasense.capture as capture
import micasense.image as image

@pytest.fixture()
def files_dir():
    return os.path.join('data', '0000SET', '000')

@pytest.fixture()
def file_list():
    return glob.glob(os.path.join(files_dir(), 'IMG_0000_*.tif'))

@pytest.fixture()
def bad_file_list():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    file2 = os.path.join(files_dir(), 'IMG_0001_1.tif')
    return [file1, file2]

def test_from_images():
    imgs = [image.Image(fle) for fle in file_list()]
    cap = capture.Capture(imgs)
    assert cap is not None
    assert len(cap.images) == len(file_list())

def test_from_filelist():
    cap = capture.Capture.from_filelist(file_list())
    assert cap is not None
    assert len(cap.images) == len(file_list())

def test_from_single_file():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    cap = capture.Capture.from_file(file1)
    assert cap is not None

def test_from_different_ids():
    with pytest.raises(RuntimeError):
        cap = capture.Capture.from_filelist(bad_file_list())

def test_append_single():
    imgs = [image.Image(fle) for fle in file_list()]
    cap = capture.Capture(imgs[0])
    assert len(cap.images) == 1
    for img in imgs[1:]:
        cap.append_image(img)
    assert len(cap.images) == 5

def test_append_list():
    imgs = [image.Image(fle) for fle in file_list()]
    cap = capture.Capture(imgs[0])
    assert len(cap.images) == 1
    cap.append_images(imgs[1:])
    assert len(cap.images) == 5

def test_less_than():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    file2 = os.path.join(files_dir(), 'IMG_0001_1.tif')
    cap1 = capture.Capture.from_file(file1)
    cap2 = capture.Capture.from_file(file2)
    assert cap1 < cap2

def test_greater_than():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    file2 = os.path.join(files_dir(), 'IMG_0001_1.tif')
    cap1 = capture.Capture.from_file(file1)
    cap2 = capture.Capture.from_file(file2)
    assert cap2 > cap1

def test_equal():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    file2 = os.path.join(files_dir(), 'IMG_0000_3.tif')
    cap1 = capture.Capture.from_file(file1)
    cap2 = capture.Capture.from_file(file2)
    assert cap2 == cap1

def test_uct_time():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    cap1 = capture.Capture.from_file(file1)
    assert cap1.utc_time().isoformat() == '2017-10-19T20:40:39.200174+00:00'

def test_location():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    cap1 = capture.Capture.from_file(file1)
    loc = cap1.location()
    assert len(loc) == 3
    assert loc == (36.576096, -119.4352689, 101.861)

def test_dls_single_file():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    cap1 = capture.Capture.from_file(file1)
    assert cap1.dls_present()
    assert cap1.dls_irradiance()[0] == pytest.approx(1.010101948, 1e-4)
    pose = cap1.dls_pose()
    assert len(pose) == 3
    assert pose[0] == pytest.approx(-3.070222992336269)
    assert pose[1] == pytest.approx(-0.18812839845718335)
    assert pose[2] == pytest.approx(-0.013387829297356699)

def test_dls_group():
    cap1 = capture.Capture.from_filelist(file_list())
    assert cap1.dls_present()
    irradiance = cap1.dls_irradiance()
    assert len(irradiance) == 5
    assert irradiance[0] == pytest.approx(1.010101948, 1e-4)
    pose = cap1.dls_pose()
    assert len(pose) == 3
    assert pose[0] == pytest.approx(-3.070222992336269)
    assert pose[1] == pytest.approx(-0.18812839845718335)
    assert pose[2] == pytest.approx(-0.013387829297356699)

def test_panel_radiance():
    cap = capture.Capture.from_filelist(file_list())
    rad = cap.panel_radiance()
    expected_rad = [0.17028382320603955, 
                    0.17940027272297152, 
                    0.1622172746785481, 
                    0.10647021248769974, 
                    0.13081077851565506]
    assert len(rad) == len(expected_rad)
    for i,_ in enumerate(expected_rad):
        assert rad[i] == pytest.approx(expected_rad[i], rel=0.001)

def test_panel_raw():
    cap = capture.Capture.from_filelist(file_list())
    raw = cap.panel_raw()
    print(raw)
    expected_raw = [45406.753893482026, 
                    46924.919447148139, 
                    53240.810340812051, 
                    56187.417482757308, 
                    54479.170371812339]
    assert len(raw) == len(expected_raw)
    for i,_ in enumerate(expected_raw):
        assert raw[i] == pytest.approx(expected_raw[i], rel=0.001)

def test_panel_irradiance():
    cap = capture.Capture.from_filelist(file_list())
    panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67]
    rad = cap.panel_irradiance(panel_reflectance_by_band)
    expected_rad = [0.79845135523772681, 0.81681533164998943, 0.74944205649335915, 0.54833776619262586, 0.61336444894797537]
    assert len(rad) == len(expected_rad)
    for i,_ in enumerate(expected_rad):
        assert rad[i] == pytest.approx(expected_rad[i], rel=0.001)