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

def test_from_images(file_list):
    imgs = [image.Image(fle) for fle in file_list]
    cap = capture.Capture(imgs)
    assert cap is not None
    assert len(cap.images) == len(file_list)

def test_from_filelist(file_list):
    cap = capture.Capture.from_filelist(file_list)
    assert cap is not None
    assert len(cap.images) == len(file_list)

def test_from_single_file(panel_image_name):
    cap = capture.Capture.from_file(panel_image_name)
    assert cap is not None

def test_append_single_file(file_list):
    cap = capture.Capture.from_file(file_list[0])
    assert len(cap.images) == 1
    cap.append_file(file_list[1])
    assert len(cap.images) == 2

def test_from_different_ids(bad_file_list):
    with pytest.raises(RuntimeError):
        cap = capture.Capture.from_filelist(bad_file_list)

def test_append_single(file_list):
    imgs = [image.Image(fle) for fle in file_list]
    cap = capture.Capture(imgs[0])
    assert len(cap.images) == 1
    for img in imgs[1:]:
        cap.append_image(img)
    assert len(cap.images) == 5

def test_append_list(file_list):
    imgs = [image.Image(fle) for fle in file_list]
    cap = capture.Capture(imgs[0])
    assert len(cap.images) == 1
    cap.append_images(imgs[1:])
    assert len(cap.images) == 5

def test_less_than(panel_image_name, flight_image_name):
    cap1 = capture.Capture.from_file(panel_image_name)
    cap2 = capture.Capture.from_file(flight_image_name)
    assert cap1 < cap2

def test_greater_than(panel_image_name, flight_image_name):
    cap1 = capture.Capture.from_file(panel_image_name)
    cap2 = capture.Capture.from_file(flight_image_name)
    assert cap2 > cap1

def test_equal(panel_image_name, panel_image_name_red):
    cap1 = capture.Capture.from_file(panel_image_name)
    cap2 = capture.Capture.from_file(panel_image_name_red)
    assert cap2 == cap1

def test_uct_time(panel_image_name):
    cap1 = capture.Capture.from_file(panel_image_name)
    assert cap1.utc_time().isoformat() == '2017-10-19T20:40:39.200174+00:00'

def test_location(panel_image_name):
    cap1 = capture.Capture.from_file(panel_image_name)
    loc = cap1.location()
    assert len(loc) == 3
    assert loc == (36.576096, -119.4352689, 101.861)

def test_dls_single_file(panel_image_name):
    cap1 = capture.Capture.from_file(panel_image_name)
    assert cap1.dls_present()
    assert cap1.dls_irradiance()[0] == pytest.approx(1.0101948, 1e-4)
    pose = cap1.dls_pose()
    assert len(pose) == 3
    assert pose[0] == pytest.approx(-3.070222992336269)
    assert pose[1] == pytest.approx(-0.18812839845718335)
    assert pose[2] == pytest.approx(-0.013387829297356699)

def test_dls_group(file_list):
    cap1 = capture.Capture.from_filelist(file_list)
    assert cap1.dls_present()
    irradiance = cap1.dls_irradiance()
    assert len(irradiance) == 5
    assert irradiance[0] == pytest.approx(1.0101948, 1e-4)
    pose = cap1.dls_pose()
    assert len(pose) == 3
    assert pose[0] == pytest.approx(-3.070222992336269)
    assert pose[1] == pytest.approx(-0.18812839845718335)
    assert pose[2] == pytest.approx(-0.013387829297356699)

def test_panel_radiance(file_list):
    cap = capture.Capture.from_filelist(file_list)
    rad = cap.panel_radiance()
    expected_rad = [0.17028382320603955, 
                    0.17940027272297152, 
                    0.1622172746785481, 
                    0.10647021248769974, 
                    0.13081077851565506]
    assert len(rad) == len(expected_rad)
    for i,_ in enumerate(expected_rad):
        assert rad[i] == pytest.approx(expected_rad[i], rel=0.001)

def test_panel_raw(file_list):
    cap = capture.Capture.from_filelist(file_list)
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

def test_panel_irradiance(file_list):
    cap = capture.Capture.from_filelist(file_list)
    panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67]
    rad = cap.panel_irradiance(panel_reflectance_by_band)
    expected_rad = [0.79845135523772681, 0.81681533164998943, 0.74944205649335915, 0.54833776619262586, 0.61336444894797537]
    assert len(rad) == len(expected_rad)
    for i,_ in enumerate(expected_rad):
        assert rad[i] == pytest.approx(expected_rad[i], rel=0.001)

def test_detect_panels_in_panel_image(file_list):
    cap = capture.Capture.from_filelist(file_list)
    assert cap.detect_panels() == 5
    assert cap.panels_in_all_expected_images() == True

def test_no_detect_panels_in_flight_image(non_panel_rededge_file_list):
    cap = capture.Capture.from_filelist(non_panel_rededge_file_list)
    assert cap.detect_panels() == 0
    assert cap.panels_in_all_expected_images() == False

def test_altum_images(non_panel_altum_file_list):
    imgs = [image.Image(fle) for fle in non_panel_altum_file_list]
    cap = capture.Capture(imgs)
    assert cap is not None
    assert len(cap.images) == len(non_panel_altum_file_list)

def test_altum_from_filelist(non_panel_altum_file_list):
    cap = capture.Capture.from_filelist(non_panel_altum_file_list)
    assert cap is not None
    assert len(cap.images) == len(non_panel_altum_file_list)

def test_altum_from_single_file(altum_flight_image_name):
    cap = capture.Capture.from_file(altum_flight_image_name)
    assert cap is not None

def test_altum_horizontal_irradiance(non_panel_altum_capture):
    assert non_panel_altum_capture.dls_present()
    good_irradiance = [1.222, 1.079, 0.914, 0.587, 0.715, 0.0]
    assert non_panel_altum_capture.dls_irradiance() == pytest.approx(good_irradiance, 1e-3)

def test_altum_panels(panel_altum_capture):
    assert panel_altum_capture.panels_in_all_expected_images() == True

@pytest.fixture()
def aligned_altum_capture(non_panel_altum_capture):
    non_panel_altum_capture.create_aligned_capture(img_type='radiance')
    return non_panel_altum_capture

def test_stack_export(aligned_altum_capture, tmpdir):
    pathstr = str(tmpdir.join('test_bgrent.tiff'))
    aligned_altum_capture.save_capture_as_stack(pathstr)
    assert os.path.exists(pathstr)
    if tmpdir.check():
        tmpdir.remove()

def test_rgb_jpg(aligned_altum_capture, tmpdir):
    pathstr = str(tmpdir.join('test_rgb.jpg'))
    aligned_altum_capture.save_capture_as_rgb(pathstr)
    assert os.path.exists(pathstr)
    if tmpdir.check():
        tmpdir.remove()

def test_rgb_png(aligned_altum_capture, tmpdir):
    pathstr = str(tmpdir.join('test_rgb.png'))
    aligned_altum_capture.save_capture_as_rgb(pathstr)
    assert os.path.exists(pathstr)
    if tmpdir.check():
        tmpdir.remove()

def test_rgb_jpg_decimation(aligned_altum_capture, tmpdir):
    import imageio
    decimations = [2,5,8]
    for decimation in decimations:
        pathstr = str(tmpdir.join('test_rgb_{}x.jpg'.format(decimation)))
        aligned_altum_capture.save_capture_as_rgb(pathstr, downsample=decimation)
        assert os.path.exists(pathstr)
        img = imageio.imread(pathstr)
        assert img.shape[0] == round(float(aligned_altum_capture.aligned_shape()[0])/float(decimation))
        assert img.shape[1] == round(float(aligned_altum_capture.aligned_shape()[1])/float(decimation))

    if tmpdir.check():
        tmpdir.remove()

def test_has_rig_relatives(non_panel_altum_capture):
    assert non_panel_altum_capture.has_rig_relatives() == True

def test_no_rig_relatives(non_panel_rededge_file_list):
    cap = capture.Capture.from_filelist(non_panel_rededge_file_list)
    assert cap.has_rig_relatives() == False

def test_panel_albedo(panel_altum_capture):
    panel_altum_capture.detect_panels()
    good_panel_albedo = [0.5282, 0.5274, 0.5263, 0.5246, 0.5258]
    assert panel_altum_capture.panel_albedo() == pytest.approx(good_panel_albedo, 1e-4)