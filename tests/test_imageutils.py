#!/usr/bin/env python
# coding: utf-8
"""
Test imageutils functionality

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
import cv2

import micasense.imageset as imageset
import micasense.capture as capture
import micasense.image as image
import micasense.imageutils as imageutils

from numpy import array
from numpy import float32

truth_warp_matrices = [array([[ 1.00523243e+00, -3.95214025e-03, -1.02620616e+01],
       [ 2.48925470e-03,  1.00346483e+00,  4.17114294e+01],
       [ 7.86653480e-07, -2.04642746e-06,  1.00000000e+00]]), array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]), array([[ 9.98681616e-01, -4.56952788e-03, -2.08530561e+00],
       [ 4.63740524e-03,  9.96737324e-01,  3.19011722e+01],
       [ 1.48930038e-06, -1.05003201e-06,  1.00000000e+00]]), array([[ 1.00149509e+00, -1.56960584e-03, -3.80940807e+00],
       [ 2.11523967e-03,  1.00222122e+00,  4.78563536e+01],
       [ 5.63914024e-07,  1.03391312e-07,  1.00000000e+00]]), array([[ 1.00305493e+00, -2.82497954e-03, -1.02199199e+01],
       [ 3.23661267e-03,  1.00139925e+00,  1.50062440e+01],
       [ 1.63746543e-06, -8.01922991e-07,  1.00000000e+00]]), array([[ 6.35209892e-02,  1.17877689e-05,  1.40322785e+01],
       [-4.56733969e-04,  6.35520044e-02,  1.15592432e+01],
       [-4.15804231e-06, -2.63551964e-06,  1.00000000e+00]])]

truth_image_sizes = [(2064, 1544), (2064, 1544), (2064, 1544), (2064, 1544), (2064, 1544), (160, 120)]

truth_lens_distortions = [array([-1.360334e-01,  2.374279e-01,  1.761687e-04,  2.373747e-04,
       -1.304408e-01]), array([-1.458199e-01,  2.681765e-01,  2.403470e-04, -6.698399e-04,
       -2.014740e-01]), array([-1.482020e-01,  2.494987e-01,  4.884159e-04, -1.989958e-04,
       -1.674770e-01]), array([-1.516688e-01,  2.483217e-01,  9.426709e-04,  1.109110e-04,
       -1.619578e-01]), array([-1.487282e-01,  2.477914e-01,  1.381469e-04,  5.226758e-04,
       -1.687072e-01]), array([-3.869621e-01,  4.784228e-01,  3.671945e-03,  4.130745e-04,
       -4.892879e-01])]

truth_camera_matrices = [array([[2.24343510e+03, 0.00000000e+00, 1.01295942e+03],
       [0.00000000e+00, 2.24343510e+03, 7.67547825e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), array([[2.23882015e+03, 0.00000000e+00, 1.01802029e+03],
       [0.00000000e+00, 2.23882015e+03, 7.30214492e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), array([[2.23464845e+03, 0.00000000e+00, 1.01673333e+03],
       [0.00000000e+00, 2.23464845e+03, 7.58373912e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), array([[2.24700170e+03, 0.00000000e+00, 1.01524638e+03],
       [0.00000000e+00, 2.24700170e+03, 7.80426086e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), array([[2.24418040e+03, 0.00000000e+00, 1.01343188e+03],
       [0.00000000e+00, 2.24418040e+03, 7.45014492e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), array([[162.63769993,   0.        ,  78.2788333 ],
       [  0.        , 162.63769993,  56.92766664],
       [  0.        ,   0.        ,   1.        ]])]

expected_dimensions = (21.0, 12.0, 2035.0, 1467.0)

def test_image_properties(non_panel_altum_capture):
    for i,image in enumerate(non_panel_altum_capture.images):
        assert(image.size() == pytest.approx(truth_image_sizes[i]))
        assert(image.cv2_distortion_coeff() == pytest.approx(truth_lens_distortions[i]))
        assert(image.cv2_camera_matrix() == pytest.approx(truth_camera_matrices[i]))

def test_warp_matrices(non_panel_altum_capture):
    warp_matrices = non_panel_altum_capture.get_warp_matrices()
    for index,warp_matrix in enumerate(warp_matrices):
        assert(warp_matrix == pytest.approx(truth_warp_matrices[index],rel=1e-2))

def test_cropping(non_panel_altum_capture):
    warp_matrices = non_panel_altum_capture.get_warp_matrices()
    cropped_dimensions,_ = imageutils.find_crop_bounds(non_panel_altum_capture,warp_matrices)
    assert(cropped_dimensions == pytest.approx(expected_dimensions,abs=1))
