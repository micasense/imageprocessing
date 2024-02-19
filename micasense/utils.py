#!/usr/bin/env python
# coding: utf-8
"""
MicaSense Image Processing Utilities
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

import cv2
import numpy as np


def raw_image_to_radiance(meta, image_raw):
    # get image dimensions
    image_raw = image_raw.T
    xDim = image_raw.shape[0]
    yDim = image_raw.shape[1]

    #  get radiometric calibration factors

    # radiometric sensitivity
    a1, a2, a3 = meta.get_item('XMP:RadiometricCalibration')
    a1 = float(a1)
    a2 = float(a2)
    a3 = float(a3)

    # get dark current pixel values
    # get number of stored values
    black_levels = [float(val) for val in meta.get_item('EXIF:BlackLevel').split(' ')]
    black_level = np.array(black_levels)
    dark_level = black_level.mean()

    # get exposure time & gain (gain = ISO/100)
    exposure_time = float(meta.get_item('EXIF:ExposureTime'))
    gain = float(meta.get_item('EXIF:ISOSpeed')) / 100.0

    # apply image correction methods to raw image
    # step 1 - row gradient correction, vignette & radiometric calibration:
    # compute the vignette map image
    V, x, y = vignette_map(meta, xDim, yDim)

    # row gradient correction
    R = 1.0 / (1.0 + a2 * y / exposure_time - a3 * y)

    # subtract the dark level and adjust for vignette and row gradient
    L = V * R * (image_raw - dark_level)

    # Floor any negative radiance's to zero (can happen due to noise around black_level)
    L[L < 0] = 0

    # L = np.round(L).astype(np.uint16)

    # apply the radiometric calibration - i.e. scale by the gain-exposure product and
    # multiply with the radiometric calibration coefficient
    # need to normalize by 2^16 for 16 bit images
    # because coefficients are scaled to work with input values of max 1.0
    bits_per_pixel = meta.get_item('EXIF:BitsPerSample')
    bit_depth_max = float(2 ** bits_per_pixel)
    radiance_image = L.astype(float) / (gain * exposure_time) * a1 / bit_depth_max

    # return both the radiance compensated image and the DN corrected image, for the
    # sake of the tutorial and visualization
    return radiance_image.T, L.T, V.T, R.T


def vignette_map(meta, x_dim, y_dim):
    # get coordinate grid across image, seem swapped because of transposed vignette
    x, y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    # meshgrid returns transposed arrays
    x = x.T
    y = y.T

    vignetting_center_size = meta.size('XMP:VignettingCenter')
    vignetting_polynomial_2d_size = meta.size('XMP:VignettingPolynomial2D')

    # if we have a radial poly
    if vignetting_center_size > 0:
        # get vignette center
        x_vignette = float(meta.get_item('XMP:VignettingCenter', 0))
        y_vignette = float(meta.get_item('XMP:VignettingCenter', 1))

        # get vignette polynomial
        nvignette_poly = meta.size('XMP:VignettingPolynomial')
        vignette_poly_list = [float(meta.get_item('XMP:VignettingPolynomial', i)) for i in range(nvignette_poly)]

        # reverse list and append 1., so that we can call with numpy polyval
        vignette_poly_list.reverse()
        vignette_poly_list.append(1.)
        vignette_poly = np.array(vignette_poly_list)

        # compute matrix of distances from image center
        r = np.hypot((x - x_vignette), (y - y_vignette))

        # compute the vignette polynomial for each distance - we divide by the polynomial so that the
        # corrected image is image_corrected = image_original * vignetteCorrection
        vignette = 1. / np.polyval(vignette_poly, r)
    elif vignetting_polynomial_2d_size > 0:  # 2d polynomial
        xv = x.T / x_dim
        yv = y.T / y_dim
        k = meta.vignette_polynomial2D()
        e = meta.vignette_polynomial2Dexponents()
        p2 = np.zeros_like(xv, dtype=float)
        for i, c in enumerate(k):
            ex = e[2 * i]
            ey = e[2 * i + 1]
            p2 += c * xv ** ex * yv ** ey
        vignette = (1. / p2).T
    return vignette, x, y


def focal_plane_resolution_px_per_mm(meta):
    fp_x_resolution = float(meta.get_item('EXIF:FocalPlaneXResolution'))
    fp_y_resolution = float(meta.get_item('EXIF:FocalPlaneYResolution'))
    return fp_x_resolution, fp_y_resolution


def focal_length_mm(meta):
    units = meta.get_item('XMP:PerspectiveFocalLengthUnits')
    if units == 'mm':
        local_focal_length_mm = float(meta.get_item('XMP:PerspectiveFocalLength'))
    else:
        focal_length_px = float(meta.get_item('XMP:PerspectiveFocalLength'))
        local_focal_length_mm = focal_length_px / focal_plane_resolution_px_per_mm(meta)[0]
    return local_focal_length_mm


def correct_lens_distortion(meta, image):
    # get lens distortion parameters
    ndistortion = meta.size('XMP:PerspectiveDistortion')
    distortion_parameters = np.array([float(meta.get_item('XMP:PerspectiveDistortion', i)) for i in range(ndistortion)])
    # get the two principal points
    pp = np.array(meta.get_item('XMP:PrincipalPoint').split(',')).astype(float)
    # values in pp are in [mm] and need to be rescaled to pixels
    focal_plane_x_resolution = float(meta.get_item('EXIF:FocalPlaneXResolution'))
    focal_plane_y_resolution = float(meta.get_item('EXIF:FocalPlaneYResolution'))

    cX = pp[0] * focal_plane_x_resolution
    cY = pp[1] * focal_plane_y_resolution
    fx = fy = focal_length_mm(meta) * focal_plane_x_resolution

    # apply perspective distortion
    h, w = image.shape

    # set up camera matrix for cv2
    cam_mat = np.zeros((3, 3))
    cam_mat[0, 0] = fx
    cam_mat[1, 1] = fy
    cam_mat[2, 2] = 1.0
    cam_mat[0, 2] = cX
    cam_mat[1, 2] = cY

    # set up distortion coefficients for cv2
    dist_coeffs = distortion_parameters[[0, 1, 3, 4, 2]]

    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(cam_mat, dist_coeffs, (w, h), 1)
    map1, map2 = cv2.initUndistortRectifyMap(cam_mat,
                                             dist_coeffs,
                                             np.eye(3),
                                             new_cam_mat,
                                             (w, h),
                                             cv2.CV_32F)  # cv2.CV_32F for 32 bit floats
    # compute the undistorted 16 bit image
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
