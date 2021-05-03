#!/usr/bin/env python
# coding: utf-8
"""
Misc. image processing utilities

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
import os
import numpy as np
import multiprocessing
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank, gaussian
from skimage.util import img_as_ubyte

def normalize(im, min=None, max=None):
    width, height = im.shape
    norm = np.zeros((width, height), dtype=np.float32)
    if min is not None and max is not None:
        norm = (im - min) / (max-min)
    else:
        cv2.normalize(im, dst=norm, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm[norm<0.0] = 0.0
    norm[norm>1.0] = 1.0
    return norm

def local_normalize(im):
    norm = img_as_ubyte(normalize(im)) # TODO: mainly using this as a type conversion, but it's expensive
    width, _ = im.shape
    disksize = int(width/5)
    if disksize % 2 == 0:
        disksize = disksize + 1
    selem = disk(disksize)
    norm2 = rank.equalize(norm, selem=selem)
    return norm2

def gradient(im, ksize=5):
    im = local_normalize(im)
    # im = normalize(im)
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=ksize)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=ksize)
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

def relatives_ref_band(capture):
    for img in capture.images:
        if img.rig_xy_offset_in_px() == (0,0):
            return img.band_index()
    return (0)

def translation_from_ref(capture, band, ref=4):
    x,y = capture.images[band].rig_xy_offset_in_px()
    rx,ry = capture.images[ref].rig_xy_offset_in_px()
    return

def align(pair):
    """ Determine an alignment matrix between two images
    @input:
    Dictionary of the following form:
    {
        'warp_mode':  cv2.MOTION_* (MOTION_AFFINE, MOTION_HOMOGRAPHY)
        'max_iterations': Maximum number of solver iterations
        'epsilon_threshold': Solver stopping threshold
        'ref_index': index of reference image
        'match_index': index of image to match to reference
    }
    @returns:
    Dictionary of the following form:
    {
        'ref_index': index of reference image
        'match_index': index of image to match to reference
        'warp_matrix': transformation matrix to use to map match image to reference image frame
    }

    Major props to Alexander Reynolds ( https://stackoverflow.com/users/5087436/alexander-reynolds ) for his
    insight into the pyramided matching process found at
    https://stackoverflow.com/questions/45997891/cv2-motion-euclidean-for-the-warp-mode-in-ecc-image-alignment-method

    """
    warp_mode = pair['warp_mode']
    max_iterations = pair['max_iterations']
    epsilon_threshold = pair['epsilon_threshold']
    ref_index = pair['ref_index']
    match_index = pair['match_index']
    translations = pair['translations']

    # Initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # warp_matrix = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        warp_matrix = pair['warp_matrix_init']
    else:
        # warp_matrix = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        warp_matrix = np.array([[1,0,translations[1]],[0,1,translations[0]]], dtype=np.float32)

    w = pair['ref_image'].shape[1]

    if pair['pyramid_levels'] is None:
        nol =  int(w / (1280/3)) - 1
    else:
        nol = pair['pyramid_levels']

    if pair['debug']:
        print("number of pyramid levels: {}".format(nol))

    warp_matrix[0][2] /= (2**nol)
    warp_matrix[1][2] /= (2**nol)

    if ref_index != match_index:

        show_debug_images = pair['debug']
        # construct grayscale pyramid
        gray1 = pair['ref_image']
        gray2 = pair['match_image']
        if gray2.shape[0] < gray1.shape[0]:
            cv2.resize(gray2, None, fx=gray1.shape[0]/gray2.shape[0], fy=gray1.shape[0]/gray2.shape[0],
                                        interpolation=cv2.INTER_AREA)
        gray1_pyr = [gray1]
        gray2_pyr = [gray2]

        for level in range(nol):
            gray1_pyr[0] = gaussian(normalize(gray1_pyr[0]))
            gray1_pyr.insert(0, cv2.resize(gray1_pyr[0], None, fx=1/2, fy=1/2,
                                        interpolation=cv2.INTER_AREA))
            gray2_pyr[0] = gaussian(normalize(gray2_pyr[0]))
            gray2_pyr.insert(0, cv2.resize(gray2_pyr[0], None, fx=1/2, fy=1/2,
                                        interpolation=cv2.INTER_AREA))

        # Terminate the optimizer if either the max iterations or the threshold are reached
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon_threshold)
        # run pyramid ECC
        for level in range(nol+1):
            grad1 = gradient(gray1_pyr[level])
            grad2 = gradient(gray2_pyr[level])

            if show_debug_images:
                import micasense.plotutils as plotutils
                plotutils.plotwithcolorbar(gray1_pyr[level], "ref level {}".format(level))
                plotutils.plotwithcolorbar(gray2_pyr[level], "match level {}".format(level))
                plotutils.plotwithcolorbar(grad1, "ref grad level {}".format(level))
                plotutils.plotwithcolorbar(grad2, "match grad level {}".format(level))
                print("Starting warp for level {} is:\n {}".format(level,warp_matrix))

            try:
                cc, warp_matrix = cv2.findTransformECC(grad1, grad2, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
            except TypeError:
                cc, warp_matrix = cv2.findTransformECC(grad1, grad2, warp_matrix, warp_mode, criteria)
            
            if show_debug_images:
                print("Warp after alignment level {} is \n{}".format(level,warp_matrix))

            if level != nol:  # scale up only the offset by a factor of 2 for the next (larger image) pyramid level
                if warp_mode == cv2.MOTION_HOMOGRAPHY:
                    warp_matrix = warp_matrix * np.array([[1,1,2],[1,1,2],[0.5,0.5,1]], dtype=np.float32)
                else:
                    warp_matrix = warp_matrix * np.array([[1,1,2],[1,1,2]], dtype=np.float32)

                

    return {'ref_index': pair['ref_index'],
            'match_index': pair['match_index'],
            'warp_matrix': warp_matrix }

def default_warp_matrix(warp_mode):
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
    else:
        return np.array([[1,0,0],[0,1,0]], dtype=np.float32)

def align_capture(capture, ref_index=1, warp_mode=cv2.MOTION_HOMOGRAPHY, max_iterations=2500, epsilon_threshold=1e-9, multithreaded=True, debug=False, pyramid_levels = None):
    '''Align images in a capture using openCV
    MOTION_TRANSLATION sets a translational motion model; warpMatrix is 2x3 with the first 2x2 part being the unity matrix and the rest two parameters being estimated.
    MOTION_EUCLIDEAN sets a Euclidean (rigid) transformation as motion model; three parameters are estimated; warpMatrix is 2x3.
    MOTION_AFFINE sets an affine motion model (DEFAULT); six parameters are estimated; warpMatrix is 2x3.
    MOTION_HOMOGRAPHY sets a homography as a motion model; eight parameters are estimated;`warpMatrix` is 3x3.
    best results will be AFFINE and HOMOGRAPHY, at the expense of speed
    '''
    # Match other bands to this reference image (index into capture.images[])
    ref_img = capture.images[ref_index].undistorted(capture.images[ref_index].radiance()).astype('float32')
    
    if capture.has_rig_relatives():
        warp_matrices_init = capture.get_warp_matrices(ref_index=ref_index)
    else:
        warp_matrices_init = [default_warp_matrix(warp_mode)]*len(capture.images)
    
    alignment_pairs = []
    for i,img in enumerate(capture.images):
        if img.rig_relatives is not None:
            translations = img.rig_xy_offset_in_px()
        else:
            translations = (0,0)
        if img.band_name != 'LWIR':
            alignment_pairs.append({'warp_mode': warp_mode,
                                    'max_iterations': max_iterations,
                                    'epsilon_threshold': epsilon_threshold,
                                    'ref_index':ref_index,
                                    'ref_image': ref_img,
                                    'match_index':i,
                                    'match_image':img.undistorted(img.radiance()).astype('float32'),
                                    'translations': translations,
                                    'warp_matrix_init': np.array(warp_matrices_init[i], dtype=np.float32),
                                    'debug': debug,
                                    'pyramid_levels': pyramid_levels})
    warp_matrices = [None]*len(alignment_pairs)

    #required to work across linux/mac/windows, see https://stackoverflow.com/questions/47852237
    if multithreaded and multiprocessing.get_start_method() != 'spawn':
        try:
            multiprocessing.set_start_method('spawn',force=True)
        except ValueError:
            multithreaded = False

    if(multithreaded):
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        for _,mat in enumerate(pool.imap_unordered(align, alignment_pairs)):
            warp_matrices[mat['match_index']] = mat['warp_matrix']
            print("Finished aligning band {}".format(mat['match_index']))
        pool.close()
        pool.join()
    else:
        # Single-threaded alternative
        for pair in alignment_pairs:
            mat = align(pair)
            warp_matrices[mat['match_index']] = mat['warp_matrix']
            print("Finished aligning band {}".format(mat['match_index']))

    if capture.images[-1].band_name == 'LWIR':
        img = capture.images[-1]
        alignment_pairs.append({'warp_mode': warp_mode,
                                'max_iterations': max_iterations,
                                'epsilon_threshold': epsilon_threshold,
                                'ref_index':ref_index,
                                'ref_image': ref_img,
                                'match_index':img.band_index,
                                'match_image':img.undistorted(img.radiance()).astype('float32'),
                                'translations': translations,
                                'debug': debug})
        warp_matrices.append(capture.get_warp_matrices(ref_index)[-1])
    return warp_matrices, alignment_pairs

#apply homography to create an aligned stack
def aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, match_index, img_type = 'reflectance',interpolation_mode=cv2.INTER_LANCZOS4):
    width, height = capture.images[0].size()

    im_aligned = np.zeros((height,width,len(warp_matrices)), dtype=np.float32 )

    for i in range(0,len(warp_matrices)):
        if img_type == 'reflectance':
            img = capture.images[i].undistorted_reflectance()
        else:
            img = capture.images[i].undistorted_radiance()

        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            im_aligned[:,:,i] = cv2.warpAffine(img,
                                            warp_matrices[i],
                                            (width,height),
                                            flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
        else:
            im_aligned[:,:,i] = cv2.warpPerspective(img,
                                                warp_matrices[i],
                                                (width,height),
                                                flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
    (left, top, w, h) = tuple(int(i) for i in cropped_dimensions)
    im_cropped = im_aligned[top:top+h, left:left+w][:]

    return im_cropped

class BoundPoint(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return "(%f, %f)" % (self.x, self.y)

    def __repr__(self):
        return self.__str__()

class Bounds(object):
    def __init__(self):
        arbitrary_large_value = 100000000
        self.max = BoundPoint(-arbitrary_large_value, -arbitrary_large_value)
        self.min = BoundPoint(arbitrary_large_value, arbitrary_large_value)

    def __str__(self):
        return "Bounds min: %s, max: %s" % (str(self.min), str(self.max))

    def __repr__(self):
        return self.__str__()

def find_crop_bounds(capture,registration_transforms,warp_mode=cv2.MOTION_HOMOGRAPHY):
    """Compute the crop rectangle to be applied to a set of images after
    registration such that no pixel in the resulting stack of images will
    include a blank value for any of the bands

    Args:

    capture- an image capture

    registration_transforms - a list of affine transforms applied to
    register the image. It is required.

    returns the left,top,w,h coordinates  of the smallest overlapping rectangle
    and the mapped edges of the images
    """
    image_sizes = [image.size() for image in capture.images]
    lens_distortions = [image.cv2_distortion_coeff() for image in capture.images]
    camera_matrices =  [image.cv2_camera_matrix() for image in capture.images]

    bounds = [get_inner_rect(s, a, d, c,warp_mode=warp_mode)[0] for s, a, d, c in zip(image_sizes,registration_transforms, lens_distortions, camera_matrices)]
    edges = [get_inner_rect(s, a, d, c,warp_mode=warp_mode)[1] for s, a, d, c in zip(image_sizes,registration_transforms, lens_distortions, camera_matrices)]
    combined_bounds = get_combined_bounds(bounds, image_sizes[0])

    left = np.ceil(combined_bounds.min.x)
    top = np.ceil(combined_bounds.min.y)
    width = np.floor(combined_bounds.max.x - combined_bounds.min.x)
    height = np.floor(combined_bounds.max.y - combined_bounds.min.y)
    return (left, top, width, height),edges

def get_inner_rect(image_size, affine, distortion_coeffs, camera_matrix,warp_mode=cv2.MOTION_HOMOGRAPHY):
    w = image_size[0]
    h = image_size[1]

    left_edge = np.array([np.ones(h)*0, np.arange(0, h)]).T
    right_edge = np.array([np.ones(h)*(w-1), np.arange(0, h)]).T
    top_edge = np.array([np.arange(0, w), np.ones(w)*0]).T
    bottom_edge = np.array([np.arange(0, w), np.ones(w)*(h-1)]).T

    left_map = map_points(left_edge, image_size, affine, distortion_coeffs, camera_matrix,warp_mode=warp_mode)
    left_bounds = min_max(left_map)
    right_map = map_points(right_edge, image_size, affine, distortion_coeffs, camera_matrix,warp_mode=warp_mode)
    right_bounds = min_max(right_map)
    top_map = map_points(top_edge, image_size, affine, distortion_coeffs, camera_matrix,warp_mode=warp_mode)
    top_bounds = min_max(top_map)
    bottom_map = map_points(bottom_edge, image_size, affine, distortion_coeffs, camera_matrix,warp_mode=warp_mode)
    bottom_bounds = min_max(bottom_map)

    bounds = Bounds()
    bounds.max.x = right_bounds.min.x
    bounds.max.y = bottom_bounds.min.y
    bounds.min.x = left_bounds.max.x
    bounds.min.y = top_bounds.max.y
    edges = (left_map,right_map,top_map,bottom_map)
    return bounds,edges

def get_combined_bounds(bounds, image_size):
    w = image_size[0]
    h = image_size[1]

    final = Bounds()

    final.min.x = final.min.y = 0
    final.max.x = w
    final.max.y = h

    for b in bounds:
        final.min.x = max(final.min.x, b.min.x)
        final.min.y = max(final.min.y, b.min.y)
        final.max.x = min(final.max.x, b.max.x)
        final.max.y = min(final.max.y, b.max.y)

    # limit to image size
    final.min.x = max(final.min.x, 0)
    final.min.y = max(final.min.y, 0)
    final.max.x = min(final.max.x, w-1)
    final.max.y = min(final.max.y, h-1)
    # Add 1 px of margin (remove one pixel on all sides)
    final.min.x += 1
    final.min.y += 1
    final.max.x -= 1
    final.max.y -= 1

    return final

def min_max(pts):
    bounds = Bounds()
    for p in pts:
        if p[0] > bounds.max.x:
            bounds.max.x = p[0]
        if p[1] > bounds.max.y:
            bounds.max.y = p[1]
        if p[0] < bounds.min.x:
            bounds.min.x = p[0]
        if p[1] < bounds.min.y:
            bounds.min.y = p[1]
    return bounds

def map_points(pts, image_size, warpMatrix, distortion_coeffs, camera_matrix,warp_mode=cv2.MOTION_HOMOGRAPHY):
    # extra dimension makes opencv happy
    pts = np.array([pts], dtype=np.float)
    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, image_size, 1)
    new_pts = cv2.undistortPoints(pts, camera_matrix, distortion_coeffs, P=new_cam_mat)
    if warp_mode == cv2.MOTION_AFFINE:
        new_pts = cv2.transform(new_pts, cv2.invertAffineTransform(warpMatrix))
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        new_pts =cv2.perspectiveTransform(new_pts,np.linalg.inv(warpMatrix).astype(np.float32))
    #apparently the output order has changed in 4.1.1 (possibly earlier from 3.4.3)
    if cv2.__version__<='3.4.4':
        return new_pts[0]
    else:
        return new_pts[:,0,:]

