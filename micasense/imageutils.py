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
import numpy as np
import multiprocessing

def normalize(im):
    width, height = im.shape
    norm = np.zeros((width, height), dtype=np.float32)
    cv2.normalize(im, dst=norm, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm[norm<0.0] = 0.0
    norm[norm>1.0] = 1.0
    return norm

def gradient(im):
    im = normalize(im)
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

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
    """
    warp_mode = pair['warp_mode']
    max_iterations = pair['max_iterations']
    epsilon_threshold = pair['epsilon_threshold']
    ref_index = pair['ref_index']
    match_index = pair['match_index']
    
    # Initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Terminate the optimizer if either the max iterations or the threshold are reached
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon_threshold)
    
    if ref_index != match_index:
        (cc, warp_matrix) = cv2.findTransformECC(
            gradient(pair['ref_image']), 
            gradient(pair['match_image']),
            warp_matrix,
            warp_mode,
            criteria)

    return {'ref_index': pair['ref_index'],
            'match_index': pair['match_index'],
            'warp_matrix': warp_matrix }

def align_capture(capture, ref_index=4, warp_mode=cv2.MOTION_AFFINE, max_iterations=2500, epsilon_threshold=1e-9):
    '''Align images in a capture using openCV
    MOTION_TRANSLATION sets a translational motion model; warpMatrix is 2x3 with the first 2x2 part being the unity matrix and the rest two parameters being estimated.
    MOTION_EUCLIDEAN sets a Euclidean (rigid) transformation as motion model; three parameters are estimated; warpMatrix is 2x3.
    MOTION_AFFINE sets an affine motion model (DEFAULT); six parameters are estimated; warpMatrix is 2x3.
    MOTION_HOMOGRAPHY sets a homography as a motion model; eight parameters are estimated;`warpMatrix` is 3x3.
    best results will be AFFINE and HOMOGRAPHY, at the expense of speed
    '''
    # Match other bands to this reference image (index into capture.images[])
    ref_img = capture.images[ref_index].undistorted(capture.images[ref_index].reflectance()).astype('float32')
    alignment_pairs = []
    for img in capture.images:
        alignment_pairs.append({'warp_mode': warp_mode,
                                'max_iterations': max_iterations,
                                'epsilon_threshold': epsilon_threshold,
                                'ref_index':ref_index,
                                'ref_image': ref_img, 
                                'match_index':img.band_index,
                                'match_image':img.undistorted(img.reflectance()).astype('float32')})

    warp_matrices = [None]*len(alignment_pairs)
    
    #required to work across linux/mac/windows, see https://stackoverflow.com/questions/47852237
    multiprocessing.set_start_method('spawn') 
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for i,mat in enumerate(pool.imap_unordered(align, alignment_pairs)):
        warp_matrices[mat['match_index']] = mat['warp_matrix']
        print("Finished aligning band {}".format(mat['match_index']))
    pool.close()
    pool.join()

    return warp_matrices, alignment_pairs

#apply homography to create an aligned stack 
def aligned_capture(warp_matrices, alignment_pairs, dimension_tuple):
    height, width = alignment_pairs[0]['ref_image'].shape
    im_aligned = np.zeros((height,width,len(warp_matrices)), dtype=np.float32 )
    
    for i in range(0,len(warp_matrices)):
        warp_mode = alignment_pairs[i]['warp_mode']
        
        if alignment_pairs[i]['match_index'] == alignment_pairs[i]['ref_index']:
            im_aligned[:,:,i] = alignment_pairs[i]['match_image']
        else:
            if warp_mode != cv2.MOTION_HOMOGRAPHY:
                im_aligned[:,:,i] = cv2.warpAffine(alignment_pairs[i]['match_image'], 
                                                warp_matrices[i], 
                                                (width,height), 
                                                flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP)
            else:
                im_aligned[:,:,i] = cv2.warpPerspective(alignment_pairs[i]['match_image'], 
                                                    warp_matrices[i], 
                                                    (width,height), 
                                                    flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP)
    (left, top, w, h) = tuple(int(i) for i in dimension_tuple)
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

def find_crop_bounds(image_size, registration_transforms, lens_distortions, camera_matrices):
    """Compute the crop rectangle to be applied to a set of images after 
    registration such that no pixel in the resulting stack of images will 
    include a blank value for any of the bands

    Args: 
    image_size - Tuple containing (width, height) of the image
    registration_transforms - a list of affine transforms applied to 
    register the image. It is required. 
    lens_distortion - A set of lens distortion coefficients to be applied
    prior to the registration transform. If provided, it must be the same
    length as registration_transforms. Each element in the list should be 
    a dict with the following keys: 'cx', 'cy', 'fx', 'fy', 'p', 'k'. 

    """

    bounds = [get_inner_rect(image_size, a, d, c) for a, d, c in zip(registration_transforms, lens_distortions, camera_matrices)]
    combined_bounds = get_combined_bounds(bounds, image_size)
    
    left = round(combined_bounds.min.x)
    top = round(combined_bounds.min.y)
    width = round(combined_bounds.max.x - combined_bounds.min.x + 0.5)
    height = round(combined_bounds.max.y - combined_bounds.min.y + 0.5)
    return (left, top, width, height)

def get_inner_rect(image_size, affine, distortion_coeffs, camera_matrix):
    w = image_size[0]
    h = image_size[1]

    left_edge = np.array([np.ones(h)*0, np.arange(0, h)]).T
    right_edge = np.array([np.ones(h)*(w-1), np.arange(0, h)]).T
    top_edge = np.array([np.arange(0, w), np.ones(w)*0]).T
    bottom_edge = np.array([np.arange(0, w), np.ones(w)*(h-1)]).T

    left_bounds = min_max(map_points(left_edge, image_size, affine, distortion_coeffs, camera_matrix))
    right_bounds = min_max(map_points(right_edge, image_size, affine, distortion_coeffs, camera_matrix))
    top_bounds = min_max(map_points(top_edge, image_size, affine, distortion_coeffs, camera_matrix))
    bottom_bounds = min_max(map_points(bottom_edge, image_size, affine, distortion_coeffs, camera_matrix))

    bounds = Bounds()
    bounds.max.x = right_bounds.min.x
    bounds.max.y = bottom_bounds.min.y
    bounds.min.x = left_bounds.max.x
    bounds.min.y = top_bounds.max.y

    return bounds

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

def map_points(pts, image_size, affine, distortion_coeffs, camera_matrix):
    #assert len(affine) == 6, "affine must have len == 6, has len {}".format(len(affine))

    # extra dimension makes opencv happy
    pts = np.array([pts], dtype=np.float)
    
    new_cam_mat, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, image_size, 1)
    new_pts = cv2.undistortPoints(pts, camera_matrix, distortion_coeffs, P=new_cam_mat)

    new_pts = cv2.transform(new_pts, cv2.invertAffineTransform(affine))

    return new_pts[0]