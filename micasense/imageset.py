#!/usr/bin/env python
# coding: utf-8
"""
RedEdge Capture Class

    A Capture is a set of images taken by one RedEdge cameras which share
    the same unique capture identifier.  Generally these images will be
    found in the same folder and also share the same filename prefix, such
    as IMG_0000_*.tif, but this is not required

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
import os, glob, fnmatch
import micasense.image as image
import micasense.capture as capture
import multiprocessing

import exiftool


def image_from_file(filename):
    return image.Image(filename)

class ImageSet(object):
    """
    An ImageSet is a container for a group of captures that are processed together
    """
    def __init__(self, captures):
        self.captures = captures
        captures.sort()
        
    @classmethod
    def from_directory(cls, directory, progress_callback=None, exiftool_path=None):
        """
        Create and ImageSet recursively from the files in a directory
        """
        cls.basedir = directory
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, '*.tif'):
                matches.append(os.path.join(root, filename))

        images = []

        if exiftool_path is None and os.environ.get('exiftoolpath') is not None:
            exiftool_path = os.path.normpath(os.environ.get('exiftoolpath'))

        with exiftool.ExifTool(exiftool_path) as exift:
            for i,path in enumerate(matches):
                images.append(image.Image(path, exiftool_obj=exift))
                if progress_callback is not None:
                    progress_callback(float(i)/float(len(matches)))

        # create a dictionary to index the images so we can sort them
        # into captures
        # {
        #     "capture_id": [img1, img2, ...]
        # }
        captures_index = {}
        for img in images:
            c = captures_index.get(img.capture_id)
            if c is not None:
                c.append(img)
            else:
                captures_index[img.capture_id] = [img]
        captures = []
        for cap_imgs in captures_index:
            imgs = captures_index[cap_imgs]
            newcap = capture.Capture(imgs)
            captures.append(newcap)
        if progress_callback is not None:
            progress_callback(1.0)
        return cls(captures)
    
    def as_nested_lists(self):
        columns = [
            'timestamp',
            'latitude','longitude','altitude',
            'capture_id',
            'dls-yaw','dls-pitch','dls-roll'
        ]
        irr = ["irr-{}".format(wve) for wve in self.captures[0].center_wavelengths()]
        columns += irr
        data = [] 
        for cap in self.captures:
            dat = cap.utc_time()
            loc = list(cap.location())
            uuid = cap.uuid
            dls_pose = list(cap.dls_pose())
            irr = cap.dls_irradiance()
            row = [dat]+loc+[uuid]+dls_pose+irr
            data.append(row)
        return data, columns

    def dls_irradiance(self):
        series = {}
        for cap in self.captures:
            dat = cap.utc_time().isoformat()
            irr = cap.dls_irradiance()
            series[dat] = irr

