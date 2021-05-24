#!/usr/bin/env python
# coding: utf-8
"""
MicaSense ImageSet Class

    An ImageSet contains a group of Captures. The Captures can be loaded from Image objects, from a list of files,
    or by recursively searching a directory for images.

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

import fnmatch
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pprint import pprint

import exiftool
from tqdm import tqdm

import micasense.capture as capture
import micasense.image as image

warnings.simplefilter(action="once")


# FIXME: mirrors Capture.append_file(). Not used. Does this still belong here?
def image_from_file(filename):
    return image.Image(filename)


def parallel_process(function, iterable, parameters, progress_callback=None, use_tqdm=False):
    """
    Multiprocessing Pool handler.
    :param function: function used in multiprocessing call
    :param iterable: iterable holding objects passed to function for each process
    :param parameters: dict of any function parameters other than the iterable object
    :param use_tqdm: boolean True to use tqdm progress bar
    :param progress_callback: function to report progress to
    :return: None
    """

    with ProcessPoolExecutor() as pool:
        # run multiprocessing
        futures = [pool.submit(partial(function, parameters), i) for i in iterable]

        if use_tqdm:
            # kwargs for tqdm
            kwargs = {
                'total': len(futures),
                'unit': 'Capture',
                'unit_scale': False,
                'leave': True
            }

            # Receive Future objects as they complete. Print out the progress as tasks complete
            for _ in tqdm(iterable=as_completed(futures), desc='Processing ImageSet', **kwargs):
                pass
        elif progress_callback is not None:
            futures_len = float(len(futures))
            for i, _ in enumerate(as_completed(futures)):
                progress_callback(float(i) / futures_len)


def save_capture(params, cap):
    """
    Process an ImageSet according to program parameters. Saves rgb
    :param params: dict of program parameters from ImageSet.process_imageset()
    :param cap: micasense.capture.Capture object
    """
    try:
        # align capture
        if len(cap.images) == params['capture_len']:
            cap.create_aligned_capture(
                irradiance_list=params['irradiance'],
                warp_matrices=params['warp_matrices'],
                img_type=params['img_type']
            )
        else:
            print(f"\tCapture {cap.uuid} only has {len(cap.images)} Images. Should have {params['capture_len']}. "
                  f"Skipping...")
            return

        if params['output_stack_dir']:
            output_stack_file_path = os.path.join(params['output_stack_dir'], cap.uuid + '.tif')
            if params['overwrite'] or not os.path.exists(output_stack_file_path):
                cap.save_capture_as_stack(output_stack_file_path)
        if params['output_rgb_dir']:
            output_rgb_file_path = os.path.join(params['output_rgb_dir'], cap.uuid + '.jpg')
            if params['overwrite'] or not os.path.exists(output_rgb_file_path):
                cap.save_capture_as_rgb(output_rgb_file_path)

        cap.clear_image_data()
    except Exception as e:
        print(e)
        pprint(params)
        quit()


class ImageSet(object):
    """An ImageSet is a container for a group of Captures that are processed together."""

    def __init__(self, captures):
        self.captures = captures
        captures.sort()

    @classmethod
    def from_directory(cls, directory, progress_callback=None, use_tqdm=False, exiftool_path=None):
        """
        Create an ImageSet recursively from the files in a directory.
        :param directory: str system file path
        :param progress_callback: function to report progress to
        :param use_tqdm: boolean True to use tqdm progress bar
        :param exiftool_path: str system file path to exiftool location
        :return: ImageSet instance
        """

        # progress_callback deprecation warning
        if progress_callback is not None:
            warnings.warn(message='The progress_callback parameter will be deprecated in favor of use_tqdm',
                          category=PendingDeprecationWarning)

        # ensure exiftoolpath is found per MicaSense setup instructions
        if exiftool_path is None and os.environ.get('exiftoolpath') is not None:
            exiftool_path = os.path.normpath(os.environ.get('exiftoolpath'))

        cls.basedir = directory
        matches = []
        for root, _, filenames in os.walk(directory):
            [matches.append(os.path.join(root, filename)) for filename in fnmatch.filter(filenames, '*.tif')]

        images = []

        with exiftool.ExifTool(exiftool_path) as exift:
            if use_tqdm:  # to use tqdm progress bar instead of progress_callback
                kwargs = {
                    'total': len(matches),
                    'unit': ' Files',
                    'unit_scale': False,
                    'leave': True
                }
                for path in tqdm(iterable=matches, desc='Loading ImageSet', **kwargs):
                    images.append(image.Image(path, exiftool_obj=exift))
            else:
                print('Loading ImageSet from: {}'.format(directory))
                for i, path in enumerate(matches):
                    images.append(image.Image(path, exiftool_obj=exift))
                    if progress_callback is not None:
                        progress_callback(float(i) / float(len(matches)))

        # create a dictionary to index the images so we can sort them into captures
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
        """
        Get timestamp, latitude, longitude, altitude, capture_id, dls-yaw, dls-pitch, dls-roll, and irradiance from all
        Captures.
        :return: List data from all Captures, List column headers.
        """
        columns = [
            'timestamp',
            'latitude', 'longitude', 'altitude',
            'capture_id',
            'dls-yaw', 'dls-pitch', 'dls-roll'
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
            row = [dat] + loc + [uuid] + dls_pose + irr
            data.append(row)
        return data, columns

    def dls_irradiance(self):
        """
        Get utc_time and irradiance for each Capture in ImageSet.
        :return: dict {utc_time : [irradiance, ...]}
        """
        series = {}
        for cap in self.captures:
            dat = cap.utc_time().isoformat()
            irr = cap.dls_irradiance()
            series[dat] = irr
        return series

    def process_imageset(self,
                         output_stack_directory=None,
                         output_rgb_directory=None,
                         warp_matrices=None,
                         irradiance=None,
                         img_type=None,
                         multiprocess=True,
                         overwrite=False,
                         progress_callback=None,
                         use_tqdm=False):
        """
        Write band stacks and rgb thumbnails to disk.
        :param warp_matrices: 2d List of warp matrices derived from Capture.get_warp_matrices()
        :param output_stack_directory: str system file path to output stack directory
        :param output_rgb_directory: str system file path to output thumbnail directory
        :param irradiance: List returned from Capture.dls_irradiance() or Capture.panel_irradiance()    <-- TODO: Write a better docstring for this
        :param img_type: str 'radiance' or 'reflectance'. Desired image output type.
        :param multiprocess: boolean True to use multiprocessing module
        :param overwrite: boolean True to overwrite existing files
        :param progress_callback: function to report progress to
        :param use_tqdm: boolean True to use tqdm progress bar
        """

        if progress_callback is not None:
            warnings.warn(message='The progress_callback parameter will be deprecated in favor of use_tqdm',
                          category=PendingDeprecationWarning)

        # ensure some output is requested
        if output_stack_directory is None and output_rgb_directory is None:
            raise RuntimeError('No output requested for the ImageSet.')

        # make output dirs if not exist
        if output_stack_directory is not None and not os.path.exists(output_stack_directory):
            os.mkdir(output_stack_directory)
        if output_rgb_directory is not None and not os.path.exists(output_rgb_directory):
            os.mkdir(output_rgb_directory)

        # processing parameters
        params = {
            'warp_matrices': warp_matrices,
            'irradiance': irradiance,
            'img_type': img_type,
            'capture_len': len(self.captures[0].images),
            'output_stack_dir': output_stack_directory,
            'output_rgb_dir': output_rgb_directory,
            'overwrite': overwrite,
        }

        print('Processing {} Captures ...'.format(len(self.captures)))

        # multiprocessing with concurrent futures
        if multiprocess:
            parallel_process(function=save_capture, iterable=self.captures, parameters=params,
                             progress_callback=progress_callback, use_tqdm=use_tqdm)

        # else serial processing
        else:
            if use_tqdm:
                kwargs = {
                    'total': len(self.captures),
                    'unit': 'Capture',
                    'unit_scale': False,
                    'leave': True
                }
                for cap in tqdm(iterable=self.captures, desc='Processing ImageSet', **kwargs):
                    save_capture(params, cap)
            else:
                for i, cap in enumerate(self.captures):
                    save_capture(params, cap)
                    if progress_callback is not None:
                        progress_callback(float(i) / float(len(self.captures)))

        print('Processing complete.')
