#!/usr/bin/env python
# coding: utf-8
"""
MicaSense Capture Class

    A Capture is a set of Images taken by one camera which share
    the same unique capture identifier (capture_id).  Generally these images will be
    found in the same folder and also share the same filename prefix, such
    as IMG_0000_*.tif, but this is not required.

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
import logging
import math
import os
from collections import namedtuple

import cv2
import imageio
import numpy as np
from skimage.feature import match_descriptors, SIFT
from skimage.measure import ransac
from skimage.transform import estimate_transform, FundamentalMatrixTransform, ProjectiveTransform, \
    resize

import micasense.image as image
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils


class Capture(object):
    """
    A Capture is a set of Images taken by one MicaSense camera which share
    the same unique capture identifier (capture_id). Generally these images will be
    found in the same folder and also share the same filename prefix, such
    as IMG_0000_*.tif, but this is not required.
    """

    def __init__(self, images, panel_corners=None):
        """
        :param images: str or List of str system file paths.
            Class is typically created using from_file() or from_file_list() methods.
            Captures are also created automatically using ImageSet.from_directory()
        :param panel_corners: 3d List of int coordinates
            e.g. [[[873, 1089], [767, 1083], [763, 1187], [869, 1193]],
                    [[993, 1105], [885, 1101], [881, 1205], [989, 1209]],
                    [[1000, 1030], [892, 1026], [888, 1130], [996, 1134]],
                    [[892, 989], [786, 983], [780, 1087], [886, 1093]],
                    [[948, 1061], [842, 1057], [836, 1161], [942, 1165]]]

            The camera should automatically detect panel corners. This instance variable will be None for aerial
            captures. You can populate this for panel captures by calling detect_panels().
        """
        if isinstance(images, image.Image):
            self.images = [images]
        elif isinstance(images, list):
            self.images = images
        else:
            raise RuntimeError("Provide an Image or list of Images to create a Capture.")
        self.num_bands = len(self.images)
        self.images.sort()
        capture_ids = [img.capture_id for img in self.images]
        if len(set(capture_ids)) != 1:
            raise RuntimeError("Images provided must have the same capture_id.")
        self.uuid = self.images[0].capture_id
        self.flightid = self.images[0].flight_id
        self.camera_model = self.images[0].camera_model
        self.camera_serial = self.images[0].camera_serial
        self.camera_serials = set([img.camera_serial for img in self.images])
        self.bits_per_pixel = self.images[0].bits_per_pixel
        self.panels = None
        self.detected_panel_count = 0
        if panel_corners is None:
            self.panelCorners = [None] * len(self.eo_indices())
        else:
            self.panelCorners = panel_corners

        self.__aligned_capture = None
        self.__aligned_radiometric_pan_sharpened_capture = None
        self.__sift_warp_matrices = None

    def set_panel_corners(self, panel_corners):
        """
        Define panel corners by hand.
        :param panel_corners: 2d List of int coordinates.
            e.g. [[536, 667], [535, 750], [441, 755], [444, 672]]
        :return: None
        """
        self.panel_corners = panel_corners
        self.panels = None
        self.detect_panels()

    def append_image(self, img):
        """
        Add an Image to the Capture.
        :param img: An Image object.
        :return: None
        """
        if self.uuid != img.capture_id:
            raise RuntimeError("Added images must have the same capture_id.")
        self.images.append(img)
        self.images.sort()

    def append_images(self, images):
        """
        Add multiple Images to the Capture.
        :param images: List of Image objects.
        """
        [self.append_image(img) for img in images]

    def append_file(self, file_name):
        """
        Add an Image to the Capture using a file path.
        :param file_name: str system file path.
        """
        self.append_image(image.Image(file_name))

    @classmethod
    def from_file(cls, file_name, allow_uncalibrated=False):
        """
        Create Capture instance from file path.
        :param file_name: str system file path
        :return: Capture object.
        """
        return cls(image.Image(file_name, allow_uncalibrated=allow_uncalibrated))

    @classmethod
    def from_filelist(cls, file_list, allow_uncalibrated=False):
        """
        Create Capture instance from List of file paths.
        :param file_list: List of str system file paths.
        :return: Capture object.
        """
        if len(file_list) == 0:
            raise IOError("No files provided. Check your file paths.")
        for fle in file_list:
            if not os.path.isfile(fle):
                raise IOError(f"All files in file list must be a file. The following file is not:\n{fle}")
        images = [image.Image(fle, allow_uncalibrated=allow_uncalibrated) for fle in file_list]
        return cls(images)

    def __get_reference_index(self):
        """
        Find the reference image which has the smallest rig offsets - they should be (0,0).
        :return: ndarray of ints - The indices of the minimum values along an axis.
        """
        return np.argmin((np.array([i.rig_xy_offset_in_px() for i in self.images]) ** 2).sum(1))

    def __plot(self, imgs, num_cols=2, plot_type=None, colorbar=True, figsize=(14, 14)):
        """
        Plot the Images from the Capture.
        :param images: List of Image objects
        :param num_cols: int number of columns
        :param plot_type: str for plot title formatting
        :param color_bar: boolean to determine color bar inclusion
        :param fig_size: Tuple size of the figure
        :return: plotutils result. matplotlib Figure and Axis in both cases.
        """
        if plot_type is None:
            plot_type = ''
        else:
            titles = [
                '{} Band {} {}'.format(str(img.band_name), str(img.band_index),
                                       plot_type if img.band_name.upper() != 'LWIR' else 'Brightness Temperature')
                for img
                in self.images
            ]
        num_rows = int(math.ceil(float(len(self.images)) / float(num_cols)))
        if colorbar:
            return plotutils.subplotwithcolorbar(num_rows, num_cols, imgs, titles, figsize)
        else:
            return plotutils.subplot(num_rows, num_cols, imgs, titles, figsize)

    def __lt__(self, other):
        return self.utc_time() < other.utc_time()

    def __gt__(self, other):
        return self.utc_time() > other.utc_time()

    def __eq__(self, other):
        return self.uuid == other.uuid

    def location(self):
        """(lat, lon, alt) tuple of WGS-84 location units are radians, meters msl"""
        return self.images[0].location

    def utc_time(self):
        """Returns a timezone-aware datetime object of the capture time."""
        return self.images[0].utc_time

    def clear_image_data(self):
        """
        Clears (dereferences to allow garbage collection) all internal image data stored in this class. Call this
        after processing-heavy image calls to manage program memory footprint. When processing many images, such as
        iterating over the Captures in an ImageSet, it may be necessary to call this after Capture is processed.
        """
        for img in self.images:
            img.clear_image_data()
        self.__aligned_capture = None
        self.__aligned_radiometric_pan_sharpened_capture = None

    def center_wavelengths(self):
        """Returns a list of the image center wavelengths in nanometers."""
        return [img.center_wavelength for img in self.images]

    def band_names(self):
        """Returns a list of the image band names as they are in the image metadata."""
        return [img.band_name for img in self.images]

    def band_names_lower(self):
        """Returns a list of the Image band names in all lower case for easier comparisons."""
        return [img.band_name.lower() for img in self.images]

    def dls_present(self):
        """Returns true if DLS metadata is present in the images."""
        return self.images[0].dls_present

    def dls_irradiance_raw(self):
        """Returns a list of the raw DLS measurements from the image metadata."""
        return [img.spectral_irradiance for img in self.images]

    def dls_irradiance(self):
        """Returns a list of the corrected earth-surface (horizontal) DLS irradiance in W/m^2/nm."""
        return [img.horizontal_irradiance for img in self.images]

    def direct_irradiance(self):
        """Returns a list of the DLS irradiance from the direct source in W/m^2/nm."""
        return [img.direct_irradiance for img in self.images]

    def scattered_irradiance(self):
        """Returns a list of the DLS scattered irradiance from the direct source in W/m^2/nm."""
        return [img.scattered_irradiance for img in self.images]

    def dls_pose(self):
        """Returns (yaw, pitch, roll) tuples in radians of the earth-fixed DLS pose."""
        return self.images[0].dls_yaw, self.images[0].dls_pitch, self.images[0].dls_roll

    def focal_length(self):
        """Returns focal length of multispectral bands or of panchromatic band if applicable."""
        if 'Panchro' in self.eo_band_names():
            return self.images[self.eo_band_names().index('Panchro')].focal_length
        else:
            return self.images[0].focal_length

    def plot_raw(self):
        """Plot raw images as the data came from the camera."""
        self.__plot([img.raw() for img in self.images],
                    plot_type='Raw')

    def plot_vignette(self):
        """Compute (if necessary) and plot vignette correction images."""
        self.__plot([img.vignette()[0].T for img in self.images],
                    plot_type='Vignette')

    def plot_radiance(self):
        """Compute (if necessary) and plot radiance images."""
        self.__plot([img.radiance() for img in self.images],
                    plot_type='Radiance')

    def plot_undistorted_radiance(self):
        """Compute (if necessary) and plot undistorted radiance images."""
        self.__plot(
            [img.undistorted(img.radiance()) for img in self.images],
            plot_type='Undistorted Radiance')

    def plot_undistorted_reflectance(self, irradiance_list):
        """
        Compute (if necessary) and plot reflectances given a list of irradiances.
        :param irradiance_list: List returned from Capture.dls_irradiance() or Capture.panel_irradiance()
        """
        self.__plot(
            self.undistorted_reflectance(irradiance_list),
            plot_type='Undistorted Reflectance')

    def compute_radiance(self):
        """
        Compute Image radiances.
        :return: None
        """
        [img.radiance() for img in self.images]

    def compute_undistorted_radiance(self):
        """
        Compute Image undistorted radiance.
        :return: None
        """
        [img.undistorted_radiance() for img in self.images]

    def compute_reflectance(self, irradiance_list=None, force_recompute=True):
        """
        Compute Image reflectance from irradiance list, but don't return.
        :param irradiance_list: List returned from Capture.dls_irradiance() or Capture.panel_irradiance()
        :param force_recompute: boolean to determine if reflectance is recomputed.
        :return: None
        """
        if irradiance_list is not None:
            [img.reflectance(irradiance_list[i], force_recompute=force_recompute) for i, img in enumerate(self.images)]
        else:
            [img.reflectance(force_recompute=force_recompute) for img in self.images]

    def compute_undistorted_reflectance(self, irradiance_list=None, force_recompute=True):
        """
        Compute undistorted image reflectance from irradiance list.
        :param irradiance_list: List returned from Capture.dls_irradiance() or Capture.panel_irradiance()   TODO: improve this docstring
        :param force_recompute: boolean to determine if reflectance is recomputed.
        :return: None
        """
        if irradiance_list is not None:
            [img.undistorted_reflectance(irradiance_list[i], force_recompute=force_recompute) for i, img in
             enumerate(self.images)]
        else:
            [img.undistorted_reflectance(force_recompute=force_recompute) for img in self.images]

    def eo_images(self):
        """Returns a list of the EO Images in the Capture."""
        return [img for img in self.images if img.band_name != 'LWIR']

    def lw_images(self):
        """Returns a list of the longwave infrared Images in the Capture."""
        return [img for img in self.images if img.band_name == 'LWIR']

    def eo_indices(self):
        """Returns a list of the indexes of the EO Images in the Capture."""
        return [index for index, img in enumerate(self.images) if img.band_name != 'LWIR']

    def eo_band_names(self):
        return [band for band in self.band_names() if band != 'LWIR']

    def lw_indices(self):
        """Returns a list of the indexes of the longwave infrared Images in the Capture."""
        return [index for index, img in enumerate(self.images) if img.band_name == 'LWIR']

    def reflectance(self, irradiance_list):
        """
        Compute reflectance Images.
        :param irradiance_list: List returned from Capture.dls_irradiance() or Capture.panel_irradiance()   TODO: improve this docstring
        :return: List of reflectance EO and long wave infrared Images for given irradiance.
        """
        eo_imgs = [img.reflectance(irradiance_list[i]) for i, img in enumerate(self.eo_images())]
        lw_imgs = [img.reflectance() for i, img in enumerate(self.lw_images())]
        return eo_imgs + lw_imgs

    def undistorted_reflectance(self, irradiance_list):
        """
        Compute undistorted reflectance Images.
        :param irradiance_list: List returned from Capture.dls_irradiance() or Capture.panel_irradiance()   TODO: improve this docstring
        :return: List of undistorted reflectance images for given irradiance.
        """
        eo_imgs = [img.undistorted(img.reflectance(irradiance_list[i])) for i, img in enumerate(self.eo_images())]
        lw_imgs = [img.undistorted(img.reflectance()) for i, img in enumerate(self.lw_images())]
        return eo_imgs + lw_imgs

    def panels_in_all_expected_images(self):
        """
        Check if all expected reflectance panels are detected in the EO Images in the Capture.
        :return: True if reflectance panels are detected.
        """
        expected_panels = sum(str(img.band_name).upper() != 'LWIR' for img in self.images)
        return self.detect_panels() == expected_panels

    def panel_raw(self):
        """Return a list of mean panel region values for raw images."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        raw_list = []
        for p in self.panels:
            mean, _, _, _ = p.raw()
            raw_list.append(mean)
        return raw_list

    def panel_radiance(self):
        """Return a list of mean panel region values for converted radiance Images."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        radiance_list = []
        for p in self.panels:
            mean, _, _, _ = p.radiance()
            radiance_list.append(mean)
        return radiance_list

    def panel_irradiance(self, reflectances=None):
        """Return a list of mean panel region values for irradiance values."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        if reflectances is None:
            reflectances = [panel.reflectance_from_panel_serial() for panel in self.panels]
        if len(reflectances) != len(self.panels):
            raise ValueError("Length of panel reflectances must match length of Images.")
        irradiance_list = []
        for i, p in enumerate(self.panels):
            mean_irr = p.irradiance_mean(reflectances[i])
            irradiance_list.append(mean_irr)
        return irradiance_list

    def panel_reflectance(self):
        """Return a list of mean panel reflectance values."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        reflectance_list = []
        for i, p in enumerate(self.panels):
            self.images[i].reflectance()
            mean_refl = p.reflectance_mean()
            reflectance_list.append(mean_refl)
        return reflectance_list

    def panel_albedo(self):
        """Return a list of panel reflectance values from metadata."""
        if self.panels_in_all_expected_images():
            albedos = [panel.reflectance_from_panel_serial() for panel in self.panels]
            if None in albedos:
                albedos = None
        else:
            albedos = None
        return albedos

    def detect_panels(self):
        """Detect reflectance panels in the Capture, and return a count."""
        from micasense.panel import Panel
        if self.panels is not None and self.detected_panel_count == len(self.images):
            return self.detected_panel_count
        self.panels = [Panel(img, panel_corners=pc) for img, pc in zip(self.images, self.panelCorners)]
        self.detected_panel_count = 0
        for p in self.panels:
            if p.panel_detected():
                self.detected_panel_count += 1
        # is panelCorners are defined by hand
        if self.panelCorners is not None and all(corner is not None for corner in self.panelCorners):
            self.detected_panel_count = len(self.panelCorners)
        return self.detected_panel_count

    def plot_panels(self):
        """Plot Panel images."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        self.__plot(
            [p.plot_image() for p in self.panels],
            plot_type='Panels',
            colorbar=False
        )

    def set_external_rig_relatives(self, external_rig_relatives):
        """
        Set external rig relatives.
        :param external_rig_relatives: TODO: Write this parameter docstring
        :return: None
        """
        for i, img in enumerate(self.images):
            img.set_external_rig_relatives(external_rig_relatives[str(i)])

    def has_rig_relatives(self):
        """
        Check if Images in Capture have rig relatives.
        :return: boolean True if all Images have rig relatives metadata.
        """
        for img in self.images:
            if img.meta.rig_relatives() is None:
                return False
        return True

    def get_warp_matrices(self, ref_index=None):
        """
        Get warp matrices.
        :param ref_index: int to specify image for homography
        :return: 2d List of warp matrices
        """
        if ref_index is None:
            ref = self.images[self.__get_reference_index()]
        else:
            ref = self.images[ref_index]
        warp_matrices = [np.linalg.inv(im.get_homography(ref)) for im in self.images]
        return [w / w[2, 2] for w in warp_matrices]

    def create_aligned_capture(self, irradiance_list=None, warp_matrices=None, img_type=None,
                               motion_type=cv2.MOTION_HOMOGRAPHY):
        """
        Creates aligned Capture. Computes undistorted radiance or reflectance images if necessary.
        :param irradiance_list: List of mean panel region irradiance.
        :param warp_matrices: 2d List of warp matrices derived from Capture.get_warp_matrices()
        :param normalize: FIXME: This parameter isn't used?
        :param img_type: str 'radiance' or 'reflectance' depending on image metadata.
        :param motion_type: OpenCV import. Also know as warp_mode. MOTION_HOMOGRAPHY or MOTION_AFFINE.
                            For Altum images only use HOMOGRAPHY.
        :return: ndarray with alignment changes
        """
        if img_type is None and irradiance_list is None and self.dls_irradiance() is None:
            self.compute_undistorted_radiance()
            img_type = 'radiance'
        elif img_type is None:
            if irradiance_list is None:
                irradiance_list = self.dls_irradiance() + [0]
            self.compute_undistorted_reflectance(irradiance_list)
            img_type = 'reflectance'
        if warp_matrices is None:
            warp_matrices = self.get_warp_matrices()
        if self.camera_model in ('RedEdge-P', 'Altum-PT'):
            match_index = 5
            reference_band = 5
            logging.warning(
                "For RedEdge-P or Altum-PT, you should use SIFT_align_capture instead of create_aligned_capture")
        # for RedEdge-MX Dual Camera System
        elif len(self.eo_band_names()) == 10:
            match_index = 4
            reference_band = 0
        else:
            match_index = 1
            reference_band = 0
        cropped_dimensions, _ = imageutils.find_crop_bounds(self, warp_matrices, warp_mode=motion_type,
                                                            reference_band=reference_band)
        self.__aligned_capture = imageutils.aligned_capture(self,
                                                            warp_matrices,
                                                            motion_type,
                                                            cropped_dimensions,
                                                            match_index,
                                                            img_type=img_type)
        return self.__aligned_capture

    def aligned_shape(self):
        """
        Get aligned_capture ndarray shape.
        :return: Tuple of array dimensions for aligned_capture
        """
        if self.__aligned_capture is None:
            raise RuntimeError("Call Capture.create_aligned_capture() prior to saving as stack.")
        return self.__aligned_capture.shape

    def radiometric_pan_sharpened_aligned_capture(self, warp_matrices=None, irradiance_list=None, img_type: str = ''):
        if irradiance_list is None and self.dls_irradiance() is None:
            self.compute_undistorted_radiance()
            img_type = 'radiance'
        elif img_type == 'reflectance' and irradiance_list is not None:
            self.compute_undistorted_reflectance(irradiance_list)
        elif irradiance_list is None:
            irradiance_list = self.dls_irradiance() + [0]
            self.compute_undistorted_reflectance(irradiance_list)
            img_type = 'reflectance'
        self.__aligned_radiometric_pan_sharpened_capture = imageutils.radiometric_pan_sharpen(self,
                                                                                              warp_matrices=warp_matrices,
                                                                                              irradiance_list=irradiance_list)
        return self.__aligned_radiometric_pan_sharpened_capture

    def save_capture_as_stack(self, outfilename, sort_by_wavelength=False, photometric='MINISBLACK', pansharpen=False,
                              write_exif=True):
        """
        Output the Images in the Capture object as GTiff image stack.
        :param out_file_name: str system file path
        :param sort_by_wavelength: boolean
        :param photometric: str GDAL argument for GTiff color matching
        """
        from osgeo.gdal import GetDriverByName, GDT_UInt16
        from osgeo import gdal
        gdal.UseExceptions()

        if self.__aligned_capture is None and self.__aligned_radiometric_pan_sharpened_capture is None:
            raise RuntimeError(
                "Call Capture.create_aligned_capture() prior to saving as stack.")
        band_names = self.band_names()
        if "Panchro" in band_names and pansharpen:
            aligned_cap = self.__aligned_radiometric_pan_sharpened_capture[0]
        if "Panchro" in band_names and not pansharpen:
            aligned_cap = self.__aligned_radiometric_pan_sharpened_capture[1]
        if "Panchro" not in band_names:
            aligned_cap = self.__aligned_capture

        rows, cols, bands = aligned_cap.shape
        driver = GetDriverByName('GTiff')

        outRaster = driver.Create(outfilename, cols, rows, bands, GDT_UInt16,
                                  options=['INTERLEAVE=BAND', 'COMPRESS=DEFLATE', f'PHOTOMETRIC={photometric}'])
        try:
            if outRaster is None:
                raise IOError("could not load gdal GeoTiff driver")

            if sort_by_wavelength:
                eo_list = list(np.argsort(np.array(self.center_wavelengths())[self.eo_indices()]))
                eo_bands = list(np.array(self.eo_band_names())[np.array(eo_list)])

            else:
                eo_list = self.eo_indices()
                eo_bands = list(np.array(self.eo_band_names())[np.array(eo_list)])

            eo_count = len(eo_list)

            for outband_count, inband in enumerate(eo_list):
                outband = outRaster.GetRasterBand(outband_count + 1)
                outdata = aligned_cap[:, :, inband]
                outdata[outdata < 0] = 0
                outdata[outdata > 2] = 2  # limit reflectance data to 200% to allow some specular reflections
                outdata = outdata * 32767 # scale reflectance images so 100% = 32768
                outdata = outdata.astype(np.ushort)
                outband.SetDescription(eo_bands[outband_count])
                outband.WriteArray(outdata)
                outband.FlushCache()

            for outband_count, inband in enumerate(self.lw_indices()):
                outband = outRaster.GetRasterBand(len(eo_bands) + outband_count + 1)
                outdata = (aligned_cap[:, :,
                           inband] + 273.15) * 100  # scale data from float degC to back to centi-Kelvin to fit into uint16
                outdata[outdata < 0] = 0
                outdata[outdata > 65535] = 65535
                outdata = outdata.astype(np.ushort)
                outband.SetDescription('LWIR')
                outband.WriteArray(outdata)
                outband.FlushCache()
        finally:
            outRaster.Close()
            if write_exif:
                imageutils.write_exif_to_stack(self, outfilename)

    def save_capture_as_rgb(self, outfilename, gamma=1.4, downsample=1, white_balance='norm', hist_min_percent=0.5,
                            hist_max_percent=99.5, sharpen=True, rgb_band_indices=None):
        """
        Output the Images in the Capture object as RGB.
        :param out_file_name: str system file path
        :param gamma: float gamma correction
        :param downsample: int downsample for cv2.resize()
        :param white_balance: str 'norm' to normalize across bands using hist_min_percent and hist_max_percent.
            Else this parameter is ignored.
        :param hist_min_percent: float for min histogram stretch
        :param hist_max_percent: float for max histogram stretch
        :param sharpen: boolean
        :param rgb_band_indices: List band order
        """
        if rgb_band_indices is None:
            rgb_band_indices = [2, 1, 0]
        if self.__aligned_capture is None and self.__aligned_radiometric_pan_sharpened_capture is None:
            raise RuntimeError(
                "Call Capture.create_aligned_capture or Capture.radiometric_pan_sharpened_aligned_capture prior to saving as RGB.")
        if self.__aligned_radiometric_pan_sharpened_capture:
            aligned_capture = self.__aligned_radiometric_pan_sharpened_capture[0]
        else:
            aligned_capture = self.__aligned_capture
        im_display = np.zeros((aligned_capture.shape[0], aligned_capture.shape[1], aligned_capture.shape[2]),
                              dtype=np.float32)

        im_min = np.percentile(aligned_capture[:, :, rgb_band_indices].flatten(),
                               hist_min_percent)  # modify these percentiles to adjust contrast
        im_max = np.percentile(aligned_capture[:, :, rgb_band_indices].flatten(),
                               hist_max_percent)  # for many images, 0.5 and 99.5 are good values

        for i in rgb_band_indices:
            # for rgb true color, we usually want to use the same min and max scaling across the 3 bands to
            # maintain the "white balance" of the calibrated image
            if white_balance == 'norm':
                im_display[:, :, i] = imageutils.normalize(aligned_capture[:, :, i], im_min, im_max)
            else:
                im_display[:, :, i] = imageutils.normalize(aligned_capture[:, :, i])

        rgb = im_display[:, :, rgb_band_indices]
        rgb = cv2.resize(rgb, None, fx=1 / downsample, fy=1 / downsample, interpolation=cv2.INTER_AREA)

        if sharpen:
            gaussian_rgb = cv2.GaussianBlur(rgb, (9, 9), 10.0)
            gaussian_rgb[gaussian_rgb < 0] = 0
            gaussian_rgb[gaussian_rgb > 1] = 1
            unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
            unsharp_rgb[unsharp_rgb < 0] = 0
            unsharp_rgb[unsharp_rgb > 1] = 1
        else:
            unsharp_rgb = rgb

        # Apply a gamma correction to make the render appear closer to what our eyes would see
        if gamma != 0:
            gamma_corr_rgb = unsharp_rgb ** (1.0 / gamma)
            imageio.imwrite(outfilename, (255 * gamma_corr_rgb).astype('uint8'))
        else:
            imageio.imwrite(outfilename, (255 * unsharp_rgb).astype('uint8'))

    def save_thermal_over_rgb(self, outfilename, figsize=(30, 23), lw_index=None, hist_min_percent=0.2,
                              hist_max_percent=99.8):
        """
        Output the Images in the Capture object as thermal over RGB.
        :param out_file_name: str system file path.
        :param fig_size: Tuple dimensions of the figure.
        :param lw_index: int Index of LWIR Image in Capture.
        :param hist_min_percent: float Minimum histogram percentile.
        :param hist_max_percent: float Maximum histogram percentile.
        """
        if self.__aligned_capture is None and self.__aligned_radiometric_pan_sharpened_capture is None:
            raise RuntimeError(
                "Call Capture.create_aligned_capture or Capture.radiometric_pan_sharpened_aligned_capture prior to "
                "saving as RGB.")
        if self.__aligned_radiometric_pan_sharpened_capture:
            aligned_capture = self.__aligned_radiometric_pan_sharpened_capture[0]
        else:
            aligned_capture = self.__aligned_capture
        # by default, we don't mask the thermal, since it's native resolution is much lower than the MS
        if lw_index is None:
            lw_index = self.lw_indices()[0]
        masked_thermal = aligned_capture[:, :, lw_index]

        im_display = np.zeros((aligned_capture.shape[0], aligned_capture.shape[1], 3), dtype=np.float32)
        rgb_band_indices = [self.band_names_lower().index('red'),
                            self.band_names_lower().index('green'),
                            self.band_names_lower().index('blue')]

        # for rgb true color, we usually want to use the same min and max scaling across the 3 bands to
        # maintain the "white balance" of the calibrated image
        im_min = np.percentile(aligned_capture[:, :, rgb_band_indices].flatten(),
                               hist_min_percent)  # modify these percentiles to adjust contrast
        im_max = np.percentile(aligned_capture[:, :, rgb_band_indices].flatten(),
                               hist_max_percent)  # for many images, 0.5 and 99.5 are good values
        for dst_band, src_band in enumerate(rgb_band_indices):
            im_display[:, :, dst_band] = imageutils.normalize(aligned_capture[:, :, src_band], im_min, im_max)

        # Compute a histogram
        min_display_therm = np.percentile(masked_thermal, hist_min_percent)
        max_display_therm = np.percentile(masked_thermal, hist_max_percent)

        fig, _ = plotutils.plot_overlay_withcolorbar(im_display,
                                                     masked_thermal,
                                                     figsize=figsize,
                                                     title='Temperature over True Color',
                                                     vmin=min_display_therm, vmax=max_display_therm,
                                                     overlay_alpha=0.25,
                                                     overlay_colormap='jet',
                                                     overlay_steps=16,
                                                     display_contours=True,
                                                     contour_steps=16,
                                                     contour_alpha=.4,
                                                     contour_fmt="%.0fC",
                                                     show=False)
        fig.savefig(outfilename)

    def output(stack, gamma, channel_order=None):
        if channel_order is None:
            channel_order = [2, 1, 0]
        out = stack[:, :, channel_order]
        out -= out.min()
        out /= out.max()
        out = out ** gamma
        scale = out.max()
        return out / scale

    @staticmethod
    def find_inliers(kp_image, kp_ref, matches, *, random_seed: int = 9):

        rng = np.random.default_rng(random_seed)
        model, inliers = ransac((kp_image[matches[:, 0]],
                                 kp_ref[matches[:, 1]]),
                                FundamentalMatrixTransform, min_samples=8,
                                residual_threshold=.25, max_trials=5000,
                                random_state=rng)
        inlier_keypoints_image = kp_image[matches[inliers, 0]]
        inlier_keypoints_ref = kp_ref[matches[inliers, 1]]
        n = len(inlier_keypoints_ref)
        return inlier_keypoints_image, inlier_keypoints_ref, np.array([np.arange(n), np.arange(n)]).T, model

    KeyPoints = namedtuple('KeyPoints', ['kpi', 'kpr', 'match', 'err'])

    @staticmethod
    def filter_keypoints(kp_image, kp_ref, match, w, scale, scale_i, threshold: float = 1.0) -> KeyPoints:
        err = []
        P0 = ProjectiveTransform(matrix=w)
        new_kpi = []
        new_kpr = []
        new_match = []
        cnt = 0
        for m in match:
            # unfortunately the coordinates between skimage and our images are reversed
            a = (kp_ref[m[1]] * scale)[::-1]
            b = (kp_image[m[0]] * scale_i)[::-1]
            e = (np.linalg.norm(P0(a) - b))  # error in pixels
            if e < threshold:
                new_kpi.append(kp_image[m[0]])
                new_kpr.append(kp_ref[m[1]])
                new_match.append([cnt, cnt])
                cnt += 1
            err.append(e)
        return np.array(new_kpi), np.array(new_kpr), np.array(new_match), np.array(err)

    def SIFT_align_capture(self, ref=5, min_matches=10, verbose=0, err_red=10.0, err_blue=12.0,
                           err_LWIR=12.):
        descriptor_extractor = SIFT()
        keypoints = []
        descriptors = []
        img_index = list(range(len(self.images)))
        img_index.pop(ref)
        ref_shape = self.images[ref].raw().shape
        rest_shape = self.images[img_index[0]].raw().shape
        scale = np.array(ref_shape) / np.array(rest_shape)

        # use the calibrated warp matrices to verify keypoints
        warp_matrices_calibrated = self.get_warp_matrices(ref_index=ref)

        if not rest_shape == ref_shape:
            ref_image_SIFT = resize(self.images[ref].undistorted(
                self.images[ref].raw()), rest_shape)
            ref_image_SIFT = (ref_image_SIFT / ref_image_SIFT.max()
                              * 65535).astype(np.uint16)

        descriptor_extractor.detect_and_extract(ref_image_SIFT)
        keypoints_ref = descriptor_extractor.keypoints
        descriptor_ref = descriptor_extractor.descriptors
        if verbose > 1:
            print('found {:d} keypoints in the reference image'.format(len(keypoints_ref)))
        match_images = []
        ratio = []
        filter_tr = []
        img_index = np.array(img_index)
        # extract keypoints % descriptors
        for ix in img_index:
            img = self.images[ix].undistorted(self.images[ix].raw())
            if not img.shape == rest_shape:
                # if we have a thermal image, upsample to match the resolution of the multispec images
                img_base = self.images[ix].raw()[self.images[ix].raw() > 0].min()
                img = img.astype(float)
                img[img > 0] = img[img > 0] - img_base
                img = resize(img, rest_shape)
                img = (img / img.max() * 65535).astype(np.uint16)
                ratio.append(1)
                filter_tr.append(err_LWIR)
            else:
                ratio.append(0.8)
                if ix <= 5:
                    filter_tr.append(err_red)
                else:
                    # less strict filtering for the BLUE images
                    filter_tr.append(err_blue)
            match_images.append(img)
            descriptor_extractor.detect_and_extract(img)
            keypoints.append(descriptor_extractor.keypoints)
            descriptors.append(descriptor_extractor.descriptors)
        if verbose > 1:
            for k, ix in zip(keypoints, img_index):
                print('found {:d} keypoints for band {:} '.format(len(k), self.images[ix].band_name))
            print(' in the remaining stack')

        matches = [match_descriptors(d, descriptor_ref, max_ratio=r)
                   for d, r in zip(descriptors, ratio)]

        # do we have dual camera capture?
        if len(img_index) > 9:
            # if yes, we first match the first channel to the reference
            img_index_Blue = img_index[img_index > 5]
            iBlueREF = img_index_Blue[0]
            # get the warp matrices for the BLUE reference
            warpBLUE = self.get_warp_matrices(ref_index=iBlueREF)
            # this is clunky, since we made our keypoint list WITHOUT the reference - consider refactoring that
            posBLUE = np.where(img_index == iBlueREF)[0][0]
            # we don't know the rig relatives between iBlueREF and ref
            # so we have to rely on SIFT to find the correct transform
            kpi, kpr, imatch, model = self.find_inliers(
                keypoints[posBLUE], keypoints_ref, matches[posBLUE])
            # we trust this match to work
            if len(kpi) < min_matches:
                print('we have just {:d} matching keypoints -the match of BLUE camera to RED failed!!'.format(len(kpi)))
            # if it worked, scale it and get the transform
            scale_i = np.array(self.images[iBlueREF].raw().shape) / np.array(rest_shape)
            P = estimate_transform('projective', (scale * kpr)[:, ::-1], (scale_i * kpi)[:, ::-1])
            warp_blue_ref = P.params
            # now modify the BLUE warp matrices
            for ix in img_index_Blue:
                warp_matrices_calibrated[ix] = np.dot(warp_blue_ref, warpBLUE[ix])

        models = []
        kp_image = []
        kp_ref = []
        for m, k, ix, t in zip(matches, keypoints, img_index, filter_tr):
            # we need to down scale the thermal image for the proper transform
            scale_i = np.array(self.images[ix].raw().shape) / np.array(rest_shape)

            filtered_kpi, filtered_kpr, filtered_match, err = self.filter_keypoints(k,
                                                                                    keypoints_ref,
                                                                                    m,
                                                                                    warp_matrices_calibrated[ix],
                                                                                    scale,
                                                                                    scale_i,
                                                                                    threshold=t)
            if verbose > 0:
                print('found {:d} matching keypoints for index {:d}'.format(len(filtered_match), ix))
            # if we have enough SIFT matches that actually correspond, compute a model
            if len(filtered_match) > min_matches:
                kpi, kpr, imatch, model = self.find_inliers(filtered_kpi,
                                                            filtered_kpr,
                                                            filtered_match)

                P = estimate_transform(
                    'projective', (scale * kpr)[:, ::-1], (scale_i * kpi)[:, ::-1])
            # otherwise, use the calibrated matrix
            # most of the time this will occur for the thermal image, as we have a hard time
            # finding a good matches between panchro & thermal in most cases
            else:
                P = ProjectiveTransform(matrix=warp_matrices_calibrated[ix])
                if verbose > 0:
                    print('no match for index {:d}'.format(ix))
            models.append(P)
            kp_image.append(kpi)
            kp_ref.append(kpr)
            img = self.images[ix].undistorted(self.images[ix].raw())

            # no need for the upsampled stacks here
            if verbose > 0:
                print("Finished aligning band", ix)

        self.__sift_aligned_capture = [np.eye(3)] * len(self.images)
        for ix, m in zip(img_index, models):
            self.__sift_aligned_capture[ix] = m.params

        return self.__sift_aligned_capture

    def adjust_transform(self, ref_index):
        warp_matrices = self.get_warp_matrices(ref_index=ref_index)
        t_matrices = []
        CR = self.images[ref_index].cv2_camera_matrix()
        for i in self.images:
            z = i.location[2] * 1e3
            C = i.cv2_camera_matrix()

            T = (np.array(i.rig_translations) - np.array(self.images[ref_index].rig_translations))
            tm = np.eye(3)
            tm[0, 2] = C[0, 0] * T[0] / z
            tm[1, 2] = C[1, 1] * T[1] / z
            t_matrices.append(tm)
        warp_new = [np.dot(t, w) for w, t in zip(warp_matrices, t_matrices)]
        return warp_new
