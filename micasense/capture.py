#!/usr/bin/env python
# coding: utf-8
"""
MicaSense Capture Class

    A Capture is a set of Images taken by one camera which share the same unique capture identifier (capture_id).
    Generally these images will be found in the same folder and also share the same filename prefix, such
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

import math
import os
import warnings

import cv2
import imageio
import numpy as np

import micasense.image as image
import micasense.imageutils as imageutils
import micasense.plotutils as plotutils

warnings.simplefilter(action="once")


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
        self.panels = None
        self.detected_panel_count = 0
        if panel_corners is None:
            self.panel_corners = [None] * len(self.eo_indices())
        else:
            self.panel_corners = panel_corners

        self.__image_type = None
        self.__aligned_capture = None

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
    def from_file(cls, file_name):
        """
        Create Capture instance from file path.
        :param file_name: str system file path
        :return: Capture object.
        """
        return cls(image.Image(file_name))

    @classmethod
    def from_filelist(cls, file_list):
        """
        Create Capture instance from List of file paths.
        :param file_list: List of str system file paths.
        :return: Capture object.
        """
        if len(file_list) == 0:
            raise IOError("No files provided. Check your file paths.")
        for file in file_list:
            if not os.path.isfile(file):
                raise IOError(f"All files in file list must be a file. The following file is not:\n{file}")
        images = [image.Image(file) for file in file_list]
        return cls(images)

    def __get_reference_index(self):
        """
        Find the reference image which has the smallest rig offsets - they should be (0,0).
        :return: ndarray of ints - The indices of the minimum values along an axis.
        """
        return np.argmin((np.array([i.rig_xy_offset_in_px() for i in self.images]) ** 2).sum(1))

    def __plot(self, images, num_cols=2, plot_type=None, color_bar=True, fig_size=(14, 14)):
        """
        Plot the Images from the Capture.
        :param images: List of Image objects
        :param num_cols: int number of columns
        :param plot_type: str for plot title formatting
        :param color_bar: boolean to determine color bar inclusion
        :param fig_size: Tuple size of the figure
        :return: plotutils result. matplotlib Figure and Axis in both cases.
        """
        if plot_type == None:
            plot_type = ''
        else:
            titles = [
                '{} Band {} {}'.format(str(img.band_name), str(img.band_index),
                                       plot_type if img.band_name.upper() != 'LWIR' else 'Brightness Temperature')
                for img
                in self.images
            ]
        num_rows = int(math.ceil(float(len(self.images)) / float(num_cols)))
        if color_bar:
            return plotutils.subplotwithcolorbar(num_rows, num_cols, images, titles, fig_size)
        else:
            return plotutils.subplot(num_rows, num_cols, images, titles, fig_size)

    def __lt__(self, other):
        return self.utc_time() < other.utc_time()

    def __gt__(self, other):
        return self.utc_time() > other.utc_time()

    def __eq__(self, other):
        return self.uuid == other.uuid

    def location(self):
        """(lat, lon, alt) tuple of WGS-84 location units are radians, meters msl"""
        # TODO: These units are "signed decimal degrees" per metadata.py comments?
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
        self.__image_type = None

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

    def panel_reflectance(self, panel_refl_by_band=None):  # FIXME: panel_refl_by_band parameter isn't used?
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
        self.panels = [Panel(img, panelCorners=pc) for img, pc in zip(self.images, self.panel_corners)]
        self.detected_panel_count = 0
        for p in self.panels:
            if p.panel_detected():
                self.detected_panel_count += 1
        # if panel_corners are defined by hand
        if self.panel_corners is not None and all(corner is not None for corner in self.panel_corners):
            self.detected_panel_count = len(self.panel_corners)
        return self.detected_panel_count

    def plot_panels(self):
        """Plot Panel images."""
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images.")
        self.__plot(
            [p.plot_image() for p in self.panels],
            plot_type='Panels',
            color_bar=False
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

    def create_aligned_capture(self, img_type, irradiance_list=None, warp_matrices=None, normalize=False,
                               motion_type=cv2.MOTION_HOMOGRAPHY):
        """
        Creates aligned Capture. Computes undistorted radiance or reflectance images if necessary.
        :param img_type: str 'radiance' or 'reflectance' depending on image metadata.
        :param irradiance_list: List of mean panel region irradiance.
        :param warp_matrices: 2d List of warp matrices derived from Capture.get_warp_matrices()
        :param normalize: FIXME: This parameter isn't used?
        :param motion_type: OpenCV import. Also know as warp_mode. MOTION_HOMOGRAPHY or MOTION_AFFINE.
                            For Altum images only use HOMOGRAPHY.
        :return: ndarray with alignment changes
        """
        if img_type == 'radiance':
            self.compute_undistorted_radiance()
            self.__image_type = 'radiance'
        elif img_type == 'reflectance':
            # TODO: Handle pre-flight Panel cap + post-flight Panel cap + DLS values
            # Add use_dls or similar user option to properly scale irradiance_list in different configurations.
            # Alternatively if this should be done elsewhere then dls_irradiance() should probably not be used here.

            # if no irradiance values provided, attempt to use DLS
            if irradiance_list is None and self.dls_present():
                irradiance_list = self.dls_irradiance() + [0]
            elif irradiance_list is None and not self.dls_present():
                raise RuntimeError('Reflectance output requested, but no irradiance values given and no DLS values '
                                   'found in image metadata.')

            self.compute_undistorted_reflectance(irradiance_list)
            self.__image_type = 'reflectance'
        else:
            raise RuntimeError('Unknown img_type output requested: {}\nMust be "radiance" or "reflectance".'
                               .format(img_type))

        if warp_matrices is None:
            warp_matrices = self.get_warp_matrices()
        cropped_dimensions, _ = imageutils.find_crop_bounds(self, warp_matrices, warp_mode=motion_type)
        self.__aligned_capture = imageutils.aligned_capture(self,
                                                            warp_matrices,
                                                            motion_type,
                                                            cropped_dimensions,
                                                            None,
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

    def save_capture_as_stack(self, out_file_name, out_data_type='GDT_UInt16',
                              sort_by_wavelength=False, photometric='MINISBLACK'):
        """
        Output the Images in the Capture object as GTiff image stack.
        :param out_file_name: str system file path
        :param out_data_type: str GDT_Float32 or GDT_UInt16
            Default: GDT_UInt16 will write images as scaled reflectance values. EO (32769=100%)
            and LWIR in centi-Kelvin (0-65535).
            GDT_Float32 will write images as floating point reflectance values. EO (1.0=100%)
            and LWIR in floating point Celsius.
            https://gdal.org/api/raster_c_api.html#_CPPv412GDALDataType
        :param sort_by_wavelength: boolean
        :param photometric: str GDAL argument for GTiff color matching
        """
        from osgeo.gdal import GetDriverByName, GDT_UInt16, GDT_Float32
        if self.__aligned_capture is None:
            raise RuntimeError("Call Capture.create_aligned_capture() prior to saving as stack.")

        rows, cols, bands = self.__aligned_capture.shape
        driver = GetDriverByName('GTiff')

        # force correct output datatype
        if self.__image_type == 'radiance':
            gdal_type = GDT_Float32  # force floating point values for radiance and degrees C
        elif self.__image_type == 'reflectance' and out_data_type == 'GDT_UInt16':
            gdal_type = GDT_UInt16
        elif self.__image_type == 'reflectance' and out_data_type == 'GDT_Float32':
            gdal_type = GDT_Float32
        else:
            warnings.warn(message='Output data type in Capture.save_capture_as_bands() was called as {}. '
                                  'Must use "GDT_UInt16" or "GDT_Float32". Defaulting to GDT_UInt16...'
                          .format(out_data_type),
                          category=UserWarning)
            gdal_type = GDT_UInt16

        out_raster = driver.Create(out_file_name, cols, rows, bands, gdal_type,
                                   options=['INTERLEAVE=BAND', 'COMPRESS=DEFLATE', f'PHOTOMETRIC={photometric}'])
        try:
            if out_raster is None:
                raise IOError("Could not load GDAL GeoTiff driver.")

            if sort_by_wavelength:
                eo_list = list(np.argsort(np.array(self.center_wavelengths())[self.eo_indices()]))
            else:
                eo_list = self.eo_indices()

            for out_band, in_band in enumerate(eo_list):
                out_band = out_raster.GetRasterBand(out_band + 1)
                out_data = self.__aligned_capture[:, :, in_band]
                out_data[out_data < 0] = 0
                out_data[out_data > 2] = 2  # limit reflectance data to 200% to allow some specular reflections
                # scale reflectance images so 100% = 32768
                out_band.WriteArray(out_data * 32768 if gdal_type == 2 else out_data)
                out_band.FlushCache()

            for out_band, in_band in enumerate(self.lw_indices()):
                out_band = out_raster.GetRasterBand(len(eo_list) + out_band + 1)
                # scale data from float degC to back to centi-Kelvin to fit into uint16
                out_data = (self.__aligned_capture[:, :, in_band] + 273.15) * 100 if gdal_type == 2 \
                    else self.__aligned_capture[:, :, in_band]
                out_data[out_data < 0] = 0
                out_data[out_data > 65535] = 65535
                out_band.WriteArray(out_data)
                out_band.FlushCache()
        finally:
            out_raster = None

    def save_capture_as_bands(self, out_file_name, out_data_type='GDT_UInt16', photometric='MINISBLACK'):
        """
        Output the Images in the Capture object as separate GTiffs.
        :param out_file_name: str system file path without file extension
        :param out_data_type: str GDT_Float32 or GDT_UInt16
            Default: GDT_UInt16 will write images as scaled reflectance values. EO (32769=100%)
            and LWIR in centi-Kelvin (0-65535).
            GDT_Float32 will write images as floating point reflectance values. EO (1.0=100%)
            and LWIR in floating point Celsius.
            https://gdal.org/api/raster_c_api.html#_CPPv412GDALDataType
        :param photometric: str GDAL argument for GTiff color matching
        :return: None
        """
        from osgeo.gdal import GetDriverByName, GDT_UInt16, GDT_Float32
        if self.__aligned_capture is None:
            raise RuntimeError("Call Capture.create_aligned_capture() prior to saving as stack.")

        # force correct output datatype
        if self.__image_type == 'radiance':
            gdal_type = GDT_Float32  # force floating point values for radiance and degrees C
        elif self.__image_type == 'reflectance' and out_data_type == 'GDT_UInt16':
            gdal_type = GDT_UInt16
        elif self.__image_type == 'reflectance' and out_data_type == 'GDT_Float32':
            gdal_type = GDT_Float32
        else:
            warnings.warn(message='Output data type in Capture.save_capture_as_bands() was called as {}. '
                                  'Must use "GDT_UInt16" or "GDT_Float32". Defaulting to GDT_UInt16...'
                          .format(out_data_type),
                          category=UserWarning)
            gdal_type = GDT_UInt16

        # predictably handle accidental .tif.tif values
        if out_file_name.endswith('.tif'):
            out_file_path = out_file_name[:-4]
        else:
            out_file_path = out_file_name

        rows, cols, bands = self.__aligned_capture.shape
        driver = GetDriverByName('GTiff')

        for i in self.eo_indices():
            out_raster = driver.Create(out_file_path + f'_{i + 1}.tif', cols, rows, 1, gdal_type,
                                       options=['INTERLEAVE=BAND', 'COMPRESS=DEFLATE', f'PHOTOMETRIC={photometric}'])
            out_band = out_raster.GetRasterBand(1)
            out_data = self.__aligned_capture[:, :, i]
            out_data[out_data < 0] = 0
            out_data[out_data > 2] = 2  # limit reflectance data to 200% to allow some specular reflections
            # if GDT_UInt16, scale reflectance images so 100% = 32768. GDT_UInt16 resolves to 2.
            out_band.WriteArray(out_data * 32768 if gdal_type == 2 else out_data)
            out_band.FlushCache()

        for i in self.lw_indices():
            out_raster = driver.Create(out_file_path + f'_{i + 1}.tif', cols, rows, 1, gdal_type,
                                       options=['INTERLEAVE=BAND', 'COMPRESS=DEFLATE', f'PHOTOMETRIC={photometric}'])
            out_band = out_raster.GetRasterBand(1)
            # if GDT_UInt16, scale data from float degC to back to centi-Kelvin to fit into UInt16.
            out_data = (self.__aligned_capture[:, :, i] + 273.15) * 100 if gdal_type == 2 \
                else self.__aligned_capture[:, :, i]
            out_band.WriteArray(out_data)
            out_band.FlushCache()

    def save_capture_as_rgb(self, out_file_name, gamma=1.4, downsample=1, white_balance='norm', hist_min_percent=0.5,
                            hist_max_percent=99.5, sharpen=True, rgb_band_indices=(2, 1, 0)):
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
        if self.__aligned_capture is None:
            raise RuntimeError("Call Capture.create_aligned_capture() prior to saving as RGB.")
        im_display = np.zeros(
            (self.__aligned_capture.shape[0], self.__aligned_capture.shape[1], self.__aligned_capture.shape[2]),
            dtype=np.float32)

        # modify these percentiles to adjust contrast. for many images, 0.5 and 99.5 are good values
        im_min = np.percentile(self.__aligned_capture[:, :, rgb_band_indices].flatten(), hist_min_percent)
        im_max = np.percentile(self.__aligned_capture[:, :, rgb_band_indices].flatten(), hist_max_percent)

        for i in rgb_band_indices:
            # for rgb true color, we usually want to use the same min and max scaling across the 3 bands to
            # maintain the "white balance" of the calibrated image
            if white_balance == 'norm':
                im_display[:, :, i] = imageutils.normalize(self.__aligned_capture[:, :, i], im_min, im_max)
            else:
                im_display[:, :, i] = imageutils.normalize(self.__aligned_capture[:, :, i])

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
            imageio.imwrite(out_file_name, (255 * gamma_corr_rgb).astype('uint8'))
        else:
            imageio.imwrite(out_file_name, (255 * unsharp_rgb).astype('uint8'))

    def save_thermal_over_rgb(self, out_file_name, fig_size=(30, 23), lw_index=None, hist_min_percent=0.2,
                              hist_max_percent=99.8):
        """
        Output the Images in the Capture object as thermal over RGB.
        :param out_file_name: str system file path.
        :param fig_size: Tuple dimensions of the figure.
        :param lw_index: int Index of LWIR Image in Capture.
        :param hist_min_percent: float Minimum histogram percentile.
        :param hist_max_percent: float Maximum histogram percentile.
        """
        if self.__aligned_capture is None:
            raise RuntimeError("Call Capture.create_aligned_capture() prior to saving as RGB.")

        # by default we don't mask the thermal, since it's native resolution is much lower than the MS
        if lw_index is None:
            lw_index = self.lw_indices()[0]
        masked_thermal = self.__aligned_capture[:, :, lw_index]

        im_display = np.zeros((self.__aligned_capture.shape[0], self.__aligned_capture.shape[1], 3), dtype=np.float32)
        rgb_band_indices = [self.band_names_lower().index('red'),
                            self.band_names_lower().index('green'),
                            self.band_names_lower().index('blue')]

        # for rgb true color, we usually want to use the same min and max scaling across the 3 bands to
        # maintain the "white balance" of the calibrated image
        im_min = np.percentile(self.__aligned_capture[:, :, rgb_band_indices].flatten(),
                               hist_min_percent)  # modify these percentiles to adjust contrast
        im_max = np.percentile(self.__aligned_capture[:, :, rgb_band_indices].flatten(),
                               hist_max_percent)  # for many images, 0.5 and 99.5 are good values
        for dst_band, src_band in enumerate(rgb_band_indices):
            im_display[:, :, dst_band] = imageutils.normalize(self.__aligned_capture[:, :, src_band], im_min, im_max)

        # Compute a histogram
        min_display_therm = np.percentile(masked_thermal, hist_min_percent)
        max_display_therm = np.percentile(masked_thermal, hist_max_percent)

        fig, _ = plotutils.plot_overlay_withcolorbar(im_display,
                                                     masked_thermal,
                                                     figsize=fig_size,
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
        fig.savefig(out_file_name)
