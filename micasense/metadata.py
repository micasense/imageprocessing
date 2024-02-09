#!/usr/bin/env python
# coding: utf-8
"""
RedEdge Metadata Management Utilities

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
# Support strings in Python 2 and 3
from __future__ import unicode_literals

import math
import os
from datetime import datetime, timedelta

import exiftool
import pytz


class Metadata(object):
    """ Container for Micasense image metadata"""

    def __init__(self, filename: str, exiftool_path=None, exiftool_obj=None):
        if exiftool_obj is not None:
            self.exif = exiftool_obj.get_metadata(filename)
            return
        if exiftool_path is not None:
            self.exiftoolPath = exiftool_path
        elif os.environ.get('exiftoolpath') is not None:
            self.exiftoolPath = os.path.normpath(os.environ.get('exiftoolpath'))
        else:
            self.exiftoolPath = None
        if not os.path.isfile(filename):
            raise IOError("Input path is not a file")
        with exiftool.ExifToolHelper() as exift:
            self.exif = exift.get_metadata(filename)

    def get_all(self):
        """ Get all extracted metadata items """
        return self.exif

    def get_item(self, item, index=None):
        """ Get metadata item by Namespace:Parameter"""
        val = None
        try:
            assert len(self.exif) > 0
            val = self.exif[0][item]
            if index is not None:
                try:
                    if isinstance(val, unicode):
                        val = val.encode('ascii', 'ignore')
                except NameError:
                    # throws on python 3 where unicode is undefined
                    pass
                if isinstance(val, str) and len(val.split(',')) > 1:
                    val = val.split(',')
                val = val[index]
        except KeyError:
            pass
        except IndexError:
            print("Item {0} is length {1}, index {2} is outside this range.".format(
                item,
                len(self.exif[item]),
                index))
        return val

    def size(self, item):
        """get the size (length) of a metadata item"""
        val = self.get_item(item)
        try:
            if isinstance(val, unicode):
                val = val.encode('ascii', 'ignore')
        except NameError:
            # throws on python 3 where unicode is undefined
            pass
        if isinstance(val, str) and len(val.split(',')) > 1:
            val = val.split(',')
        if val is not None:
            return len(val)
        else:
            return 0

    def print_all(self):
        for item in self.get_all():
            print("{}: {}".format(item, self.get_item(item)))

    def dls_present(self):
        return self.get_item("XMP:Irradiance") is not None \
            or self.get_item("XMP:HorizontalIrradiance") is not None \
            or self.get_item("XMP:DirectIrradiance") is not None

    def supports_radiometric_calibration(self):
        if (self.get_item('XMP:RadiometricCalibration')) is None:
            return False
        return True

    def position(self):
        """get the WGS-84 latitude, longitude tuple as signed decimal degrees"""
        lat = self.get_item('EXIF:GPSLatitude')
        latref = self.get_item('EXIF:GPSLatitudeRef')
        if latref == 'S':
            lat *= -1.0
        lon = self.get_item('EXIF:GPSLongitude')
        lonref = self.get_item('EXIF:GPSLongitudeRef')
        if lonref == 'W':
            lon *= -1.0
        alt = self.get_item('EXIF:GPSAltitude')
        return lat, lon, alt

    def utc_time(self):
        """ Get the timezone-aware datetime of the image capture """
        str_time = self.get_item('EXIF:DateTimeOriginal')
        if str_time:
            utc_time = datetime.strptime(str_time, "%Y:%m:%d %H:%M:%S")
            subsec = float(f"0.{self.get_item('EXIF:SubSecTime')}")
            negative = 1.0
            if subsec < 0:
                negative = -1.0
                subsec *= -1.0
            subsec *= negative
            ms = subsec * 1e3
            utc_time += timedelta(milliseconds=ms)
            timezone = pytz.timezone('UTC')
            utc_time = timezone.localize(utc_time)
        else:
            utc_time = None
        return utc_time

    def dls_pose(self):
        """ get DLS pose as local earth-fixed yaw, pitch, roll in radians """
        if self.get_item('XMP:Yaw') is not None:
            yaw = float(self.get_item('XMP:Yaw'))  # should be XMP.DLS.Yaw, but exiftool doesn't expose it that way
            pitch = float(self.get_item('XMP:Pitch'))
            roll = float(self.get_item('XMP:Roll'))
        else:
            yaw = pitch = roll = 0.0
        return yaw, pitch, roll

    def rig_relatives(self):
        if self.get_item('XMP:RigRelatives') is not None:
            nelem = self.size('XMP:RigRelatives')
            return [float(self.get_item('XMP:RigRelatives', i)) for i in range(nelem)]
        else:
            return None

    def rig_translations(self):
        if self.get_item('XMP:RigTranslations') is not None:
            nelem = self.size('XMP:RigTranslations')
            return [float(self.get_item('XMP:RigTranslations', i)) for i in range(nelem)]
        else:
            return None

    def capture_id(self):
        return self.get_item('XMP:CaptureId')

    def flight_id(self):
        return self.get_item('XMP:FlightId')

    def camera_make(self):
        return self.get_item('EXIF:Make')

    def camera_model(self):
        return self.get_item('EXIF:Model')

    def camera_serial(self):
        return self.get_item('EXIF:SerialNumber')

    def firmware_version(self):
        return self.get_item('EXIF:Software')

    def band_name(self):
        return self.get_item('XMP:BandName')

    def band_index(self):
        return self.get_item('XMP:RigCameraIndex')

    def exposure(self):
        exp = self.get_item('EXIF:ExposureTime')
        # correct for incorrect exposure in some legacy RedEdge firmware versions
        if self.camera_model() != "Altum":
            if math.fabs(exp - (1.0 / 6329.0)) < 1e-6:
                exp = 0.000274
        return exp

    def gain(self):
        return self.get_item('EXIF:ISOSpeed') / 100.0

    def image_size(self):
        return self.get_item('EXIF:ImageWidth'), self.get_item('EXIF:ImageHeight')

    def center_wavelength(self):
        return self.get_item('XMP:CentralWavelength')

    def bandwidth(self):
        return self.get_item('XMP:WavelengthFWHM')

    def radiometric_cal(self):
        nelem = self.size('XMP:RadiometricCalibration')
        return [float(self.get_item('XMP:RadiometricCalibration', i)) for i in range(nelem)]

    def black_level(self):
        if self.get_item('EXIF:BlackLevel') is None:
            return 0
        black_lvl = self.get_item('EXIF:BlackLevel').split(' ')
        total = 0.0
        num = len(black_lvl)
        for pixel in black_lvl:
            total += float(pixel)
        return total / float(num)

    def dark_pixels(self):
        """ get the average of the optically covered pixel values
        Note: these pixels are raw, and have not been radiometrically
              corrected. Use the black_level() method for all
              radiomentric calibrations """
        dark_pixels = self.get_item('XMP:DarkRowValue')
        total = 0.0
        num = len(dark_pixels)
        for pixel in dark_pixels:
            total += float(pixel)
        return total / float(num)

    def bits_per_pixel(self):
        """ get the number of bits per pixel, which defines the maximum digital number value in the image """
        return self.get_item('EXIF:BitsPerSample')

    def vignette_center(self):
        """ get the vignette center in X and Y image coordinates"""
        nelem = self.size('XMP:VignettingCenter')
        return [float(self.get_item('XMP:VignettingCenter', i)) for i in range(nelem)]

    def vignette_polynomial(self):
        """ get the radial vignette polynomial in the order it's defined in the metadata"""
        nelem = self.size('XMP:VignettingPolynomial')
        return [float(self.get_item('XMP:VignettingPolynomial', i)) for i in range(nelem)]

    def vignette_polynomial2Dexponents(self):
        """ get exponents of the 2D polynomial """
        nelem = self.size('XMP:VignettingPolynomial2DName')
        return [float(self.get_item('XMP:VignettingPolynomial2DName', i)) for i in range(nelem)]

    def vignette_polynomial2D(self):
        """ get the 2D polynomial coefficients in the order it's defined in the metadata"""
        nelem = self.size('XMP:VignettingPolynomial2D')
        return [float(self.get_item('XMP:VignettingPolynomial2D', i)) for i in range(nelem)]

    def distortion_parameters(self):
        nelem = self.size('XMP:PerspectiveDistortion')
        return [float(self.get_item('XMP:PerspectiveDistortion', i)) for i in range(nelem)]

    def principal_point(self):
        if self.get_item('XMP:PrincipalPoint') is not None:
            return [float(item) for item in self.get_item('XMP:PrincipalPoint').split(',')]
        else:
            return [0, 0]

    def focal_plane_resolution_px_per_mm(self):
        if self.get_item('EXIF:FocalPlaneXResolution') is not None:
            fp_x_resolution = float(self.get_item('EXIF:FocalPlaneXResolution'))
            fp_y_resolution = float(self.get_item('EXIF:FocalPlaneYResolution'))
        else:
            fp_x_resolution, fp_y_resolution = 0, 0
        return fp_x_resolution, fp_y_resolution

    def focal_length_mm(self):
        units = self.get_item('XMP:PerspectiveFocalLengthUnits')
        focal_length_mm = 0.0
        if units is not None:
            if units == 'mm':
                focal_length_mm = float(self.get_item('XMP:PerspectiveFocalLength'))
            else:
                focal_length_px = float(self.get_item('XMP:PerspectiveFocalLength'))
                focal_length_mm = focal_length_px / self.focal_plane_resolution_px_per_mm()[0]
        return focal_length_mm

    def focal_length_35_mm_eq(self):
        return float(self.get_item('Composite:FocalLength35efl'))

    @staticmethod
    def __float_or_zero(val):
        if val is not None:
            return float(val)
        else:
            return 0.0

    def irradiance_scale_factor(self):
        """ Get the calibration scale factor for the irradiance measurements in this image metadata.
            Due to calibration differences between DLS1 and DLS2, we need to account for a scale factor
            change in their respective units. This scale factor is pulled from the image metadata, or, if
            the metadata doesn't give us the scale, we assume one based on a known combination of tags"""
        if self.get_item('XMP:IrradianceScaleToSIUnits') is not None:
            # the metadata contains the scale
            scale_factor = self.__float_or_zero(self.get_item('XMP:IrradianceScaleToSIUnits'))
        elif self.get_item('XMP:HorizontalIrradiance') is not None:
            # DLS2 but the metadata is missing the scale, assume 0.01
            scale_factor = 0.01
        else:
            # DLS1, so we use a scale of 1
            scale_factor = 1.0
        return scale_factor

    def horizontal_irradiance_valid(self):
        """ Defines if horizontal irradiance tag contains a value that can be trusted
            some firmware versions had a bug whereby the direct and scattered irradiance were correct,
            but the horizontal irradiance was calculated incorrectly """
        if self.get_item('XMP:HorizontalIrradiance') is None:
            return False
        from packaging import version
        version_string = self.firmware_version().strip('v')
        if self.camera_model() == "Altum":
            good_version = "1.2.3"
        elif self.camera_model() == 'RedEdge' or self.camera_model() == 'RedEdge-M':
            good_version = "5.1.7"
        elif self.camera_model() == 'RedEdge-P':
            return True
        elif self.camera_model() == 'Altum-PT':
            return True
        else:
            raise ValueError("Camera model is required to be RedEdge or Altum, not {} ".format(self.camera_model()))
        return version.parse(version_string) >= version.parse(good_version)

    def spectral_irradiance(self):
        """ Raw spectral irradiance measured by an irradiance sensor.
            Calibrated to W/m^2/nm using irradiance_scale_factor, but not corrected for angles """
        return self.__float_or_zero(self.get_item('XMP:SpectralIrradiance')) * self.irradiance_scale_factor()

    def horizontal_irradiance(self):
        """ Horizontal irradiance at the earth's surface below the DLS on the plane normal to the gravity
            vector at the location (local flat plane spectral irradiance) """
        return self.__float_or_zero(self.get_item('XMP:HorizontalIrradiance')) * self.irradiance_scale_factor()

    def scattered_irradiance(self):
        """ scattered component of the spectral irradiance """
        return self.__float_or_zero(self.get_item('XMP:ScatteredIrradiance')) * self.irradiance_scale_factor()

    def direct_irradiance(self):
        """ direct component of the spectral irradiance on a ploane normal to the vector towards the sun """
        return self.__float_or_zero(self.get_item('XMP:DirectIrradiance')) * self.irradiance_scale_factor()

    def solar_azimuth(self):
        """ solar azimuth at the time of capture, as calculated by the camera system """
        return self.__float_or_zero(self.get_item('XMP:SolarAzimuth'))

    def solar_elevation(self):
        """ solar elevation at the time of capture, as calculated by the camera system """
        return self.__float_or_zero(self.get_item('XMP:SolarElevation'))

    def estimated_direct_vector(self):
        """ estimated direct light vector relative to the DLS2 reference frame"""
        if self.get_item('XMP:EstimatedDirectLightVector') is not None:
            return [self.__float_or_zero(item) for item in self.get_item('XMP:EstimatedDirectLightVector')]
        else:
            return None

    def auto_calibration_image(self):
        """ True if this image is an auto-calibration image, where the camera has found and identified
            a calibration panel """
        cal_tag = self.get_item('XMP:CalibrationPicture')
        return cal_tag is not None and \
            cal_tag == 2 and \
            self.panel_albedo() is not None and \
            self.panel_region() is not None and \
            self.panel_serial() is not None

    def panel_albedo(self):
        """ Surface albedo of the active portion of the reflectance panel as calculated by the camera
            (usually from the information in the panel QR code) """
        albedo = self.get_item('XMP:Albedo')
        if albedo is not None:
            return self.__float_or_zero(albedo)
        return albedo

    def panel_region(self):
        """ A 4-tuple containing image x,y coordinates of the panel active area """
        if self.get_item('XMP:ReflectArea') is not None:
            coords = [int(item) for item in self.get_item('XMP:ReflectArea').split(',')]
            return list(zip(coords[0::2], coords[1::2]))
        else:
            return None

    def panel_serial(self):
        """ The panel serial number as extracted from the image by the camera """
        return self.get_item('XMP:PanelSerial')
