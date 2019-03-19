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

import exiftool
from datetime import datetime, timedelta
import pytz
import os
import math

class Metadata(object):
    ''' Container for Micasense image metadata'''
    def __init__(self, filename, exiftoolPath=None):
        self.xmpfile = None
        if exiftoolPath is not None:
            self.exiftoolPath = exiftoolPath
        elif os.environ.get('exiftoolpath') is not None:
            self.exiftoolPath = os.path.normpath(os.environ.get('exiftoolpath'))
        else:
            self.exiftoolPath = None
        if not os.path.isfile(filename):
            raise IOError("Input path is not a file")
        with exiftool.ExifTool(self.exiftoolPath) as exift:
            self.exif = exift.get_metadata(filename)

    def get_all(self):
        ''' Get all extracted metadata items '''
        return self.exif

    def get_item(self, item, index=None):
        ''' Get metadata item by Namespace:Parameter'''
        val = None
        try:
            val = self.exif[item]
            if index is not None:
                try:
                    if isinstance(val, unicode):
                        val = val.encode('ascii','ignore')
                except NameError:
                    #throws on python 3 where unicode is undefined
                    pass
                if isinstance(val,str) and len(val.split(',')) > 1:
                    val = val.split(',')
                val = val[index]
        except KeyError:
            #print ("Item "+item+" not found")
            pass
        except IndexError:
            print("Item {0} is length {1}, index {2} is outside this range.".format(
                item,
                len(self.exif[item]),
                index))
        return val

    def size(self, item):
        '''get the size (length) of a metadata item'''
        val = self.get_item(item)
        try:
            if isinstance(val, unicode):
                val = val.encode('ascii','ignore')
        except NameError:
            #throws on python 3 where unicode is undefined
            pass
        if isinstance(val,str) and len(val.split(',')) > 1:
            val = val.split(',')
        if val is not None:
            return len(val)
        else:
            return 0

    def print_all(self):
        for item in self.get_all():
            print("{}: {}".format(item, self.get_item(item)))

    def dls_present(self):
        return self.get_item("XMP:Irradiance") is not None

    def supports_radiometric_calibration(self):
        if(self.get_item('XMP:RadiometricCalibration')) is None:
            return False
        return True

    def position(self):
        '''get the WGS-84 latitude, longitude tuple as signed decimal degrees'''
        lat = self.get_item('EXIF:GPSLatitude')
        latref = self.get_item('EXIF:GPSLatitudeRef')
        if latref=='S':
            lat *= -1.0
        lon = self.get_item('EXIF:GPSLongitude')
        lonref = self.get_item('EXIF:GPSLongitudeRef')
        if lonref=='W':
            lon *= -1.0
        alt = self.get_item('EXIF:GPSAltitude')
        return lat, lon, alt

    def utc_time(self):
        ''' Get the timezone-aware datetime of the image capture '''
        str_time = self.get_item('EXIF:DateTimeOriginal')
        utc_time = datetime.strptime(str_time, "%Y:%m:%d %H:%M:%S")
        subsec = int(self.get_item('EXIF:SubSecTime'))
        negative = 1.0
        if subsec < 0:
            negative = -1.0
            subsec *= -1.0
        subsec = float('0.{}'.format(int(subsec)))
        subsec *= negative
        ms = subsec * 1e3
        utc_time += timedelta(milliseconds = ms)
        timezone = pytz.timezone('UTC')
        utc_time = timezone.localize(utc_time)
        return utc_time

    def dls_pose(self):
        ''' get DLS pose as local earth-fixed yaw, pitch, roll in radians '''
        if self.get_item('XMP:Yaw') is not None:
            yaw = float(self.get_item('XMP:Yaw')) # should be XMP.DLS.Yaw, but exiftool doesn't expose it that way
            pitch = float(self.get_item('XMP:Pitch'))
            roll = float(self.get_item('XMP:Roll'))
        else:
            yaw = pitch = roll = 0.0
        return yaw, pitch, roll

    def dls_irradiance(self):
        if self.get_item('XMP:SpectralIrradiance') is not None:
            irr = float(self.get_item('XMP:SpectralIrradiance'))
        else:
            irr = 0.0
        return irr

    def rig_relatives(self):
        if self.get_item('XMP:RigRelatives') is not None:
            nelem = self.size('XMP:RigRelatives')
            return [float(self.get_item('XMP:RigRelatives', i)) for i in range(nelem)]
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

    def firmware_version(self):
        return self.get_item('EXIF:Software')

    def band_name(self):
        return self.get_item('XMP:BandName')

    def band_index(self):
        return self.get_item('XMP:RigCameraIndex')

    def exposure(self):
        exp = self.get_item('EXIF:ExposureTime')
        if math.fabs(exp-(1.0/6329.0)) < 0.0001:
            exp = 0.000274
        return exp

    def gain(self):
        return self.get_item('EXIF:ISOSpeed')/100.0

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
        return total/float(num)

    def dark_pixels(self):
        ''' get the average of the optically covered pixel values
        Note: these pixels are raw, and have not been radiometrically
              corrected. Use the black_level() method for all
              radiomentric calibrations '''
        dark_pixels = self.get_item('XMP:DarkRowValue')
        total = 0.0
        num = len(dark_pixels)
        for pixel in dark_pixels:
            total += float(pixel)
        return total/float(num)

    def bits_per_pixel(self):
        ''' get the number of bits per pixel, which defines the maximum digital number value in the image '''
        return self.get_item('EXIF:BitsPerSample')

    def vignette_center(self):
        ''' get the vignette center in X and Y image coordinates'''
        nelem = self.size('XMP:VignettingCenter')
        return [float(self.get_item('XMP:VignettingCenter', i)) for i in range(nelem)]

    def vignette_polynomial(self):
        ''' get the radial vignette polynomial in the order it's defined in the metadata'''
        nelem = self.size('XMP:VignettingPolynomial')
        return [float(self.get_item('XMP:VignettingPolynomial', i)) for i in range(nelem)]

    def distortion_parameters(self):
        nelem = self.size('XMP:PerspectiveDistortion')
        return [float(self.get_item('XMP:PerspectiveDistortion', i)) for i in range(nelem)]

    def principal_point(self):
        return [float(item) for item in self.get_item('XMP:PrincipalPoint').split(',')]

    def focal_plane_resolution_px_per_mm(self):
        fp_x_resolution = float(self.get_item('EXIF:FocalPlaneXResolution'))
        fp_y_resolution = float(self.get_item('EXIF:FocalPlaneYResolution'))
        return fp_x_resolution, fp_y_resolution

    def focal_length_mm(self):
        units = self.get_item('XMP:PerspectiveFocalLengthUnits')
        focal_length_mm = 0.0
        if units == 'mm':
            focal_length_mm = float(self.get_item('XMP:PerspectiveFocalLength'))
        else:
            focal_length_px = float(self.get_item('XMP:PerspectiveFocalLength'))
            focal_length_mm = focal_length_px / self.focal_plane_resolution_px_per_mm()[0]
        return focal_length_mm

    def __float_or_zero(self, str):
        if str is not None:
            return float(str)
        else:
            return 0.0

    def spectral_irradiance(self):
        return self.__float_or_zero(self.get_item('XMP:SpectralIrradiance'))

    def horizontal_irradiance(self):
        return self.__float_or_zero(self.get_item('XMP:HorizontalIrradiance'))/100.0

    def scattered_irradiance(self):
        return self.__float_or_zero(self.get_item('XMP:ScatteredIrradiance'))

    def direct_irradiance(self):
        return self.__float_or_zero(self.get_item('XMP:DirectIrradiance'))

    def solar_azimuth(self):
        return self.__float_or_zero(self.get_item('XMP:SolarAzimuth'))

    def solar_elevation(self):
        return self.__float_or_zero(self.get_item('XMP:SolarElevation'))

    def estimated_direct_vector(self):
        if self.get_item('XMP:EstimatedDirectLightVector') is not None:
            return [self.__float_or_zero(item) for item in self.get_item('XMP:EstimatedDirectLightVector')]
        else:
            return None

    def auto_calibration_image(self):
        cal_tag = self.get_item('XMP:CalibrationPicture')
        return cal_tag is not None and \
               cal_tag == 2 and \
               self.panel_albedo() is not None and \
               self.panel_region() is not None and \
               self.panel_serial() is not None

    def panel_albedo(self):
        albedo = self.get_item('XMP:Albedo')
        if albedo is not None:
            return self.__float_or_zero(albedo)
        return albedo

    def panel_region(self):
        if self.get_item('XMP:ReflectArea') is not None:
            coords = [int(item) for item in self.get_item('XMP:ReflectArea').split(',')]
            return list(zip(coords[0::2], coords[1::2]))
        else:
            return None

    def panel_serial(self):
        return self.get_item('XMP:PanelSerial')
