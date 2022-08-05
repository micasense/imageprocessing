#!/usr/bin/env python
# coding: utf-8
"""
PanelResolver class

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
import numpy as np
import cv2
import re
import pyzbar.pyzbar as pyzbar

from skimage import measure
import matplotlib.pyplot as plt
import micasense.imageutils as imageutils

class Panel(object):

    def __init__(self, img,panelCorners=None,ignore_autocalibration=False):
        # if we have panel images with QR metadata, panel detection is not called,
        # so this can be forced here 
        if img is None:
            raise IOError("Must provide an image")

        self.image = img
        bias = img.radiance().min()
        scale = (img.radiance().max() - bias)
        self.gray8b = np.zeros(img.radiance().shape, dtype='uint8')
        cv2.convertScaleAbs(img.undistorted(img.radiance()), self.gray8b, 256.0/scale, -1.0*scale*bias)
        
        if (self.image.auto_calibration_image) and ~ignore_autocalibration:
            self.__panel_type = "auto" ## panels the camera found we call auto
            if panelCorners is not None:
                self.__panel_bounds = np.array(panelCorners)
            else:
                self.__panel_bounds = np.array(self.image.panel_region)
            self.panel_albedo = self.image.panel_albedo
            self.serial = self.image.panel_serial
            self.qr_area = None
            self.qr_bounds = None
            self.panel_std = None
            self.saturated_panel_pixels_pct = None
            self.panel_pixels_mean = None
            self.panel_version = None
            if re.search(r'RP\d{2}-(\d{7})-\D{2}', self.image.panel_serial):
                self.serial = self.image.panel_serial
                self.panel_version = int(self.image.panel_serial[2:4])
        else:
            self.__panel_type = "search" ## panels we search for we call search
            self.serial = None
            self.qr_area = None
            self.qr_bounds = None
            self.panel_std = None
            self.saturated_panel_pixels_pct = None
            self.panel_pixels_mean = None
            self.panel_version = None
            if panelCorners is not None:
                self.__panel_bounds = np.array(panelCorners)
            else:
                self.__panel_bounds = None

    def __expect_panel(self):
        return self.image.band_name.upper() != 'LWIR'

    def __find_qr(self):
        decoded = pyzbar.decode(self.gray8b, symbols=[pyzbar.ZBarSymbol.QRCODE])
        for symbol in decoded:
            serial_str = symbol.data.decode('UTF-8')
            m = re.search(r'RP\d{2}-(\d{7})-\D{2}', serial_str)
            if m:
                self.serial = serial_str
                self.panel_version = int(self.serial[2:4])
                self.qr_bounds = []
                for point in symbol.polygon:
                    self.qr_bounds.append([point.x,point.y])
                self.qr_bounds = np.asarray(self.qr_bounds, np.int32)
                self.qr_area = cv2.contourArea(self.qr_bounds)
                # print (symbol.polygon)
                # print (self.qr_bounds)
                break

    def __pt_in_image_bounds(self, pt):
        width, height = self.image.size()
        if pt[0] >= width or pt[0] < 0:
            return False
        if pt[1] >= height or pt[1] < 0:
            return False
        return True
    
    def reflectance_from_panel_serial(self):
        if self.__panel_type == 'auto':
            return self.panel_albedo
        
        if self.serial is None:
            self.__find_qr()
        if self.serial is None:
            raise ValueError("Panel serial number not found")
        if self.panel_version >= 4:
            min_wl = float(self.serial[-14:-10])
            min_rf = float(self.serial[-10:-7])/1000.0
            max_wl = float(self.serial[-7:-3])
            max_rf = float(self.serial[-3:])/1000.0
            c = np.polyfit([min_wl,max_wl], [min_rf,max_rf], 1)
            p = np.poly1d(c)
            return p(self.image.center_wavelength)
        else:
            return None

    def qr_corners(self):
        if self.__panel_type == 'auto':
            return None
        
        if self.qr_bounds is None:
            self.__find_qr()
        return self.qr_bounds

    def panel_detected(self):
        if self.__expect_panel() == False:
            return False
        
        if self.__panel_type == 'auto':
            return True
        
        if self.serial is None:
            self.__find_qr()
        return self.qr_bounds is not None

    def panel_corners(self):
        """ get the corners of a panel region based on the qr code location 
            Our algorithm to do this uses a 'reference' qr code location and
            it's associate panel region.  We find the affine transform
            between the reference qr and our qr, and apply that same transform to the
            reference panel region to find our panel region. Because of a limitation
            of the pyzbar library, the rotation of the absolute QR code isn't known, 
            so we then try all 4 rotations and test against a cost function which is the 
            minimum of the standard devation divided by the mean value for the panel region"""
        if self.__panel_bounds is not None:
            return self.__panel_bounds
        if self.serial is None:
            self.__find_qr()
        if self.serial is None: # didn't find a panel in this image
            return None
        
        if self.panel_version < 3:
            # reference_panel_pts = np.asarray([[894, 469], [868, 232], [630, 258], [656, 496]], 
            #                                 dtype=np.int32)
            # reference_qr_pts = np.asarray([[898, 748], [880, 567], [701, 584], [718, 762]], 
            #                             dtype=np.int32)
            
            # use the actual panel measures here - we use units of [mm]
            # the panel is 154.4 x 152.4 mm , vs. the 84 x 84 mm for the QR code
            # it is left 143.20 mm from the QR code 
            # use the inner 50% square of the panel
            s = 76.2
            p = 42
            T = np.array([-143.2,0])
            
        elif (self.panel_version >= 3) and (self.panel_version<6):
            s = 50
            p = 45
            T = np.array([-145.8,0])
            # reference_panel_pts = np.asarray([[557, 350], [550, 480], [695, 480], [700, 350]], dtype=np.int32)
            # reference_qr_pts = np.asarray([[821, 324], [819, 506], [996, 509], [999, 330]], dtype=np.int32) 
        elif self.panel_version >= 6 :
            # use the actual panel measures here - we use units of [mm]
            # the panel is 100 x 100 mm , vs. the 91 x 91 mm for the QR code
            # it is down 125.94 mm from the QR code 
            # use the inner 50% square of the panel
            p = 41
            s = 50
            T = np.array([0,-130.84])
           
                     
        reference_panel_pts = np.asarray([[-s, s], [s, s], [s, -s], [-s, -s]], dtype=np.float32)*.5+T
        reference_qr_pts = np.asarray([[-p, p], [p, p], [p, -p], [-p, -p]], dtype=np.float32)
        bounds = []
        costs = []
        for rotation in range(0,4):
            qr_points = np.roll(reference_qr_pts, rotation, axis=0)

            src = np.asarray([tuple(row) for row in qr_points[:]], np.float32)
            dst = np.asarray([tuple(row) for row in self.qr_corners()[:]], np.float32)
            
            # we determine the homography from the 4 corner points
            warp_matrix = cv2.getPerspectiveTransform(src,dst)
            
            #warp_matrix = cv2.getAffineTransform(src, dst)

            pts = np.asarray([reference_panel_pts], 'float32')
            panel_bounds = cv2.convexHull(cv2.perspectiveTransform(pts, warp_matrix), clockwise=False)
            panel_bounds = np.squeeze(panel_bounds) # remove nested lists
            
            bounds_in_image = True
            for i, point in enumerate(panel_bounds):
                if not self.__pt_in_image_bounds(point):
                    bounds_in_image = False
            if bounds_in_image:
                mean, std, _, _ = self.region_stats(self.image.raw(),panel_bounds, sat_threshold=65000)
                bounds.append(panel_bounds.astype(np.int32))
                costs.append(std/mean)

        idx = costs.index(min(costs))
        self.__panel_bounds = bounds[idx]
        return self.__panel_bounds

    def ordered_panel_coordinates(self):
        """
        Return panel region coordinates in a predictable order. Panel region coordinates that are automatically
        detected by the camera are ordered differently than coordinates detected by Panel.panel_corners().
        :return: [ (ur), (ul), (ll), (lr) ] to mirror Image.panel_region attribute order
        """
        pc = self.panel_corners()
        pc = sorted(pc, key=lambda x: x[0])

        # get the coordinates on the "left" and "right" side of the bounding box
        left_coords = pc[:2]
        right_coords = pc[2:]

        # sort y values ascending for correct order
        left_coords = sorted(left_coords, key=lambda y: y[0])
        right_coords = sorted(right_coords, key=lambda y: y[0])

        return [tuple(right_coords[1]), tuple(left_coords[1]), tuple(left_coords[0]), tuple(right_coords[0])]

    def region_stats(self, img, region, sat_threshold=None):
        """Provide regional statistics for a image over a region
        Inputs: img is any image ndarray, region is a skimage shape
        Outputs: mean, std, count, and saturated count tuple for the region"""
        rev_panel_pts = np.fliplr(region) #skimage and opencv coords are reversed
        w, h = img.shape
        mask = measure.grid_points_in_poly((w,h),rev_panel_pts)
        num_pixels = mask.sum()
        panel_pixels = img[mask]
        stdev = panel_pixels.std()
        mean_value = panel_pixels.mean()
        saturated_count = 0
        if sat_threshold is not None:
            saturated_count = (panel_pixels > sat_threshold).sum()
            #set saturated pixels here
            if num_pixels>0:
                self.saturated_panel_pixels_pct = (100.0*saturated_count)/num_pixels
        return mean_value, stdev, num_pixels, saturated_count
        
    def raw(self):
        raw_img = self.image.undistorted(self.image.raw())
        return self.region_stats(raw_img,
                                 self.panel_corners(),
                                 sat_threshold=65000)
    def intensity(self):
        intensity_img = self.image.undistorted(self.image.intensity())
        return self.region_stats(intensity_img,
                                 self.panel_corners(),
                                 sat_threshold=65000)
    def radiance(self):
        radiance_img = self.image.undistorted(self.image.radiance())
        return self.region_stats(radiance_img,
                                 self.panel_corners())
    
    def reflectance_mean(self):
        reflectance_image = self.image.reflectance()
        if reflectance_image is None:
            print("First calculate the reflectance image by providing a\n band specific irradiance to the calling image.reflectance(irradiance)")
        mean, _, _, _ = self.region_stats(reflectance_image,
                                          self.panel_corners())
        return mean

    def irradiance_mean(self, reflectance):
        radiance_mean, _, _, _ = self.radiance()
        return radiance_mean * math.pi / reflectance

    def plot_image(self):
        display_img = cv2.cvtColor(self.gray8b,cv2.COLOR_GRAY2RGB)
        if self.panel_detected():
            if self.qr_corners() is not None:
                cv2.drawContours(display_img,[self.qr_corners()], 0, (255, 0, 0), 3)
            cv2.drawContours(display_img,[self.panel_corners()], 0, (0,0, 255), 3)

        font = cv2.FONT_HERSHEY_DUPLEX
        if self.panel_detected():
            if self.qr_corners() is not None:
                xloc = self.qr_corners()[0][0]-100
                yloc = self.qr_corners()[0][1]+100
            else:
                xloc = self.panel_corners()[0][0]-100
                yloc = self.panel_corners()[0][1]+100
            cv2.putText(display_img, str(self.serial).split('_')[0], (xloc,yloc), font, 1, 255, 2)
        return display_img

    def plot(self, figsize=(14,14)):
        display_img = self.plot_image()
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(display_img)
        plt.tight_layout()
        plt.show()
        return fig, ax
