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
import micasense.image as image
import micasense.dls as dls
import micasense.plotutils as plotutils
import micasense.imageutils as imageutils
import math
import numpy as np
import cv2
import os
import imageio

class Capture(object):
    """
    A capture is a set of images taken by one RedEdge cameras which share
    the same unique capture identifier.  Generally these images will be
    found in the same folder and also share the same filename prefix, such
    as IMG_0000_*.tif, but this is not required
    """
    def __init__(self, images, panelCorners=[None]*5):
        if isinstance(images, image.Image):
            self.images = [images]
        elif isinstance(images, list):
            self.images = images
        else:
            raise RuntimeError("Provide an image or list of images to create a Capture")
        self.num_bands = len(self.images)
        self.images.sort()
        capture_ids = [img.capture_id for img in self.images]
        if len(set(capture_ids)) != 1:
            raise RuntimeError("Images provided are required to all have the same capture id")
        self.uuid = self.images[0].capture_id
        self.panels = None
        self.detected_panel_count = 0
        self.panelCorners = panelCorners

        self.__aligned_capture = None

    def set_panelCorners(self,panelCorners):
        self.panelCorners = panelCorners
        self.panels = None
        self.detect_panels()

    def append_image(self, image):
        if self.uuid != image.capture_id:
            raise RuntimeError("Added images must have the same capture id")
        self.images.append(image)
        self.images.sort()

    def append_images(self, images):
        [self.append_image(img) for img in images]

    def append_file(self, file_name):
        self.append_image(image.Image(file_name))
    
    @classmethod
    def from_file(cls, file_name):
        return cls(image.Image(file_name))

    @classmethod
    def from_filelist(cls, file_list):
        if len(file_list) == 0:
            raise IOError("No files provided. Check your file paths")
        for fle in file_list:
            if not os.path.isfile(fle):
                raise IOError("All files in file list must be a file. The following file is not:\nfle")
        images = [image.Image(fle) for fle in file_list]
        return cls(images)

    def __get_reference_index(self):
        # find the reference image which has the smallest rig offsets - they should be (0,0)
        return np.argmin((np.array([i.rig_xy_offset_in_px() for i in self.images])**2).sum(1))

    def __plot(self, imgs, num_cols=2, plot_type=None, colorbar=True, figsize=(14, 14)):
        ''' plot the radiance images for the capture '''
        if plot_type == None:
            plot_type = ''
        else:
            titles = [
                '{} Band {} {}'.format(str(img.band_name), str(img.band_index), plot_type if img.band_name.upper() != 'LWIR' else 'Brightness Temperature')
                for img
                in self.images
            ]
        num_rows = int(math.ceil(float(len(self.images))/float(num_cols)))
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
        ''' (lat, lon, alt) tuple of WGS-84 location units are radians, meters msl'''
        return (self.images[0].location)

    def utc_time(self):
        ''' returns a timezone-aware datetime object of the capture time '''
        return self.images[0].utc_time

    def clear_image_data(self):
        '''Clears (dereferences to allow garbage collection) all internal image
           data stored in this class.  Call this after processing-heavy image
           calls to manage program memory footprint.  When processing many images,
           such as iterating over the captures in an ImageSet, it may be necessary
           to call this after capture is processed'''
        for img in self.images:
            img.clear_image_data()
        self.__aligned_capture = None

    def center_wavelengths(self):
        '''Returns a list of the image center wavelenghts in nanometers'''
        return [img.center_wavelength for img in self.images]

    def band_names(self):
        '''Returns a list of the image band names'''
        return [img.band_name for img in self.images]

    def dls_present(self):
        '''Returns true if DLS metadata is present in the images'''
        return self.images[0].dls_present

    def dls_irradiance_raw(self):
        '''Returns a list of the raw DLS measurements from the image metadata'''
        return [img.spectral_irradiance for img in self.images]

    def dls_irradiance(self):
        '''Returns a list of the corrected earth-surface (horizontal) DLS irradiance in W/m^2/nm'''
        return [img.horizontal_irradiance for img in self.images]

    def dls_pose(self):
        '''Returns (yaw,pitch,roll) tuples in radians of the earth-fixed dls pose'''
        return (self.images[0].dls_yaw, self.images[0].dls_pitch, self.images[0].dls_roll)

    def plot_raw(self):
        '''Plot raw images as the data came from the camera'''
        self.__plot([img.raw() for img in self.images],
                    plot_type='Raw')

    def plot_vignette(self):
        '''Compute (if necessary) and plot vignette correction images'''
        self.__plot([img.vignette()[0].T for img in self.images],
                    plot_type='Vignette')

    def plot_radiance(self):
        '''Compute (if necessary) and plot radiance images'''
        self.__plot([img.radiance() for img in self.images],
                    plot_type='Radiance')

    def plot_undistorted_radiance(self):
        '''Compute (if necessary) and plot undistorted radiance images'''
        self.__plot(
                    [img.undistorted(img.radiance()) for img in self.images],
                    plot_type='Undistored Radiance')

    def plot_undistorted_reflectance(self, irradiance_list):
        '''Compute (if necessary) and plot reflectances given a list of irrdiances'''
        self.__plot(
                    self.undistorted_reflectance(irradiance_list),
                    plot_type='Undistorted Reflectance')

    def compute_radiance(self):
        [img.radiance() for img in self.images]

    def compute_undistorted_radiance(self):
        [img.undistorted_radiance() for img in self.images]

    def compute_reflectance(self, irradiance_list=None, force_recompute=True):
        '''Compute image reflectance from irradiance list, but don't return'''
        if irradiance_list is not None:
            [img.reflectance(irradiance_list[i], force_recompute=force_recompute) for i,img in enumerate(self.images)]
        else:
            [img.reflectance(force_recompute=force_recompute) for img in self.images]

    def compute_undistorted_reflectance(self, irradiance_list=None, force_recompute=True):
        '''Compute image reflectance from irradiance list, but don't return'''
        if irradiance_list is not None:
            [img.undistorted_reflectance(irradiance_list[i], force_recompute=force_recompute) for i,img in enumerate(self.images)]
        else:
            [img.undistorted_reflectance(force_recompute=force_recompute) for img in self.images]


    def eo_images(self):
        return [img for img in self.images if img.band_name != 'LWIR']
    def lw_images(self):
        return [img for img in self.images if img.band_name == 'LWIR']

    def reflectance(self, irradiance_list):
        '''Comptute and return list of reflectance images for given irradiance'''
        eo_imgs = [img.reflectance(irradiance_list[i]) for i,img in enumerate(self.eo_images())]
        lw_imgs = [img.reflectance() for i,img in enumerate(self.lw_images())]
        return eo_imgs + lw_imgs

    def undistorted_reflectance(self, irradiance_list):
        '''Comptute and return list of reflectance images for given irradiance'''
        eo_imgs = [img.undistorted(img.reflectance(irradiance_list[i])) for i,img in enumerate(self.eo_images())]
        lw_imgs = [img.undistorted(img.reflectance()) for i,img in enumerate(self.lw_images())]
        return eo_imgs + lw_imgs

    def panels_in_all_expected_images(self):
        expected_panels = sum(str(img.band_name).upper() != 'LWIR' for img in self.images)
        return self.detect_panels() == expected_panels

    def panel_raw(self):
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images")
        raw_list = []
        for p in self.panels:
            mean, _, _, _ = p.raw()
            raw_list.append(mean)
        return raw_list

    def panel_radiance(self):
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images")
        radiance_list = []
        for p in self.panels:
            mean, _, _, _ = p.radiance()
            radiance_list.append(mean)
        return radiance_list

    def panel_irradiance(self, reflectances=None):
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images")
        if reflectances == None:
            reflectances = [panel.reflectance_from_panel_serial() for panel in self.panels]
        if len(reflectances) != len(self.panels):
            raise ValueError("Length of panel reflectances must match lengh of images")
        irradiance_list = []
        for i,p in enumerate(self.panels):
            mean_irr = p.irradiance_mean(reflectances[i])
            irradiance_list.append(mean_irr)
        return irradiance_list

    def panel_reflectance(self, panel_refl_by_band=None):
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images")
        reflectance_list = []
        for i,p in enumerate(self.panels):
            self.images[i].reflectance()
            mean_refl = p.reflectance_mean()
            reflectance_list.append(mean_refl)
        return reflectance_list

    def panel_albedo(self):
        if self.panels is not None:
            return [panel.reflectance_from_panel_serial() for panel in self.panels]
        else:
            return None

    def detect_panels(self):
        from micasense.panel import Panel
        if self.panels is not None and self.detected_panel_count == len(self.images):
            return self.detected_panel_count
        self.panels = [Panel(img,panelCorners=pc) for img,pc in zip(self.images,self.panelCorners)]
        self.detected_panel_count = 0
        for p in self.panels:
            if p.panel_detected():
                self.detected_panel_count += 1
        # is panelCorners are defined by hand
        if self.panelCorners is not None and all(corner is not None for corner in self.panelCorners):
           self.detected_panel_count = len(self.panelCorners)
        return self.detected_panel_count

    def plot_panels(self):
        if self.panels is None:
            if not self.panels_in_all_expected_images():
                raise IOError("Panels not detected in all images")
        self.__plot(
            [p.plot_image() for p in self.panels],
            plot_type='Panels',
            colorbar=False
        )

    def set_external_rig_relatives(self,external_rig_relatives):
        for i,img in enumerate(self.images):
            img.set_external_rig_relatives(external_rig_relatives[str(i)])
    
    def has_rig_relatives(self):
        for img in self.images:
            if img.meta.rig_relatives() is None:
                return False
        return True
        
    def get_warp_matrices(self, ref_index=None):
        if ref_index is None:
            ref = self.images[self.__get_reference_index()]
        else:
            ref = self.images[ref_index]
        warp_matrices  =[np.linalg.inv(im.get_homography(ref)) for im in self.images]
        return [w/w[2,2] for w in warp_matrices]

    def create_aligned_capture(self, irradiance_list=None, warp_matrices=None, normalize=False, img_type=None):
        if img_type is None and irradiance_list is None and self.dls_irradiance() is None:
            self.compute_undistorted_radiance()
            img_type = 'radiance'
        elif img_type is None:
            if irradiance_list is None:
                irradiance_list = self.dls_irradiance()+[0]
            self.compute_undistorted_reflectance(irradiance_list)
            img_type = 'reflectance'
        if warp_matrices is None:
            warp_matrices = self.get_warp_matrices()
        cropped_dimensions,_ = imageutils.find_crop_bounds(self,warp_matrices)
        self.__aligned_capture = imageutils.aligned_capture(self, 
                                                warp_matrices, 
                                                cv2.MOTION_HOMOGRAPHY, 
                                                cropped_dimensions, 
                                                None, 
                                                img_type=img_type)
        return self.__aligned_capture

    def aligned_shape(self):
        if self.__aligned_capture is None:
            raise RuntimeError("call Capture.create_aligned_capture prior to saving as stack")
        return self.__aligned_capture.shape

    def save_capture_as_stack(self, outfilename):
        from osgeo.gdal import GetDriverByName, GDT_UInt16
        if self.__aligned_capture is None:
            raise RuntimeError("call Capture.create_aligned_capture prior to saving as stack")

        rows, cols, bands = self.__aligned_capture.shape
        driver = GetDriverByName('GTiff')
        outRaster = driver.Create(outfilename, cols, rows, bands, GDT_UInt16, options = [ 'INTERLEAVE=BAND','COMPRESS=DEFLATE' ])
        if outRaster is None:
            raise IOError("could not load gdal GeoTiff driver")
        for i in range(0,5):
            outband = outRaster.GetRasterBand(i+1)
            outdata = self.__aligned_capture[:,:,i]
            outdata[outdata<0] = 0
            outdata[outdata>2] = 2   #limit reflectance data to 200% to allow some specular reflections
            outband.WriteArray(outdata*32768) # scale reflectance images so 100% = 32768
            outband.FlushCache()

        if bands == 6:
            outband = outRaster.GetRasterBand(6)
            outdata = (self.__aligned_capture[:,:,5]+273.15) * 100 # scale data from float degC to back to centi-Kelvin to fit into uint16
            outdata[outdata<0] = 0
            outdata[outdata>65535] = 65535
            outband.WriteArray(outdata)
            outband.FlushCache()
        outRaster = None

    def save_capture_as_rgb(self, outfilename, gamma=1.4, downsample=1, white_balance='norm', hist_min_percent=0.5, hist_max_percent=99.5, sharpen=True, rgb_band_indices = [2,1,0]):        
        if self.__aligned_capture is None:
            raise RuntimeError("call Capture.create_aligned_capture prior to saving as RGB")
        im_display = np.zeros((self.__aligned_capture.shape[0],self.__aligned_capture.shape[1],self.__aligned_capture.shape[2]), dtype=np.float32 )

        im_min = np.percentile(self.__aligned_capture[:,:,rgb_band_indices].flatten(), hist_min_percent)  # modify these percentiles to adjust contrast
        im_max = np.percentile(self.__aligned_capture[:,:,rgb_band_indices].flatten(), hist_max_percent)  # for many images, 0.5 and 99.5 are good values

        for i in rgb_band_indices:
            # for rgb true color, we usually want to use the same min and max scaling across the 3 bands to 
            # maintain the "white balance" of the calibrated image  
            if white_balance == 'norm':
                im_display[:,:,i] =  imageutils.normalize(self.__aligned_capture[:,:,i], im_min, im_max)
            else:
                im_display[:,:,i] =  imageutils.normalize(self.__aligned_capture[:,:,i])

        rgb = im_display[:,:,rgb_band_indices]
        rgb = cv2.resize(rgb, None, fx=1/downsample, fy=1/downsample, interpolation=cv2.INTER_AREA)

        if sharpen:
            gaussian_rgb = cv2.GaussianBlur(rgb, (9,9), 10.0)
            gaussian_rgb[gaussian_rgb<0] = 0
            gaussian_rgb[gaussian_rgb>1] = 1
            unsharp_rgb = cv2.addWeighted(rgb, 1.5, gaussian_rgb, -0.5, 0)
            unsharp_rgb[unsharp_rgb<0] = 0
            unsharp_rgb[unsharp_rgb>1] = 1
        else:
            unsharp_rgb = rgb

        # Apply a gamma correction to make the render appear closer to what our eyes would see
        if gamma != 0:
            gamma_corr_rgb = unsharp_rgb**(1.0/gamma)
            imageio.imwrite(outfilename, (255*gamma_corr_rgb).astype('uint8'))
        else:
            imageio.imwrite(outfilename, (255*unsharp_rgb).astype('uint8'))
    
    def save_thermal_over_rgb(self, outfilename, figsize=(30,23)):
        if self.__aligned_capture is None:
            raise RuntimeError("call Capture.create_aligned_capture prior to saving as RGB")
        
        # by default we don't mask the thermal, since it's native resolution is much lower than the MS
        masked_thermal = self.__aligned_capture[:,:,5]
        
        im_display = np.zeros((self.__aligned_capture.shape[0],self.__aligned_capture.shape[1],3), dtype=np.float32 )
        rgb_band_indices = [2,1,0]
        hist_min_percent = 0.2
        hist_max_percent = 99.8
        # for rgb true color, we usually want to use the same min and max scaling across the 3 bands to 
        # maintain the "white balance" of the calibrated image  
        im_min = np.percentile(self.__aligned_capture[:,:,rgb_band_indices].flatten(), hist_min_percent)  # modify these percentiles to adjust contrast
        im_max = np.percentile(self.__aligned_capture[:,:,rgb_band_indices].flatten(), hist_max_percent)  # for many images, 0.5 and 99.5 are good values
        for dst_band,src_band in enumerate(rgb_band_indices):
            im_display[:,:,dst_band] =  imageutils.normalize(self.__aligned_capture[:,:,src_band], im_min, im_max)
       
        # Compute a histogram
        min_display_therm = np.percentile(masked_thermal, 1)
        max_display_therm = np.percentile(masked_thermal, 99)

        fig, axis = plotutils.plot_overlay_withcolorbar(im_display,
                                            masked_thermal, 
                                            figsize=figsize, 
                                            title='Temperature over True Color',
                                            vmin=min_display_therm,vmax=max_display_therm,
                                            overlay_alpha=0.25,
                                            overlay_colormap='jet',
                                            overlay_steps=16,
                                            display_contours=True,
                                            contour_steps=16,
                                            contour_alpha=.4,
                                            contour_fmt="%.0fC",
                                            show=False)
        fig.savefig(outfilename)
