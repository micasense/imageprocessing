import cv2
import datetime
import glob
import os
import math
from mapboxgl.utils import df_to_geojson
import numpy as np
import pandas as pd
import random
import re
import subprocess
import sys

import micasense.capture
import micasense.imageutils
import micasense.imageset as imageset


class BatchProcess():
    def __init__(self, imagepath, **kwargs):
        self.startTime = datetime.datetime.now()
        self.process_complete = False # Init bool flag for completion to false
        # Check validity of imagepath passed
        self.imagePath = imagepath
        if not os.path.isdir(self.imagePath):
            print(f'{self.imagePath} is not recognised as a valid system path, please check and retry')
            return None
        # Check and validate referenceimage path passed, else default to random image from imagepath as reference
        if kwargs['referenceimage']: # If reference image kwarg present
            if not os.path.isfile(kwargs['referenceimage']): # If referenceimage kwarg file does not exist, console printout and return None
                print(f'referenceimage exception\n{kwargs["referenceimage"]} not recognised as valid path/file in system, please check and retry')
                return None
            else: # If referenceimage kwarg file does exist, store IMG_XXXX_ section of filename for use in set_warp_matrices
                self.referenceImage = self.get_named_reference_img(kwargs['referenceimage'])
            if not self.referenceImage: # If referenceimage kwarg file is still not found, console printout and return None
                print(f'referenceimage exception\n{kwargs["referenceimage"]} not recognised as valid path/file in system, please check and retry')
                return None
        else: # Get random reference image from imagepath if referenceimage kwarg not present
            self.referenceImage = self.get_rand_reference_img()
        # Validate outputpath if present
        if kwargs['outputpath'] and not os.path.isdir(kwargs['outputpath']):
                print(f'outputpath exception\n{kwargs["outputpath"]} not recognised as valid path/file in system, please check and retry')
                return None
        # Assign outputpath to member variable if valid path passed, default to imagepath 'stacks' dir if not
        self.outputPath = os.path.join(kwargs['outputpath'], 'stacks') if kwargs['outputpath'] else os.path.join(self.imagePath, 'stacks')

        self.thumbnails = kwargs['thumbnails']

        self.thumbnailPath = os.path.join(kwargs['outputpath'], 'thumbnails') if kwargs['outputpath'] else os.path.join(self.imagePath, 'thumbnails')

        # Validate panelpath passed if present
        self.panelPath = kwargs['panelpath']
        if self.panelPath:
            if not os.path.isfile(self.panelPath):
                print(f'panelpath exception\n{self.panelPath} not recognised as valid path/file in system, please check and retry')
                return None
            else:
                panelroot = re.search('.*?\d{4}_\d+\.tif', self.panelPath) # Validate proper formatting of panel image i.e IMG_XXXX_*.tif
                if not panelroot:
                    print(f'panelpath exception\n{self.panelPath} does not point to a capture file using the proper "IMG_XXXX_*.tif" naming convention')
                    return None

        self.panelNames = None
        self.panelCap = None
        self.set_panels()

        # Set warp_matrices
        self.imageNames = glob.glob(os.path.join(self.imagePath, '**', f'{self.referenceImage}*.tif'), recursive=True)
        self.capture = micasense.capture.Capture.from_filelist(self.imageNames)
        match_index = 1 # Index of the band
        max_alignment_iterations = 10
        warp_mode = cv2.MOTION_HOMOGRAPHY # MOTION_HOMOGRAPHY or MOTION_AFFINE. For Altum images only use HOMOGRAPHY
        self.pyramidLevels = kwargs['pyramidlevels'] # for images with RigRelatives, setting this to 0 or 1 may improve alignment, default kwarg val == 0
        print(self)
        '''IMPORTANT - THIS CANNOT YET BE ABSTRACTED AWAY TO A FUNCTION DUE TO MULTIPROCESSING IMPLEMENTATION IN micasense.imageutils?'''
        print("Aligning images. Depending on settings this can take from a few seconds to many minutes")
        warp_matrices, alignment_pairs = micasense.imageutils.align_capture(self.capture,
                                                                  ref_index = match_index,
                                                                  max_iterations = max_alignment_iterations,
                                                                  warp_mode = warp_mode,
                                                                  pyramid_levels = self.pyramidLevels)
        print('Finished aligning warp matrices')
        #print(warp_matrices)
        self.warpMatrices = warp_matrices

        self.write_stacks()
        self.process_complete = True # Flag to mark completion of batchprocessing with no exceptions

    def set_panels(self):
        if not self.panelPath: # If panelpath kwarg not defined then search imagepath directory/sub-directories for IMG_0000_*.tif images for panels
            self.panelNames = glob.glob(os.path.join(self.imagePath, '**', 'IMG_0000_*.tif'), recursive=True)
        else: # Else in case of panelpath kwarg being passed
            panelroot = re.search('.*?\d{4}_(?=_\d+\.tif)', os.path.basename(self.panelPath)) # Validate proper formatting of panel image i.e IMG_XXXX_*.tif
            self.panelNames = glob.glob(os.path.join(os.path.dirname(self.panelPath), f'{panelroot}*.tif'))
        if len(self.panelNames) > 0: # If panels found, load panel images to micasense Capture object list
            self.panelCap = micasense.capture.Capture.from_filelist(self.panelNames)
        else: # Otherwise set panelcap to None
            self.panelCap = None

    def get_named_reference_img(self, referenceimage):
        ref = re.search('IMG_\d{4}_', os.path.basename(referenceimage))
        if ref:
            return ref.group(0)
        else:
            return None

    def get_rand_reference_img(self):
        images = glob.glob(os.path.join(self.imagePath, '**', 'IMG_*.tif'), recursive=True)
        references = []
        for i in images:
            image = re.search('IMG_\d{4}(?<!0000)_', i)
            if image:
                references.append(image.group(0))
        return references[random.randint(0, len(references)-1)]

    def write_stacks(self):
        if not self.warpMatrices:
            print('warp_matrices not present, please run BatchProcess.set_warp_matrices before attempting to run BatchProcess.write_stacks')
            return None
        useDLS = True
        overwrite = False # can be set to set to False to continue interrupted processing

        if self.panelCap is not None:
            if self.panelCap.panel_albedo() is not None and not any(v is None for v in self.panelCap.panel_albedo()):
                panel_reflectance_by_band = self.panelCap.panel_albedo()
            else:
                panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67] #RedEdge band_index order
            panel_irradiance = self.panelCap.panel_irradiance(panel_reflectance_by_band)
            img_type = "reflectance"
        else:
            if useDLS:
                img_type= 'reflectance'
            else:
                img_type = 'radiance'

        imgset = imageset.ImageSet.from_directory(self.imagePath)

        data, columns = imgset.as_nested_lists()
        df = pd.DataFrame.from_records(data, index='timestamp', columns=columns)
        geojson_data = df_to_geojson(df,columns[3:],lat='latitude',lon='longitude')

        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)
        if self.thumbnails and not os.path.exists(self.thumbnailPath):
            os.makedirs(self.thumbnailPath)
        # Save out geojson data so we can open the image capture locations in our GIS
        with open(os.path.join(self.outputPath,'imageSet.json'),'w') as f:
            f.write(str(geojson_data))
        try:
            irradiance = panel_irradiance+[0]
        except NameError:
            irradiance = None

        start = datetime.datetime.now()
        for i,capture in enumerate(imgset.captures):
            outputFilename = capture.uuid+'.tif'
            thumbnailFilename = capture.uuid+'.jpg'
            fullOutputPath = os.path.join(self.outputPath, outputFilename)
            fullThumbnailPath= os.path.join(self.thumbnailPath, thumbnailFilename)
            if (not os.path.exists(fullOutputPath)) or overwrite:
                if(len(capture.images) == len(imgset.captures[0].images)):
                    capture.create_aligned_capture(irradiance_list=irradiance, warp_matrices=self.warpMatrices)
                    capture.save_capture_as_stack(fullOutputPath)
                    if self.thumbnails:
                        # TODO
                        '''
                        IMPORTANT - CIR WRITEOUT IMPLEMENTATION COMMENTED OUT BELOW, AWAITING RESPONSE FROM MICASENSE
                        '''
                        #capture.save_thermal_over_rgb(fullThumbnailPath)
                        capture.save_capture_as_rgb(fullThumbnailPath)
            capture.clear_image_data()
        end = datetime.datetime.now()

        print("Saving time: {}".format(end-start))
        print("Alignment+Saving rate: {:.2f} images per second".format(float(len(imgset.captures))/float((end-start).total_seconds())))

        '''WRITE METADATA TO ALIGNED STACKS'''
        header = "SourceFile,\
        GPSDateStamp,GPSTimeStamp,\
        GPSLatitude,GpsLatitudeRef,\
        GPSLongitude,GPSLongitudeRef,\
        GPSAltitude,GPSAltitudeRef,\
        FocalLength,\
        XResolution,YResolution,ResolutionUnits\n"

        lines = [header]
        for capture in imgset.captures:
            #get lat,lon,alt,time
            outputFilename = capture.uuid+'.tif'
            fullOutputPath = os.path.join(self.outputPath, outputFilename)
            lat,lon,alt = capture.location()
            #write to csv in format:
            # IMG_0199_1.tif,"33 deg 32' 9.73"" N","111 deg 51' 1.41"" W",526 m Above Sea Level
            latdeg, latmin, latsec = self.decdeg2dms(lat)
            londeg, lonmin, lonsec = self.decdeg2dms(lon)
            latdir = 'North'
            if latdeg < 0:
                latdeg = -latdeg
                latdir = 'South'
            londir = 'East'
            if londeg < 0:
                londeg = -londeg
                londir = 'West'
            resolution = capture.images[0].focal_plane_resolution_px_per_mm

            linestr = '"{}",'.format(fullOutputPath)
            linestr += capture.utc_time().strftime("%Y:%m:%d,%H:%M:%S,")
            linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},'.format(int(latdeg),int(latmin),latsec,latdir[0],latdir)
            linestr += '"{:d} deg {:d}\' {:.2f}"" {}",{},{:.1f} m Above Sea Level,Above Sea Level,'.format(int(londeg),int(lonmin),lonsec,londir[0],londir,alt)
            linestr += '{}'.format(capture.images[0].focal_length)
            linestr += '{},{},mm'.format(resolution,resolution)
            linestr += '\n' # when writing in text mode, the write command will convert to os.linesep
            lines.append(linestr)

        fullCsvPath = os.path.join(self.outputPath,'log.csv')
        with open(fullCsvPath, 'w') as csvfile: #create CSV
            csvfile.writelines(lines)

        if os.environ.get('exiftoolpath') is not None:
            exiftool_cmd = os.path.normpath(os.environ.get('exiftoolpath'))
        else:
            exiftool_cmd = 'exiftool'

        cmd = '{} -csv="{}" -overwrite_original {}'.format(exiftool_cmd, fullCsvPath, self.outputPath)
        subprocess.check_call(cmd, shell=True)


    def decdeg2dms(self, dd):
       is_positive = dd >= 0
       dd = abs(dd)
       minutes,seconds = divmod(dd*3600,60)
       degrees,minutes = divmod(minutes,60)
       degrees = degrees if is_positive else -degrees
       return (degrees,minutes,seconds)

    def __repr__(self):
        output = ''
        tn = f'thumbnails: {self.thumbnails}'
        if self.thumbnails:
            tn += f'\nthumbnail path: {self.thumbnailPath}'
        if self.panelNames:
            panels = '\t'+'\n\t'.join(self.panelNames)
        else:
            panels = 'None\n'
        reference = '\t'+'\n\t'.join(self.imageNames)
        return f'''
BatchProcess Configuration
start time: {self.startTime}
image path: {self.imagePath}
output path: {self.outputPath}
{tn}
pyramidlevels: {self.pyramidLevels}
panel group:
{panels}
reference group:
{reference}
'''
