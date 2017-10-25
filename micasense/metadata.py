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

import exiftool
class Metadata(object):
    ''' Container for Micasense image metadata'''
    def __init__(self, filename, exiftoolPath=None):
        self.xmpfile = None
        with exiftool.ExifTool(exiftoolPath) as exift:
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
                val = val[index]
        except KeyError:
            print ("Item "+item+" not found")
        except IndexError:
            print("Item {0} is length {1}, inxex {2} is outside this range.".format(
                item,
                len(self.exif[item]),
                index)
            )

        return val

    def size(self, item):
        '''get the size (length) of a metadata item'''
        val = self.get_item(item)
        return len(val)
