#!/usr/bin/env python
# coding: utf-8
"""
Test imageset class

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


import pytest
import os, glob

import micasense.imageset as imageset
import micasense.capture as capture
import micasense.image as image

@pytest.fixture()
def files_dir():
    return os.path.join('data', '0000SET', '000')

progress_val = 0.0
def progress(p):
    global progress_val
    progress_val = p
    
def test_from_captures():
    file1 = os.path.join(files_dir(), 'IMG_0000_1.tif')
    file2 = os.path.join(files_dir(), 'IMG_0001_1.tif')
    cap1 = capture.Capture.from_file(file1)
    cap2 = capture.Capture.from_file(file2)
    imgset = imageset.ImageSet([cap1,cap2])
    assert imgset.captures is not None

def test_from_directory():
    global progress_val
    progress(0.0)
    imgset = imageset.ImageSet.from_directory(files_dir(), progress)
    assert imgset is not None
    assert progress_val == 1.0
    assert len(imgset.captures) == 2

def test_as_nested_lists():
    imgset = imageset.ImageSet.from_directory(files_dir())
    assert imgset is not None
    data, columns = imgset.as_nested_lists()
    assert data[0][1] == 36.576096
    assert columns[0] == 'timestamp'