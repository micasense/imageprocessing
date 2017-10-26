#!/usr/bin/env python
# coding: utf-8
"""
MicaSense Plotting Utilities

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

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plotwithcolorbar(img, title=''):
    ''' Plot an image with a colorbar '''
    fig, axis = plt.subplots(1, 1, figsize=(8, 6))
    rad2 = axis.imshow(img)
    axis.set_title(title)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(rad2, cax=cax)
    plt.tight_layout()
    plt.show()

def subplotwithcolorbar(rows, cols, images, titles=None, fig_size=None):
    ''' Plot a set of images in subplots '''
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    for i in range(cols*rows):
        column = int(i%cols)
        row = int(i/cols)
        if i < len(images):
            rad = axes[row][column].imshow(images[i])
            if titles is not None:
                axes[row][column].set_title(titles[i])
            divider = make_axes_locatable(axes[row][column])
            cax = divider.append_axes("right", size="3%", pad=0.05)
            fig.colorbar(rad, cax=cax)
        else:
            axes[row, column].axis('off')
    plt.tight_layout()
    plt.show()
    return fig, axes

def colormap(cmap):
    ''' Set the defalut plotting colormap
    Could be one of 'gray, viridis, plasma, inferno, magma, nipy_spectral'
    '''
    plt.set_cmap(cmap)
