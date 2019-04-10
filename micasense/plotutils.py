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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pylab import cm

def plotwithcolorbar(img, title=None, figsize=None, vmin=None, vmax=None):
    ''' Plot an image with a colorbar '''
    fig, axis = plt.subplots(1, 1, figsize=figsize)
    rad2 = axis.imshow(img, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(rad2, cax=cax)
    plt.tight_layout()
    plt.show()
    return fig, axis

def subplotwithcolorbar(rows, cols, images, titles=None, figsize=None):
    ''' Plot a set of images in subplots '''
    fig, axes = plt.subplots(rows, cols, figsize=figsize,squeeze=False)
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

def plot_overlay_withcolorbar(imgbase, imgcolor, title=None, figsize=None, vmin=None, vmax=None, overlay_alpha=1.0, overlay_colormap='viridis', overlay_steps=None, display_contours=False, contour_fmt=None, contour_steps=None, contour_alpha=None):
    ''' Plot an image with a colorbar '''
    fig, axis = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    base = axis[0][0].imshow(imgbase)
    if overlay_steps is not None:
        overlay_colormap = cm.get_cmap(overlay_colormap,overlay_steps)
    rad2 = axis[0][0].imshow(imgcolor, vmin=vmin, vmax=vmax, alpha=overlay_alpha, cmap=overlay_colormap)
    if display_contours:
        if contour_steps is None:
            contour_steps = overlay_steps
        if contour_alpha is None:
            contour_alpha = overlay_alpha
        contour_cmap = cm.get_cmap(overlay_colormap,contour_steps)
        contour_list = np.arange(vmin, vmax, (vmax-vmin)/contour_steps)
        rad3 = axis[0][0].contour(imgcolor, contour_list, cmap=contour_cmap, alpha=contour_alpha)
        fontsize=8+(max(figsize)/10)*2
        axis[0][0].clabel(rad3,rad3.levels,inline=True,fontsize=fontsize,fmt=contour_fmt)
    axis[0][0].set_title(title)
    divider = make_axes_locatable(axis[0][0])
    cax = divider.append_axes("right", size="3%", pad=0.05)
    fig.colorbar(rad2, cax=cax)
    plt.tight_layout()
    plt.show()
    return fig, axis[0][0]

def subplot(rows, cols, images, titles=None, figsize=None):
    ''' Plot a set of images in subplots '''
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for i in range(cols*rows):
        column = int(i%cols)
        row = int(i/cols)
        if i < len(images):
            rad = axes[row][column].imshow(images[i])
            if titles is not None:
                axes[row][column].set_title(titles[i])
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

import numpy as np
def plot_ned_vector3d(x,y,z, u=0,v=0,w=0, title=None, figsize=(8,5)):
    '''Create a 3d plot of a North-East-Down vector. XYZ is the (tip of the) vector,
       uvw is the base location of the vector '''
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.quiver(u, v, w, x, y, z, color='r')
    ax.quiver(u, v, w, x, y, 0, color='b')
    ax.quiver(x, y, 0, 0, 0, z, color='g')
    
    ax.legend()
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([0,1])
    ax.set_xlabel("West - East")
    ax.set_ylabel("South - North")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()
    return fig,ax