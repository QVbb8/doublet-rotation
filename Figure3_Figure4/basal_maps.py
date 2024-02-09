# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:56:19 2022

@author: Riveline LAB
"""

import os

import numpy as np

import pandas as pd
import open3d as o3d

from skimage.io import imread, imsave
from tifffile import imwrite
from scipy import ndimage
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull, Delaunay

from skimage.morphology import skeletonize_3d
from skimage import measure
from pyntcloud import PyntCloud
from pyntcloud.geometry.models.plane import Plane

from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL

from mpl_toolkits.axes_grid1 import make_axes_locatable

import read_ply

import useful_functions_ply_files as uf

import useful_functions_interface as uf_int

import function_final_interface as ff_int


def get_names_ply(path2):
    
    A = np.array(os.listdir(path=path2))
    B = np.array([file.endswith(".ply") for file in A])
    A = A[B]
    return(A)

def get_com_cell_time(time, PATHS):
    com = []
    for p in PATHS :
        A = get_names_ply(p)
        
        for file in A:
            name, ext = os.path.splitext(file)
            if name.split('_')[0].split('time')[1] ==str(time):
                mesh,V,T = uf.get_vertices_triangles(file)
                com.append(uf.calc_centre_of_mass(V,T))
    return(com)

def get_cells_cloud_time(time,PATHS):
    V_cell = []
    for p in PATHS :
        A = get_names_ply(p)
        for file in A:
            name, ext = os.path.splitext(file)
            if name.split('_')[0].split('time')[1] ==str(time):
                mesh,V,T = uf.get_vertices_triangles(p+'/'+file)
                V_cell.append(V)
    cell1 = V_cell[0]
    cell2= V_cell[1]
    return(cell1,cell2)

def get_mesh_vert_tri_doublet(time,path):
    
    names_doublet = get_names_ply(path)
    for name_doublet in names_doublet:
        name, ext = os.path.splitext(name_doublet)
        if name.split('_')[0].split('time')[1] ==str(time):
            mesh_d, V_d, T_d = uf.get_vertices_triangles(name_doublet)
    return(mesh_d,V_d,T_d)

def get_new_coord_syst(com1,com2,V_d,com_d):
    
    # #### new coordinate system ####
    n = (com2-com1)/np.linalg.norm(com2-com1)
    e1 = np.array([0,0,1])
    e1 = e1 - np.dot(e1, n) * n
    e1/= np.sqrt((e1**2).sum())
    e2 = np.cross(n, e1)
    
    basal_points = V_d - com_d
    
    x_n = np.dot(basal_points,e1)
    y_n = np.dot(basal_points,e2)
    z_n = np.dot(basal_points,n)
    
    return(x_n,y_n,z_n)

def get_phi(x_n,y_n):
    # ################### compute cylindrical coordinates (rho, phi, h) ############
    phi = np.arctan2(y_n,x_n)
    return(phi)


def get_mean_cell_intensity(com1,com2, img, n_pix, scale_factor):
    
    c1 = np.copy(com1)
    c2 = np.copy(com2)
    
    c1[2]/= scale_factor 
    c2[2] /= scale_factor
    c1,c2 = np.round(c1).astype(np.int), np.round(c2).astype(np.int)
    v=[]
    for i in range(-n_pix,n_pix+1):
            for j in range(-n_pix,n_pix+1):
                x01 = c1[0] + i
                y01 = c1[1] + j
                z01 = c1[2]
                x02 = c2[0] + i
                y02 = c2[1] + j
                z02 = c2[2]
                v.append(img[z01][x01][y01])
                v.append(img[z02][x02][y02])
                
    I_cell = np.mean(v)
    
    return(I_cell)

def get_intensity(vertices, I_cell, img, n_pix, scale_factor):
    
    
    pos = uf.convert_pos_to_pix(vertices,scale_factor)
    values = []
    for k in range(len(pos)):
        v = []
        for i in range(-n_pix,n_pix+1):
            for j in range(-n_pix,n_pix+1):
                x0 = pos[k,1] + i
                y0 = pos[k,2] + j
                z0 = pos[k,0]
                v.append(img[z0][x0][y0])
        value_signal = np.mean(v)
        values.append(value_signal)    
        
    # values = np.array([img[pos[i,0]][pos[i,1]][pos[i,2]] for i in range(len(vertices))])
    
    return((np.array(values)-I_cell)/I_cell)

def plot(phi,z_n, I, name, colormap, min_value ,max_value): 
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='4%', pad=0)
    
    plot = ax.scatter(phi,z_n, c = I, s= 20, vmin = min_value, vmax = max_value, cmap = colormap)
    cbar = fig.colorbar(plot, cax=cax,orientation = 'horizontal')
    cbar.set_ticks([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    fig.savefig(name+'.jpg')

    # plt.title(name, fontsize = 14)
    return()

def basal_points(cell1,cell2,doublet):
    
    dist1, dist2 = ff_int.compute_distance_between_two_clouds(cell1,cell2)
    bas_cell1, bas_cell2 = cell1[dist1/np.max(dist1)>0.1], cell2[dist2/np.max(dist2)>0.1]
    complete = np.vstack((bas_cell1,bas_cell2,doublet))
    return(complete)