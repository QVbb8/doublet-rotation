#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 09:16:45 2022

@author: vagne
"""

#------------------------------------------------------------------------------
#This script must be executed first and will generate either the file 
#'all_pz_single.txt' or 'all_pz_single_blur.txt' that contain the length scales
# to be used to correct the intensity decay in z of the myosin signal of single
#cells.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#decide here which file to generate 
#(False : all_pz_single.txt, True: all_pz_single_blur.txt)
#------------------------------------------------------------------------------
blur_case=True



#------------------------------------------------------------------------------
#import packages and define functions
#------------------------------------------------------------------------------
from skimage.io import imread
import matplotlib.font_manager as font_manager
import numpy as np
import os
os.chdir('../Figure3_Figure4')
import useful_functions_interface as uf_int
import useful_functions_ply_files as uf
import function_final_interface as ff_int
os.chdir('../Supp_Fig1_single_cells')
from scipy.spatial import Delaunay
from numba import jit
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def get_normalangle_signal(vertices,triangles,sig):
    
    u1 = vertices[triangles[:, 0], :]
    u2 = vertices[triangles[:, 1], :]
    u3 = vertices[triangles[:, 2], :]
        
    # calculate normal
    v1 = u2 - u1
    v2 = u3 - u1
    Normal = np.cross(v1, v2)
    Normal = Normal/np.linalg.norm(Normal,axis=1)[:,None]
    
    angle1=np.arccos(Normal[:,2])
    angle2=np.pi-np.arccos(Normal[:,2])
    angle=np.min(np.stack((angle1,angle2),axis=1),axis=1)
    
    altitude=(u1[:,2]+u2[:,2]+u3[:,2])/3.0
    
    
    avg_signal=(sig[triangles[:,0]]+sig[triangles[:,1]]+sig[triangles[:,2]])/3.0
    
    #plt.figure()
    #plt.scatter(angle,avg_signal,c=altitude,s=1)
    #plt.axhline(np.mean(avg_signal),color='r')
    #plt.show()
    
    return angle,avg_signal

def get_pixels_inside(V1,V2,im_mrlc, scale_factor):
    
    vertices=np.concatenate((V1,V2),axis=0)
    
    pos = uf.convert_pos_to_pix(vertices,scale_factor)
    
    tri = Delaunay(pos)
    xyz = create_xyz(pos)
    
    inside_cell = tri.find_simplex(xyz)>=0
    xyz_cell = xyz[inside_cell]
    xyz_scale_cell = np.copy(xyz_cell)
    xyz_scale_cell[:,0]*=scale_factor
    pos_scale = np.copy(pos).astype(np.float32)
    pos_scale[:,0] *= scale_factor
    
    #tree = cKDTree(pos_scale)
    #distances,indices = tree.query(xyz_scale_cell, k = 1)
    #xyz_inside = xyz_cell[distances>20]
    signal_inside = get_signal_at_xyz(xyz_cell, im_mrlc)
    
    result=np.zeros((signal_inside.shape[0],4))
    result[:,0:3]=xyz_scale_cell
    result[:,3]=signal_inside
    
    return result

@jit
def get_signal_at_xyz(xyz_inside, im_mrlc):
    
    npoints=xyz_inside.shape[0]
    img_inside_cell=np.zeros(npoints)
    
    for i in range(npoints):
        z = int(xyz_inside[i][0])
        x = int(xyz_inside[i][1])
        y = int(xyz_inside[i][2])
        img_inside_cell[i]=im_mrlc[z][x][y]

    return img_inside_cell


def get_signal_inside_cell(vertices,im_mrlc, scale_factor):
    
    pos = uf.convert_pos_to_pix(vertices,scale_factor)
    
    tri = Delaunay(pos)
    xyz = create_xyz(pos)
    
    inside_cell = tri.find_simplex(xyz)>=0
    xyz_cell = xyz[inside_cell]
    xyz_scale_cell = np.copy(xyz_cell)
    xyz_scale_cell[:,0]*=scale_factor
    pos_scale = np.copy(pos).astype(np.float32)
    pos_scale[:,0] *= scale_factor
    
    tree = cKDTree(pos_scale)
    distances,indices = tree.query(xyz_scale_cell, k = 1)
    xyz_inside = xyz_cell[distances>20]
    avg_inside_signal = avg_signal(xyz_inside, im_mrlc)
    
    return(avg_inside_signal)

@jit
def create_xyz(pos):
    
    xmin = np.min(pos[:,1])
    xmax = np.max(pos[:,1])
    ymin = np.min(pos[:,2])
    ymax = np.max(pos[:,2])
    zmin = np.min(pos[:,0])
    zmax = np.max(pos[:,0])
    
    X = np.arange(xmin, xmax, 10)
    Y = np.arange(ymin, ymax, 10)
    Z = np.arange(zmin, zmax, 1)
    
    i=0
    xyz = np.zeros((len(X)*len(Y)*len(Z),3))
    for x in X:
        for y in Y:
            for z in Z:
                xyz[i,:] = np.array([int(z),int(x),int(y)])
                i+=1

    return(xyz)

@jit
def avg_signal(xyz_inside, im_mrlc):
    
    img_inside_cell = 0.0
    
    for triplet in xyz_inside:
        z = int(triplet[0])
        x = int(triplet[1])
        y = int(triplet[2])
        img_inside_cell+=im_mrlc[z][x][y]
    avg_inside_signal = img_inside_cell/len(xyz_inside)
    return(avg_inside_signal)

@jit
def get_signal_vertices_single(vertices, img, n_pix_xy, n_pix_z, scale_factor, avg_signal_cell):
    
    # reorder pixel positions to be [z, rows, cols]
    # pix are the  real coordinates of the vertices XYZ in units of ImageJ 
    # but WARNING img = imread(img_name) you give youi img as [z][y][x] 
    pix = np.zeros_like(vertices)
    pix[:, 0] = vertices[:, 0]
    pix[:, 1] = vertices[:, 1]
    pix[:, 2] = vertices[:, 2]/scale_factor
        
    Npoints = (2*n_pix_xy+1)*(2*n_pix_xy+1)*(2*n_pix_z+1)
    values = np.zeros(len(pix))
    for k in range(len(pix)):
        v = np.zeros(Npoints)
        n=0
        for i in range(-n_pix_xy,n_pix_xy+1):
            for j in range(-n_pix_xy,n_pix_xy+1):
                for l in range(-n_pix_z, n_pix_z+1):
                    x0 = int(np.round(pix[k,0])) + i
                    y0 = int(np.round(pix[k,1])) + j
                    z0 = int(np.round(pix[k,2])) + l
                    if ((z0 < img.shape[0]) and (z0>=0)):
                        v[n] = img[z0][y0][x0]
                    n+=1
        if n < Npoints-1:
            print(Npoints-1-n)
        v = v[v>0]
        npoints = len(v)
        sorted_value = np.sort(v)[::-1]
        
        value_signal = np.mean(sorted_value[:int(npoints*(20/100))])
        
        # value_signal = v/n
        diff = value_signal - avg_signal_cell
        values[k] = diff

           
    return(values)
    

cm = 1/2.54

fname = 'Arial'
font = font_manager.FontProperties(family=fname,
                                    weight='normal',
                                    style='normal', size=5)


all_paths = './data_single_cells/Segmentation_'

scale_factors = 1/0.103


#------------------------------------------------------------------------------
#FIRST PART: COMPUTE Zmin and Zmax, also xmin,xmax,ymin,ymax
#------------------------------------------------------------------------------
all_zM=[]
all_zm= []

all_xM=[]
all_xm= []

all_yM=[]
all_ym= []

#loop on doublets (individual calibration for each doublet)
for k in range(1,13):
    print(f'doublet {k}')
    
    if blur_case:
        img_name = str(k)+'_blur.tif'
    else:
        img_name = str(k)+'.tif'
   
    scale_factor = scale_factors
    
    dist_threshold = int(np.floor(scale_factor) + 1) 
    xy_pix = 5
    # xy_pix = int(np.floor(scale_factor) + 1)
    z_pix = 1
    
    path=all_paths+str(k)+'/'
    img = imread(path+img_name)

    PATHS = [path + 'Cell_1']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    
    #now loop on time points of the doublet to accumulate data
    i = 0
    #we don't skip the last time point
    for t in range(startpoint, endpoint+1): 
        mesh1, V1, T1 = uf.get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')
        
        im_mrlc = img[t-1,:,:,:]
        
        #first we only accumulate the zmin,zmax of the doublet for all times
        if t==startpoint:
            zm=np.min(V1[:,2])
            zM=np.max(V1[:,2])
        else:
            if np.min(V1[:,2]) < zm:
                zm=np.min(V1[:,2])
            if np.max(V1[:,2]) > zM:
                zM=np.max(V1[:,2])
                
        if t==startpoint:
            xm=np.min(V1[:,0])
            xM=np.max(V1[:,0])
        else:
            if np.min(V1[:,0]) < xm:
                xm=np.min(V1[:,0])
            if np.max(V1[:,0]) > xM:
                xM=np.max(V1[:,0])
                
        if t==startpoint:
            ym=np.min(V1[:,1])
            yM=np.max(V1[:,1])
        else:
            if np.min(V1[:,1]) < ym:
                ym=np.min(V1[:,1])
            if np.max(V1[:,1]) > yM:
                yM=np.max(V1[:,1])
        i+=1
    all_zm.append(zm)
    all_zM.append(zM)
    all_xm.append(xm)
    all_xM.append(xM)
    all_ym.append(ym)
    all_yM.append(yM)
    
#----------------------------------------------------------------------------    
#SECOND PART: getting the length scales (requires all_zm,all_zM,all_xm,all_xM,all_ym,all_yM
#----------------------------------------------------------------------------

all_px=[]
all_py=[]
all_pz=[]

all_px_um=[]
all_py_um=[]
all_pz_um=[]

#loop on doublets (individual calibration for each doublet)
for k in range(1,13):
    
    print(f'doublet {k}')
    
    if blur_case:
        img_name = str(k)+'_blur.tif'
    else:
        img_name = str(k)+'.tif'

    scale_factor = scale_factors
    
    dist_threshold = int(np.floor(scale_factor) + 1) 
    xy_pix = 5
    # xy_pix = int(np.floor(scale_factor) + 1)
    z_pix = 1
    
    path=all_paths+str(k)+'/'
    img = imread(path+img_name)

    PATHS = [path + 'Cell_1']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    
    nbin=20
    signal_binned_z=np.zeros((endpoint-startpoint+1,nbin))
    signal_timeavg_z=np.zeros(nbin)
    zbins=np.linspace(all_zm[k-1],all_zM[k-1],nbin+1)
    zmid=(zbins[1:]+zbins[:-1])/2
    
    signal_binned_x=np.zeros((endpoint-startpoint+1,nbin))
    signal_timeavg_x=np.zeros(nbin)
    xbins=np.linspace(all_xm[k-1],all_xM[k-1],nbin+1)
    xmid=(xbins[1:]+xbins[:-1])/2
    
    signal_binned_y=np.zeros((endpoint-startpoint+1,nbin))
    signal_timeavg_y=np.zeros(nbin)
    ybins=np.linspace(all_ym[k-1],all_yM[k-1],nbin+1)
    ymid=(ybins[1:]+ybins[:-1])/2
    
    #now loop on time points of the doublet to accumulate data
    i = 0
    #we don't skip the last time point
    for t in range(startpoint, endpoint+1): 

        mesh1, V1, T1 = uf.get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')
 
        im_mrlc = img[t-1,:,:,:]
        
        
        #avg_inside_cell1 = get_signal_inside_cell(V1, im_mrlc, scale_factor)
        #avg_inside_cell2 = get_signal_inside_cell(V2,im_mrlc, scale_factor)
        
        #here we DONT substract the average inside the cell because the raw signal is the one to work with.
        mrlc_values_cell1 = get_signal_vertices_single(V1, im_mrlc, xy_pix, z_pix, scale_factor, 0)
        
        #get data on angles of angle of normal and intensity of signal
        angle,avgsig=get_normalangle_signal(V1,T1,mrlc_values_cell1)
        
        
        #F.show_doublet_outer(V1,mrlc_values_cell1)
        
        #put signal into bins
        for z in range(nbin):
            indices=np.logical_and(zbins[z] <= V1[:,2],V1[:,2] <= zbins[z+1])
            signal_binned_z[i][z] = np.mean(mrlc_values_cell1[indices])
            
        for x in range(nbin):
            indices=np.logical_and(xbins[x] <= V1[:,0],V1[:,0] <= xbins[x+1])
            signal_binned_x[i][x] = np.mean(mrlc_values_cell1[indices])
            
        for y in range(nbin):
            indices=np.logical_and(ybins[y] <= V1[:,1],V1[:,1] <= ybins[y+1])
            signal_binned_y[i][y] = np.mean(mrlc_values_cell1[indices])
        
        
        i+=1
    
    
    signal_timeavg_z=np.nanmean(signal_binned_z,axis=0)
    signal_timeavg_x=np.nanmean(signal_binned_x,axis=0)
    signal_timeavg_y=np.nanmean(signal_binned_y,axis=0)
    
    
    indices=~np.isnan(signal_timeavg_x)
    #px=np.polyfit(xmid[indices],signal_timeavg_x[indices],1)
    px=np.polyfit(xmid[indices],np.log(signal_timeavg_x[indices]),1)
    
    indices=~np.isnan(signal_timeavg_y)
    #py=np.polyfit(ymid[indices],signal_timeavg_y[indices],1)
    py=np.polyfit(ymid[indices],np.log(signal_timeavg_y[indices]),1)
    
    indices=~np.isnan(signal_timeavg_z)
    #pz=np.polyfit(zmid[indices],signal_timeavg_z[indices],1)
    pz=np.polyfit(zmid[indices],np.log(signal_timeavg_z[indices]),1)
    
    all_px.append(px[0])
    all_py.append(py[0])
    all_pz.append(pz[0])
    
    all_px_um.append(px[0]*scale_factor)
    all_py_um.append(py[0]*scale_factor)
    all_pz_um.append(pz[0]*scale_factor)
    
    max_yaxis=np.max([np.nanmax(signal_binned_z),np.nanmax(signal_binned_x),np.nanmax(signal_binned_y)])
    min_yaxis=np.min([np.nanmin(signal_binned_z),np.nanmin(signal_binned_x),np.nanmin(signal_binned_y)])
    
    

#------------------------------------------------------------------------------
# Save the file
#------------------------------------------------------------------------------

if blur_case:
    np.savetxt('all_pz_single_blur.txt',all_pz)
else:
    np.savetxt('all_pz_single.txt',all_pz)

    