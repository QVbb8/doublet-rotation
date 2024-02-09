#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:07:10 2023

@author: vagne
"""

#------------------------------------------------------------------------------
#This script generates the plots of Figure 5.b and Figure 5.c about 
#blebbistatin treatment. The code will analyse the experimental data contained
#in the './blebbistatin_data' folder
#------------------------------------------------------------------------------
        
        
#packages to import
import os
import numpy as np
import pandas as pd
import open3d as o3d
from skimage.io import imread, imsave
from tifffile import imwrite
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy import stats
from skimage.morphology import skeletonize_3d
from skimage import measure
from pyntcloud import PyntCloud
from pyntcloud.geometry.models.plane import Plane
from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from pyevtk.hl import pointsToVTK
from pyevtk.hl import unstructuredGridToVTK
from pyevtk.vtk import VtkTriangle, VtkQuad
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.font_manager as font_manager
from numba import jit

#custom functions (same as for the plots of Figure 3 and 4)
os.chdir('../Figure3_Figure4')
import functions_doublet as F
import read_ply
import useful_functions_ply_files as uf
import useful_functions_interface as uf_int
import function_final_interface as ff_int
import basal_maps as bm
os.chdir('../Figure5_blebbistatin')

#additional functions
def get_names_ply(path2):
    
    A = np.array(os.listdir(path=path2))
    B = np.array([file.endswith(".ply") for file in A])
    A = A[B]
    return(A)

def normalize_vector(vec):
    
    norm_vec = np.copy(vec)
    for i,v in enumerate(vec):
        norm_vec[i] = v/np.linalg.norm(v)
    return(norm_vec)


def get_series_com(cells_path):
    
    startpoint,endpoint = uf_int.find_startpoint_endpoint(cells_path[0])

    com_cell1 = np.zeros((endpoint-startpoint+1,3))
    com_cell2 = np.zeros((endpoint-startpoint+1,3))
    for path in cells_path:
        filenames = get_names_ply(path)
        for file in filenames:
            t = int(file.split('time')[1].split('_')[0])
            mesh,V,T = uf.get_vertices_triangles(path+'/'+file)
            com_cell = uf.calc_centre_of_mass(V,T)
            
            if 'cell_1' in file: 
                com_cell1[t-1,:] = com_cell
                
            else :
                com_cell2[t-1,:] = com_cell
                
    
    return(np.array(com_cell1),np.array(com_cell2))

def bin_plot_negative(x, y, nbins):
    
    x = np.array(x)
    y = np.array(y)
    
    y_mean = np.zeros(nbins)
    y_std = np.zeros(nbins)
    nvalues = []
    minx = np.min(x)
    miny = np.min(y)
    maxy = np.max(y)
    
    shift_x = x-minx
    maxx = np.max(shift_x)
    
    for i in range(nbins):
        y_tmp = y[np.logical_and(shift_x>=i*maxx/nbins, shift_x<=((i+1)*maxx/nbins))]
        y_mean[i] = np.mean(y_tmp)
        y_std[i] = 1.96*np.std(y_tmp)/np.sqrt(len(y_tmp))
        nvalues.append(len(y_tmp))
    x_bin = np.linspace(0.5*maxx/nbins,maxx-0.5*maxx/nbins,nbins) + minx
    
    return(x_bin, y_mean, y_std, np.array(nvalues))

from random import choices
def bootstrap_double(data1,data2,nsample):
    #give p-value for mean of data1 being larger than mean of data2
    sample_diff=[]
    for i in range(nsample):
        y1=np.mean(choices(data1,k=len(data1)))
        y2=np.mean(choices(data2,k=len(data2)))
        sample_diff.append(y1-y2)
    
    
    return (sample_diff,np.sum(np.array(sample_diff)<0)/len(sample_diff))


cm = 1/2.54

fname = 'Arial'
font = font_manager.FontProperties(family=fname,
                                   weight='normal',
                                   style='normal', size=5)

#initialise the two figures to generate
fig0, ax0 = plt.subplots(figsize=(2.6*cm,1.6*cm))
fig01, ax01 = plt.subplots(figsize=(2.6*cm,1.6*cm))


paths = ['./blebbistatin_data/Segmentation_3bebbistatin-1_ecad/', 
         './blebbistatin_data/Segmentation_denoised_4_blebbistatin_ecad/',
         './blebbistatin_data/Segmentation_denoised_5_blebbistatin_ecad/',
         './blebbistatin_data/Segmentation_denoised_6_blebbistatin_ecad/',
         './blebbistatin_data/Segmentation_denoised_9_blebbistatin_ecad/']

img_names = ['3bebbistatin-1.tif', 
             '4-1raw-1.tif',
             '5-1rwa-1-1.tif',
             '6-raw-1.tif',
             '9_de-1-1.tif']


scale_factors = [1.0/0.103, 
                 1.0/0.103,
                 1.0/0.103,
                 1.0/0.103,
                 1.0/0.103]

#------------------------------------------------------------------------------
#Analysis loop, which also plots the trajectories of individual doublets
#------------------------------------------------------------------------------
time_omega_all = []
time_height_all = []
omega_all = []
height_all = []
for k,path in enumerate(paths) :
    
    print(path)
    
    img = imread(path+img_names[k])
    
    scale_factor = scale_factors[k]
    
    PATHS = [path + 'Cell_1', path+ 'Cell_2']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    files_1 = get_names_ply(PATHS[0])
    files_2 = get_names_ply(PATHS[1])
    
    
    com_cell1, com_cell2 = get_series_com(PATHS)
    
    u1 = normalize_vector(com_cell2-com_cell1)
    
    
    dt = 15 #in min
    time_int = dt*60 #in sec
    
    r_g = (com_cell1 + com_cell2)/2
    vec1 = com_cell1-r_g
    vec2 = com_cell2-r_g
    
    cross_product1 = np.cross(vec1[0:-1],vec1[1:])
    n1 = np.zeros(np.shape(cross_product1))
    
    for i,e in enumerate(cross_product1) :
        n1[i] = e/np.linalg.norm(e)
        
    dtheta1 = np.arccos(np.dot(vec1[0:-1],vec1[1:].T).diagonal(0,0,1)/np.linalg.norm(vec1[1:],axis=1)/np.linalg.norm(vec1[0:-1],axis=1))
    w1 = np.zeros(np.shape(n1))
    for i,e in enumerate(n1) :
        w1[i] = e * dtheta1[i]/time_int
    
    scale_rotation = 180/np.pi*60*60/360
    cell_velocity = np.linalg.norm(w1,axis=1)*scale_rotation
    
            
    time = np.linspace(0,(len(com_cell1)-1)*dt,len(com_cell1))
    
    
    timepoints = np.linspace(startpoint, endpoint, endpoint-startpoint+1,dtype=int)
    for t in timepoints:

        cell1, cell2 = bm.get_cells_cloud_time(t, PATHS)
        dist1, dist2 = ff_int.compute_distance_between_two_clouds(cell1, cell2)
        
        dist_threshold = 5
        
        interface_cloud = np.vstack((cell1[dist1<dist_threshold],cell2[dist2<dist_threshold]))
        rest_cells = np.vstack((cell1[dist1>=dist_threshold],cell2[dist2>=dist_threshold]))
        
        filename = 'x_y_h_interface_t_'+str(t)
        
        #save data of the interface in individual files, in case they must be 
        #analysed later
        save_path_interface=path+'x_y_H_interface/'
        data = ff_int.save_x_y_h_quantities(interface_cloud, save_path_interface, filename)
    

    
    hsq2 = np.zeros(timepoints.shape) 
    for i,t in enumerate(timepoints):
        
        XYH = np.load(path+'x_y_H_interface/x_y_h_interface_t_'+str(t)+'.npy')
        
        hsq,coeff,R,lamb,err=F.extract_modes(XYH)
        
        #We compute the interface deflection using the mode decomposition of the interface
        #amplitude of modes
        W0=coeff[0]**2
        W1=coeff[1]**2+coeff[2]**2
        WS=coeff[3]**2+coeff[4]**2
        WB=coeff[5]**2
        WPL=coeff[6]**2+coeff[7]**2
        WY=coeff[8]**2+coeff[9]**2
        
        hsq2[i]=R*np.sqrt(W0+W1+WS+WB+WPL+WY)/scale_factor/np.sqrt(np.pi) #version avec dimensions


    
    ax0.plot(time[:-1]-time[3],cell_velocity, 'o-', ms = 0.4, lw = 1, color = 'k', alpha = 0.1)
    ax01.plot(time-time[3],hsq2, 'o-', ms = 0.4, lw = 1, color = 'k', alpha = 0.1)
    
    time_omega_all += list(time[:-1]-time[3])
    time_height_all += list(time-time[3])
    omega_all += list(cell_velocity)
    height_all += list(hsq2)


time_height_all = np.array(time_height_all)
time_omega_all = np.array(time_omega_all)
omega_all = np.array(omega_all)
height_all = np.array(height_all)  

  
#------------------------------------------------------------------------------
#Computing means and errorbars for the rotation rate and the interface deflection
#as a function of time. Also performs the statistical tests
#------------------------------------------------------------------------------
t_omega_mean = np.mean(time_omega_all.reshape((5,9)),axis = 0)
t_height_mean = np.mean(time_height_all.reshape((5,10)),axis = 0)

omega_mean = np.mean(omega_all.reshape((5,9)),axis = 0)
omega_std = (1.96*np.std(omega_all.reshape((5,9)),axis = 0))/np.sqrt(5)
height_mean = np.mean(height_all.reshape((5,10)),axis = 0)
height_std = (1.96*np.std(height_all.reshape((5,10)),axis = 0))/np.sqrt(5)

array_om=omega_all.reshape((5,9))


#------------------------------------------------------------------------------
#Finishing the plots 5.b and 5.c by adding the average rotation rate and the average 
#interface deflection as a function of time
#------------------------------------------------------------------------------

ax0.errorbar(t_omega_mean, omega_mean, yerr = omega_std, color = 'r', linewidth = 0.8, markersize = 2)
ax0.axvline(time[3]-time[3],color ='red', lw = 0.8, ls = '--', alpha = 1.0, label = 'Blebbistatin addition')
ax0.set_ylabel(r'||$\vec{\omega}$|| (revolution.h$^{-1}$)', fontsize = 5, labelpad = 3, fontname=fname)
ax0.set_xlabel('Time (min)', fontsize =5, labelpad = 3,  fontname=fname)
ax0.set_xlim([np.min(time-time[3]), np.max(time-time[3])])
lx = [-50, -25 , 0, 25, 50, 75, 100]
x = [str(e) for e in lx]
ax0.set_xticks(lx)
ax0.set_xticklabels(x)
# ax0.set_ylim([0,0.7])
ly = [0.0, 0.25, 0.5]
y = [str(e) for e in ly]    
ax0.set_yticks(ly)
ax0.set_yticklabels(y)
ax0.tick_params(axis='both', which='major', labelsize=5, pad=2)
ax0.tick_params(axis='both', which='major', pad=2)

ax01.errorbar(t_height_mean, height_mean, yerr = height_std, color = 'r', linewidth = 0.8, markersize = 2)
ax01.axvline(time[3]-time[3],color ='red', lw = 0.8, ls = '--', alpha = 1.0, label = 'Blebbistatin addition')
ax01.set_ylabel(r'Interface height ($\mu$m)', fontsize =5, labelpad = 3, fontname=fname)
ax01.set_xlabel('Time (min)', fontsize =5, labelpad = 3,  fontname=fname)
ax01.tick_params(axis='both', which='major', labelsize=5, pad=2)
ax01.set_xlim([np.min(time-time[3]), np.max(time-time[3])])
lx = [-50, -25 , 0, 25, 50, 75, 100]
x = [str(e) for e in lx]
ax01.set_xticks(lx)
ax01.set_xticklabels(x)
ax01.set_ylim([0,3.0])
ax01.tick_params(axis='both', which='major', pad=2)

#Statistical tests for the decrease un rotation velocity (adjust sample number
# for more accurate p-value, it stays zero even with 100 000 samples)
#a,b=bootstrap_double(array_om[:,3],array_om[:,8],200000)
#b

#Alternative ttest that shows a p-value of 1.2.10^(-4)
#stats.ttest_ind(array_om[:,3],array_om[:,8],equal_var=False,alternative='greater')

#statistical test using bootstrapping for 
array_h=height_all.reshape((5,10))
a,b=bootstrap_double(array_h[:,3],array_h[:,9],2000000)
b



stats.ttest_ind(array_h[:,3],array_h[:,9],equal_var=False,alternative='greater')



a,b=bootstrap_double(array_h[:,0:4].flatten(),array_h[:,9],2000000)
b

data=plt.hist(a,bins=1000)

def Gauss(x,x0,a,b):
    y=a*np.exp(-b*(x-x0)**2)
    return y

from scipy.optimize import curve_fit

x=(data[1][:-1]+data[1][1:])*0.5
y=data[0]
par,cov=curve_fit(Gauss,x,y)

x0=par[0]
a=par[1]
b=par[2]
plt.figure()
plt.plot(x,y)
plt.plot(x,a*np.exp(-b*(x-x0)**2))



stats.ttest_ind(array_h[:,0:4].flatten(),array_h[:,9],equal_var=False,alternative='greater')







