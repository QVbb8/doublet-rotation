#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:16:56 2024

@author: vagne
"""

#-----------------------------------------------------------------------------
#This script generates the plot of Extended Figure 1c, which shows the temporal
#dynamics of polarity in cell doublets vs single cells. It requires the single
#cell analysis to already have been performed. In the folder 
#'Supp_Fig1_single_cells', the script 'Supp_Fig1cd.py' must have been run so 
#that the file 'data_single_blur.pickle' already exists.
#-----------------------------------------------------------------------------


#Import necessary packages and functions
import os
os.chdir('../Figure3_Figure4')
import useful_functions_interface as uf_int
import useful_functions_ply_files as uf
import function_final_interface as ff_int
os.chdir('../Supp_Fig1_single_cells')
import functions as func
os.chdir('../2min_time_resolution_analysis')
import numpy as np
from skimage.io import imread
import pickle
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontm


#-----------------------------------------------------------------------------
#Extract polarity information from the dataset of the doublets taken at 2min
#time resolution and save it into a pickle file
#-----------------------------------------------------------------------------

all_paths = ['./data_2min/13/',
             './data_2min/14/',
             './data_2min/15/']

all_img = ['13_ecad_mrlc_blur.tif',
            '14_ecad_mrlc_blur.tif',
            '15_ecad_mrlc_blur.tif']

name = ['13','14','15']

scale_factors = [1/0.103,
                  1/0.103,
                  1/0.103]

#load z calibration data
all_pz=np.loadtxt('all_pz_blur.txt')

all_p1 = []

for k,path in enumerate(all_paths):
    
    print(path)
    
    img_name = all_img[k]
    scale_factor = scale_factors[k]
    
    dist_threshold = int(np.floor(scale_factor) + 1) 
    xy_pix = 5
    # xy_pix = int(np.floor(scale_factor) + 1)
    z_pix = 1
    
    
    img = imread(path+img_name)
    
    PATHS = [path + 'Cell_1', path+ 'Cell_2']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    
    com_cell1, com_cell2 = func.get_series_com(PATHS)
    
    r_g = (com_cell1 + com_cell2)/2
    
    vec1 = com_cell1-r_g
    vec2 = com_cell2-r_g


    cross_product1 = np.cross(vec1[0:-1],vec1[1:])
    n1 = np.zeros(np.shape(cross_product1))
    for h,e in enumerate(cross_product1) :
        n1[h] = e/np.linalg.norm(e)
        
    dtheta1 = np.arccos(np.dot(vec1[0:-1],vec1[1:].T).diagonal(0,0,1)/np.linalg.norm(vec1[1:],axis=1)/np.linalg.norm(vec1[0:-1],axis=1))
    
    M = func.rotmat(n1, dtheta1) #all rotation matrices between frames
        
    all_r12_omega_maps = [[],[]]
    all_norm_maps = [[],[]]
    
    all_p1_single_doub1 = []
    all_p1_single_doub2 = []
    
    i=0
    for t in range(startpoint, endpoint+1):
        print('Timepoint =', t)
        mesh1, V1, T1 = uf.get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')
        mesh2, V2, T2 = uf.get_vertices_triangles(PATHS[1]+'/time'+str(t)+'_cell_2.ply')
        im_mrlc = img[t-1,:,0,:,:] ### ECAD is 1
        
        
        #rescale the myosin intensity using an exponential curve to compensate for the effect of a distance to the microscope.
        im_mrlc=func.exponential_calibration(im_mrlc,scale_factor,-1/all_pz[k])
        
        dist1, dist2 = ff_int.compute_distance_between_two_clouds(V1, V2)
        dist_threshold = int(np.floor(scale_factor) + 1) 
        interface_cloud = np.vstack((V1[dist1<dist_threshold],V2[dist2<dist_threshold]))
        
        interface_V1 = np.zeros(len(V1))
        interface_V2 = np.zeros(len(V2))
        
        interface_V1[dist1<dist_threshold] = 1
        interface_V2[dist2<dist_threshold] = 1
        interface_V1 = interface_V1.astype(np.uint)
        interface_V2 = interface_V2.astype(np.uint)
        
        avg_inside_cell1 = func.get_signal_inside_cell(V1, im_mrlc, scale_factor)
        avg_inside_cell2 = func.get_signal_inside_cell(V2,im_mrlc, scale_factor)
                
        mrlc_values_cell1 = func.get_signal_vertices(V1, im_mrlc, xy_pix, z_pix, scale_factor, avg_inside_cell1, avg_inside_cell2, interface_V1)
        mrlc_values_cell2 = func.get_signal_vertices(V2, im_mrlc, xy_pix, z_pix, scale_factor, avg_inside_cell2, avg_inside_cell1, interface_V2)
        
        p1_c1,lol = func.polarity4(V1, T1, mrlc_values_cell1)
        p1_c2,lol = func.polarity4(V2, T2, mrlc_values_cell2)
        
        all_p1_single_doub1.append(p1_c1)
        all_p1_single_doub2.append(p1_c2)
        
        i+=1
    all_p1.append([all_p1_single_doub1,all_p1_single_doub2])
    
with open('polarities_doublet_blur.pickle','wb') as f:
    pickle.dump(all_p1,f)
    
#-----------------------------------------------------------------------------
#Load the single cell polarity data, and combine it with the doublet polarity
#data to create the plot of Extended Figure 1c
#-----------------------------------------------------------------------------

cm = 1/2.54

fname = 'Arial'
font = fontm.FontProperties(family=fname,
                                   weight='normal',
                                   style='normal', size=5)

#Single cells first
with open('../Supp_Fig1_single_cells/data_single_blur.pickle','rb') as f:
    inter_avg_ratio,max_avg_ratio,Polarities_single=pickle.load(f)
    
    
N=len(Polarities_single)
msd=[]
time_array=[]
dt=2 #minutes
for i in range(N):
    nt=len(Polarities_single[i])
    
    for i1 in range(nt-1):
        for i2 in range(i1+1,nt):
            pi1=Polarities_single[i][i1]/np.linalg.norm(Polarities_single[i][i1])
            pi2=Polarities_single[i][i2]/np.linalg.norm(Polarities_single[i][i2])
            
            dth=np.arccos(np.sum(pi1*pi2))
            msd.append(dth)
            time_array.append((i2-i1)*dt)
            
msd=np.array(msd)
time_array=np.array(time_array)

msd_mean=[0]
msd_err=[0]
time_mean=[0]
for i in range(1,int(max(time_array)/dt)):
    msd_mean.append(np.mean(msd[time_array==i*dt]))
    msd_err.append(1.96*np.std(msd[time_array==i*dt])/np.sqrt(len(msd[time_array==i*dt])))
    time_mean.append(i*dt)
    
fig=plt.figure(figsize=(6.7*cm,3.2*cm))
plt.errorbar(time_mean,msd_mean,yerr=msd_err,fmt='b-',linewidth=1)
plt.plot(time_array,msd,'bo',alpha=0.2,markersize=0.3)
plt.xlabel(r'$\Delta t$ (min)',fontsize =7, labelpad = 3, fontname=fname)
plt.ylabel(r'$\Delta \theta (\Delta t)$',fontsize =7, labelpad = 3, fontname=fname)
plt.ylim([0,np.pi])
plt.tick_params(axis='both', which='major', labelsize=7, pad=2)
plt.yticks([0,np.pi/2,np.pi],['0', r'$\pi/2$',r'$\pi$'],fontsize =7, fontname=fname)
            
#Cells in doublet second
with open('polarities_doublet_blur.pickle','rb') as f:
    Polarities_doub=pickle.load(f)
    
N=len(Polarities_doub)

msd=[]
time_array=[]

dt=2 #minutes
for i in range(N):
    nt=len(Polarities_doub[i][0])
    for i1 in range(nt-1):
        for i2 in range(i1+1,nt):
            for c in range(2):
                pi1=Polarities_doub[i][c][i1]/np.linalg.norm(Polarities_doub[i][c][i1])
                pi2=Polarities_doub[i][c][i2]/np.linalg.norm(Polarities_doub[i][c][i2])
                dth=np.arccos(np.sum(pi1*pi2))
                msd.append(dth)
                time_array.append((i2-i1)*dt)
           
msd=np.array(msd)
time_array=np.array(time_array)

msd_mean=[0]
msd_err=[0]
time_mean=[0]
for i in range(1,int(max(time_array)/dt)):
    msd_mean.append(np.mean(msd[time_array==i*dt]))
    msd_err.append(1.96*np.std(msd[time_array==i*dt])/np.sqrt(len(msd[time_array==i*dt])))
    time_mean.append(i*dt)
   
plt.errorbar(time_mean,msd_mean,yerr=msd_err,fmt='r-',linewidth=1)
plt.plot(time_array,msd,'ro',alpha=0.2,markersize=0.3)


