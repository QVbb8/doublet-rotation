#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:56:04 2023

@author: vagne
"""


#------------------------------------------------------------------------------
#This script generates the plot of Extended Figure 4e and 4g. It constructs
#kymographs of the myosin signal around the spot of brightest intensity.
#
#In order to work, it requires the script 'calibration_z_2min.py' to have been
#run with the non-blurred images so that the file 'all_pz.txt' exists.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#Import packages, functions, and define plotting parameters
#------------------------------------------------------------------------------

import os
import pickle
import numpy as np
from skimage.io import imread
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.animation as animation
os.chdir('../Figure3_Figure4')
import useful_functions_interface as uf_int
import useful_functions_ply_files as uf
import function_final_interface as ff_int
os.chdir('../Supp_Fig1_single_cells')
import functions as func
os.chdir('../2min_time_resolution_analysis')
cm = 1/2.54

fname = 'Arial'
font = font_manager.FontProperties(family=fname,
                                    weight='normal',
                                    style='normal', size=5)

#------------------------------------------------------------------------------
#Loop over the unlburred doublets at 2min time resolution, and construct maps
#around the brightest spot
#This will generate pickle files to store each map in ...
#------------------------------------------------------------------------------
all_paths = ['./data_2min/13/',
             './data_2min/14/',
             './data_2min/15/']

all_img = ['13_ecad_mrlc.tif',
            '14_ecad_mrlc.tif',
            '15_ecad_mrlc.tif']

name = ['13','14','15']

scale_factors = [1/0.103,
                  1/0.103,
                  1/0.103]

#load z calibration data
all_pz=np.loadtxt('all_pz.txt')

Ntheta = 80
Nphi = 160

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
    for h,e in enumerate(cross_product1):
        n1[h] = e/np.linalg.norm(e)
        
    dtheta1 = np.arccos(np.dot(vec1[0:-1],vec1[1:].T).diagonal(0,0,1)/np.linalg.norm(vec1[1:],axis=1)/np.linalg.norm(vec1[0:-1],axis=1))
    
    M = func.rotmat(n1, dtheta1) #all rotation matrices between frames
        
    all_r12_omega_maps = [[],[]]
    
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
         
        # compute r12_omega_maps
        u1 = com_cell2[i] - com_cell1[i]
        
        
        #First option with maximum of intensity
        theta_r12_omega_c1, phi_r12_omega_c1, norm_c1 = func.get_angular_distribution_maxsignal_r12(V1, T1, mrlc_values_cell1, u1)
        theta_r12_omega_c2, phi_r12_omega_c2, norm_c2 = func.get_angular_distribution_maxsignal_r12(V2, T2, mrlc_values_cell2, -u1)
        
        theta_phi_r12_omega_c1 = np.zeros((Ntheta, Nphi)) 
        theta_phi_r12_omega_c2 = np.zeros((Ntheta, Nphi))
        

        func.get_angular_maps_interpolation(theta_r12_omega_c1, phi_r12_omega_c1, mrlc_values_cell1, theta_phi_r12_omega_c1)
        func.get_angular_maps_interpolation(theta_r12_omega_c2, phi_r12_omega_c2, mrlc_values_cell2, theta_phi_r12_omega_c2)
        
        
        all_r12_omega_maps[0].append(theta_phi_r12_omega_c1)
        all_r12_omega_maps[1].append(theta_phi_r12_omega_c2)
        
        
        i+=1
    all_p1.append([all_p1_single_doub1,all_p1_single_doub2])
        
        
    
    with open(f'./maps_pickle/maps{name[k]}_0.pickle', 'wb') as f:
        pickle.dump(all_r12_omega_maps[0], f)
    with open(f'./maps_pickle/maps{name[k]}_1.pickle', 'wb') as f:
        pickle.dump(all_r12_omega_maps[1], f)



#------------------------------------------------------------------------------
#For the plot of Extended Figure 4e and 4g, we use the cell 0 of doublet 14.
#First we generate Extended Figure 4e, showing the time frames 11 to 15
#------------------------------------------------------------------------------

# setoffiles=["maps13_0.pickle",
#             "maps13_1.pickle",
#             "maps14_0.pickle",
#             "maps14_1.pickle",
#             "maps15_0.pickle",
#             "maps15_1.pickle"]

setoffiles=["maps14_0.pickle"]

for fil in setoffiles:
    with open("./maps_pickle/"+fil,'rb') as f:
        maps=pickle.load(f)
    for i in range(11,16):
        fig,ax=plt.subplots()
        im=plt.imshow(maps[i],extent=[-180,180,180,0])
        plt.axvline(-60,c='r')
        plt.axvline(60,c='r')
        plt.title(str(i))

#------------------------------------------------------------------------------
#We then generate the kymograph of Extended Figure 4g 
#------------------------------------------------------------------------------

for fil in setoffiles:

    with open("./maps_pickle/"+fil,'rb') as f:
        maps=pickle.load(f)
    
    kymo=func.make_kymograph(maps,60)
    
    fig=plt.figure()
    plt.xlabel('time (min)')
    plt.ylabel(r'$\theta$')
    plt.imshow(kymo,extent=[0,2*(len(maps)-1),180,0],aspect=0.4)
        


    
    


