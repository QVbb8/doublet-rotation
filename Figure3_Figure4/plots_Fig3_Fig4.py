#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:07:10 2023

@author: vagne
"""

#------------------------------------------------------------------------------
#This script generates the plots of Figure 3 and Figure 4 (as well as Extended
#Figure 7).

#First one must run 'calibration_pz.py' to generate the file 'all_pz_blur.txt' 
#that contains the corrections to apply to the myosin signal for each doublet
#It also generates the plots in  Extended Figure 5.d and Extended Figure 5.e
#------------------------------------------------------------------------------

import os
import pickle
import numpy as np
import pandas as pd
import open3d as o3d
from skimage.io import imread, imsave
from tifffile import imwrite
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d
from skimage import measure
from pyntcloud import PyntCloud
from pyntcloud.geometry.models.plane import Plane
from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cmap
import matplotlib.font_manager as font_manager
from pyevtk.hl import pointsToVTK
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import functions_doublet as F
import read_ply
import useful_functions_ply_files as uf
import useful_functions_interface as uf_int
import function_final_interface as ff_int
import functions_figure4 as F4

from random import choices
#bootstraping function for the statistical test in Fig. 3j
def bootstrap_beta(beta,beta_ref,pos,nsample):
    #give p-value for the average "angle" of beta being larger (True) or
    #lower (False) than beta_ref
    sample_beta_angle=[]
    for i in range(nsample):
        y=choices(beta,k=len(beta))
        vx=np.mean(np.cos(y))
        vy=np.mean(np.sin(y))
        beta_angle=np.arctan2(vy,vx)%(2*np.pi)
        
        sample_beta_angle.append(beta_angle)

    if pos==True:
        return (sample_beta_angle, np.sum(np.array(sample_beta_angle)<beta_ref)/len(sample_beta_angle))
    else:
        return (sample_beta_angle, np.sum(np.array(sample_beta_angle)>beta_ref)/len(sample_beta_angle))






#load the 'all_pz_blur.txt' calibration file for the myosin signal
all_pz=np.loadtxt("all_pz_blur.txt")


cm = 1/2.54

fname = 'Arial'
font = font_manager.FontProperties(family=fname,
                                    weight='normal',
                                    style='normal', size=5)


all_paths = ['./myosin_data/b1/Segmentation_Denoised-1_B1_ecad/',
              './myosin_data/b1/Segmentation_2_B1_demo42_s1-2_ecad/',
              './myosin_data/b3/Segmentation_Denoised-1_demo43_s3-1-1_ecad/',
              './myosin_data/b3/Segmentation_Denoised-8_demo43_s7-1-1_ecad/',
              './myosin_data/b3/Segmentation_Denoised-2_demo43_s4-1-1_ecad/',
              './myosin_data/b3/Segmentation_Denoised-6_demo43_s5-1-1_ecad/',
              './myosin_data/b3/Segmentation_Denoised-9_demo43_s7-1-1_ecad/',
              './myosin_data/b4/Segmentation_Denoised-7_b5_ecadherinmrlcp1_s5-1-1/',
              './myosin_data/b4/Segmentation_Denoised-2_b5_ecadherinmrlcp1_s3-1-1/',
              './myosin_data/b4/Segmentation_Denoised-3_b5_ecadherinmrlcp1_s3-1-1/',
              './myosin_data/b4/Segmentation_Denoised-5_b5_ecadherinmrlcp1_s4-1-1/',
              './myosin_data/b4/Segmentation_Denoised-1_b5_ecadherinmrlcp1_s1-1-1/']

all_img = ['1_ecad_mrlc_blur.tif',
            '2_ecad_mrlc_blur.tif',
            '3_ecad_mrlc_blur.tif',
            '4_ecad_mrlc_blur.tif',
            '5_ecad_mrlc_blur.tif',
            '6_ecad_mrlc_blur.tif',
            '7_ecad_mrlc_blur.tif',
            '8_ecad_mrlc_blur.tif',
            '9_ecad_mrlc_blur.tif',
            '10_ecad_mrlc_blur.tif',
            '11_ecad_mrlc_blur.tif',
            '12_ecad_mrlc_blur.tif']


scale_factors = [1/0.206,
                  1/0.206,
                  1/0.103,
                  1/0.103,
                  1/0.103,
                  1/0.103,
                  1/0.103,
                  1/0.103,
                  1/0.103,
                  1/0.103,
                  1/0.103,
                  1/0.103]

#initialiasing variables
ang_velocity = []
all_polar_maps = [[] for e in all_paths] + [[]for e in all_paths]
all_r12_omega_maps = [[] for e in all_paths] + [[]for e in all_paths]
all_r12_omega_interface_maps = [[] for e in all_paths] + [[]for e in all_paths]
avg_r12_omega_interface_maps = []
norm_p1_c1 = []
norm_p1_c2 = []
dQ = [] #Q1-Q2
dQn = [] #(Q1-Q2)/|Q1-Q2|
NdQ = [] #|Q1-Q2|
Qs= [] #Q of saddle node multiplide by saddle_node amplitude sqrt(saddle_node_amp)
Qsn= [] #Qs/|Qs|
nth_Q=50
Theta1_binned=[]
Theta2_binned=[]
S2d = [] #2d vectors of saddle node in reference frame of interface and omega
yin_yang_height = []
interface_height = []
dgamma = []
db = []
saddle_node_amp = []
lam = []
bowl_amplitude = []
yin_yang_amp = []
pl_amp = []
dgamma_mean = []
b_mean = []
ddgamma_avg = []
db_avg = []
lam_avg = []
gamma_avg = []
b_avg = []
p1_sum_avg = []
all_p1_dot_ydir = []
all_p1_dot_sdir = []
all_omega_corr_p1_diff_avg = []
all_omega_p1_diff_not_proj =[]
all_alpha_p1 = []
all_beta_p1 = []
all_delta_signal = []
all_int_signal_c1 = []
all_int_signal_c2 = []
all_sigma_c1=[]
all_sigma_c2=[]
all_r12_cross_p1mp2_norm = []
#storing r12, rg, omega vector, polarity vectors
all_r12=[]
all_rg=[]
all_w=[]
all_p1_c1 =[]
all_p1_c2=[]
all_vg=[]
#storing ratio of intensities
max_avg_ratio=[]
inter_avg_ratio=[]
indices_doublets=[]

for k,path in enumerate(all_paths):
    PATHS = [path + 'Cell_1', path+ 'Cell_2']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    indices_doublets.append(endpoint-startpoint)

indices_cum=np.concatenate((np.array([0]),np.cumsum(indices_doublets)))




#------------------------------------------------------------------------------
#This part of the script analyses all the data in the './myosin_data' folder 
#which contains segmented meshes of cells as well as blurred 3d microscopy videos
#of the myosin signal
#
#This part can be run once and then all the relevant data is saved in a pickle
#file 'data_blur.pkl' so that the subsequent analysis and plot generations can be done 
#independently if needed.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#Loop of analysis - START
#------------------------------------------------------------------------------
for k,path in enumerate(all_paths):
    
    print(path)
    img_name = all_img[k]
    scale_factor = scale_factors[k]
    
    dist_threshold = int(np.floor(scale_factor) + 1) 
    xy_pix = 5
    z_pix = 1
    
    img = imread(path+img_name)

    PATHS = [path + 'Cell_1/', path+ 'Cell_2/']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    
    com_cell1, com_cell2 = F4.get_series_com(PATHS)
    vol_cell1, vol_cell2 = F4.get_series_volumes(PATHS)
    com_signal1 = np.zeros(com_cell1.shape)
    com_signal2 = np.zeros(com_cell2.shape)
    sum_polarities = []
    all_mrlc_values = []
        
    Ntheta = 80
    Nphi = 160
    
    p1_c1_list = []
    p1_c2_list = []
    p1_c1_alpha = []
    p1_c2_alpha = []
    p1_beta = []
    p1_dot_ydir = []
    p1_dot_sdir = []
    omega_corr_p1_diff_avg = []
    omega_p1_diff_not_proj = []
    r12_cross_p1mp2_norm = []

    delta_signal_c1 = []
    delta_signal_c2 = []
    
    r_g = (com_cell1 + com_cell2)/2
    all_rg += r_g[:-1,:].tolist() #skip last point
    all_vg += (r_g[1:,:]-r_g[:-1,:]).tolist()
    
    all_r12 += (com_cell2[:-1,:] - com_cell1[:-1,:]).tolist() #skip last point
    vec1 = com_cell1-r_g
    vec2 = com_cell2-r_g

    dt = 10 #in min
    time_int = dt*60 #in sec
    time = np.linspace(0,(len(com_cell1)-1)*dt,len(com_cell1))

    cross_product1 = np.cross(vec1[0:-1],vec1[1:])
    n1 = np.zeros(np.shape(cross_product1))
    for h,e in enumerate(cross_product1) :
        n1[h] = e/np.linalg.norm(e)
    
    
    dtheta1 = np.arccos(np.dot(vec1[0:-1],vec1[1:].T).diagonal(0,0,1)/np.linalg.norm(vec1[1:],axis=1)/np.linalg.norm(vec1[0:-1],axis=1))
    w1 = np.zeros(np.shape(n1))
    for i,e in enumerate(n1) :
        w1[i] = e * dtheta1[i]/time_int
        ang_velocity.append(np.linalg.norm(w1[i])*180/np.pi*60*60/360)
    all_w += w1.tolist()
    
    
    i = 0

    for t in range(startpoint, endpoint): #normalement jusqu'à endpoint + 1 mais à cause de omega on perd un timepoint
        
        print('Timepoint =', t)
        mesh1, V1, T1 = uf.get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')
        mesh2, V2, T2 = uf.get_vertices_triangles(PATHS[1]+'/time'+str(t)+'_cell_2.ply')
        im_mrlc = img[t-1,:,0,:,:] ### ECAD is 1
        
        #rescale the myosin intensity using an exponential curve to compensate for the effect of a distance to the microscope.
        im_mrlc=F4.exponential_calibration(im_mrlc,scale_factor,-1/all_pz[k])
        
        
        dist1, dist2 = ff_int.compute_distance_between_two_clouds(V1, V2)
        dist_threshold = int(np.floor(scale_factor) + 1) 
        interface_cloud = np.vstack((V1[dist1<dist_threshold],V2[dist2<dist_threshold]))
        
        interface_V1 = np.zeros(len(V1))
        interface_V2 = np.zeros(len(V2))
        
        interface_V1[dist1<dist_threshold] = 1
        interface_V2[dist2<dist_threshold] = 1
        interface_V1 = interface_V1.astype(np.uint)
        interface_V2 = interface_V2.astype(np.uint)
        
        #no more of this, to see if it works with a simpler method
        avg_inside_cell1 = F4.get_signal_inside_cell(V1, im_mrlc, scale_factor)
        avg_inside_cell2 = F4.get_signal_inside_cell(V2,im_mrlc, scale_factor)
              
        mrlc_values_cell1,mean_b1,max_b1 = F4.get_signal_vertices(V1, im_mrlc, xy_pix, z_pix, scale_factor, avg_inside_cell1, avg_inside_cell2, interface_V1)
        mrlc_values_cell2,mean_b2,max_b2 = F4.get_signal_vertices(V2, im_mrlc, xy_pix, z_pix, scale_factor, avg_inside_cell2, avg_inside_cell1, interface_V2)
        
        
        max_avg_ratio.append(max_b1/avg_inside_cell1)
        max_avg_ratio.append(max_b2/avg_inside_cell2)
        
        inter_avg_ratio.append(mean_b1/avg_inside_cell1)
        inter_avg_ratio.append(mean_b2/avg_inside_cell2)
        
        
        com_1 = com_cell1[i]
        com_2 = com_cell2[i]
        
        #Potentially record a vtu file of cell 1 and cell 2 with the myosin data on it
        #F4.save_mesh_vtu(V1, T1, [mrlc_values_cell1], ["myosin"], path+'vtu_files_blur/cell1_time_'+str(t))
        #F4.save_mesh_vtu(V2, T2, [mrlc_values_cell2], ["myosin"], path+'vtu_files_blur/cell2_time_'+str(t))
        
        #get_saddle_node
        u1 = com_cell2[i] - com_cell1[i]
        
        [ex, ey, Nint], [X,Y,Hfit] = F4.naive_fit_plane_oriented(interface_cloud, u1)
        com_interface=np.mean(interface_cloud,axis=0)
        XYH = np.vstack((np.vstack((X,Y)),Hfit)).T
        hsq,coeff,R,lamb,err=F.extract_modes(XYH)
        
        W0=coeff[0]**2
        W1=coeff[1]**2+coeff[2]**2
        WS=coeff[3]**2+coeff[4]**2
        WB=coeff[5]**2
        WB_signed = coeff[5]
        WPL=coeff[6]**2+coeff[7]**2
        WY=coeff[8]**2+coeff[9]**2
        
        yin_yang_height.append(np.sqrt(WY)/np.sqrt(np.pi))
        interface_height.append(np.sqrt(W0+W1+WS+WB+WPL+WY)/np.sqrt(np.pi))
    
        saddle_node_amp.append(WS)
        bowl_amplitude.append(WB_signed) #*np.abs(WB_signed)/(WS+WB+WPL+WY))
        yin_yang_amp.append(WY)
        pl_amp.append(WPL)
                
        #coeff 8 and 9 correspond to Y1 and Y2 for the directioni of the Yin-Yang
        ydirtemp = (ex*coeff[8]+ey*coeff[9])/np.sqrt(coeff[8]**2+coeff[9]**2)
        
        alpha= 0.5*np.arctan2(coeff[4],coeff[3])+np.pi
        sdirtmp = ex*np.cos(alpha) + ey*np.sin(alpha)
        #we also need an orthogonal vector to get the proper Q tensor for the saddle-node
        sdirtmp_orth = -ex*np.sin(alpha) + ey*np.cos(alpha)
        
        #record cell in a band along Nint centered around the interface of size DeltaH
        DeltaH=20
        tc1=F4.get_tricentres(V1,T1)
        dist1=np.sum((tc1-com_interface)*Nint[None,:],axis=1)
        ind1=np.logical_and(dist1 <= DeltaH,dist1 >= -DeltaH)
        
        tc2=F4.get_tricentres(V2,T2)
        dist2=np.sum((tc2-com_interface)*Nint[None,:],axis=1)
        ind2=np.logical_and(dist2 <= DeltaH,dist2 >= -DeltaH)
        
        #go from triangle indices to vertices indices
        vind1=np.unique(T1[ind1,:].flatten())
        vind2=np.unique(T2[ind2,:].flatten())
        
        vind1=F4.bool_array_from_index_list(V1.shape[0],vind1)
        vind2=F4.bool_array_from_index_list(V2.shape[0],vind2)
        
        
        ## nematic tensor of myosin distribution
        
        Q_int1 = F4.Q_tensor3d_myosin_simple(V1, T1, mrlc_values_cell1)
        Q_int2 = F4.Q_tensor3d_myosin_simple(V2, T2, mrlc_values_cell2)
        #get the eigenvectors to make a graphic representation
        #eigenvectors
        w,evc=np.linalg.eig(Q_int1)
        wM=w[np.argmax(w)]
        evc1=evc[:,np.argmax(w)]

        w,evc=np.linalg.eig(Q_int2)
        wM=w[np.argmax(w)]
        evc2=evc[:,np.argmax(w)]
        
        ## projection of myosin intensity in polar coordinates (cylindrical) around the center of mass of the doublet
        ## Using omega and Nint to make a reference frame
        
        #project omega in interface plane
        omega_proj=n1[i]-np.dot(n1[i],Nint)*Nint
        #frame of reference
        ex_pol= omega_proj/np.linalg.norm(omega_proj)
        ez_pol= np.copy(Nint)
        ey_pol= np.cross(ez_pol,ex_pol)
        x1_pol = np.sum((V1-r_g[i][None,:])*ex_pol[None,:],axis=1)
        y1_pol = np.sum((V1-r_g[i][None,:])*ey_pol[None,:],axis=1)
        
        x2_pol = np.sum((V2-r_g[i][None,:])*ex_pol[None,:],axis=1)
        y2_pol = np.sum((V2-r_g[i][None,:])*ey_pol[None,:],axis=1)
        
        #store the coordinates of sdirtmp in the (ex_pol,ey_pol,ez_pol) reference frame
        s2d=np.array([np.sum(sdirtmp*ex_pol),np.sum(sdirtmp*ey_pol)])
        s2d=s2d/np.linalg.norm(s2d)
        S2d.append(s2d)
        
        
        theta1=np.arctan2(y1_pol,x1_pol)
        theta2=np.arctan2(y2_pol,x2_pol)
        #bin signal and store
        theta1_binned=np.zeros(nth_Q)
        theta2_binned=np.zeros(nth_Q)
        for ind_th in range(nth_Q):
            indices_theta=np.logical_and(theta1 >= -np.pi+ind_th*(2*np.pi)/nth_Q,theta1 <= -np.pi+(ind_th+1)*(2*np.pi)/nth_Q)
            #incorporate here to remain in a small band around the interface
            indices_theta=np.logical_and(indices_theta,vind1)
            theta1_binned[ind_th]=np.mean(mrlc_values_cell1[indices_theta])
            
            indices_theta=np.logical_and(theta2 >= -np.pi+ind_th*(2*np.pi)/nth_Q,theta2 <= -np.pi+(ind_th+1)*(2*np.pi)/nth_Q)
            indices_theta=np.logical_and(indices_theta,vind2)
            theta2_binned[ind_th]=np.mean(mrlc_values_cell2[indices_theta])
            
        Theta1_binned.append(theta1_binned)
        Theta2_binned.append(theta2_binned)
        
        
        ## create a tensor associated to sdirtemp
        
        #Q_sdir = create_Q_from_vec(sdirtmp)
        Q_sdir = F4.create_Qs_from_2vec(sdirtmp,sdirtmp_orth)
        
                
        dQint = Q_int1 - Q_int2
        norm2_dQ_int = np.sum((dQint*dQint).flatten())
        
        
        w,evc=np.linalg.eig(dQint)
        wM=w[np.argmax(w)]
        evc1=evc[:,np.argmax(w)]
        
        dQ.append(dQint) #Q1-Q2
        dQn.append(dQint/np.sqrt(norm2_dQ_int)) #(Q1-Q2)/|Q1-Q2|
        NdQ.append(np.sqrt(norm2_dQ_int)) #|Q1-Q2|
        Qs.append(Q_sdir*np.sqrt(WS)) #Q of saddle node multiplide by saddle_node amplitude sqrt(saddle_node_amp)
        Qsn.append(Q_sdir) #Qs/|Qs|
        
        ###compute polarities
        # first method is to compute the center of mass of the signal
        # method 1 calc_centre_of_mass_signal1
        
        p1_c1,lol = F4.polarity4(V1, T1, mrlc_values_cell1)
        p1_c2,lol = F4.polarity4(V2, T2, mrlc_values_cell2)
        
        p1n=p1_c1/np.linalg.norm(p1_c1)
        theta_c1=np.arccos(p1n[2])
        phi_c1=np.arctan2(p1n[1],p1n[0])
        
        p2n=p1_c2/np.linalg.norm(p1_c2)
        theta_c2=np.arccos(p2n[2])
        phi_c2=np.arctan2(p2n[1],p2n[0])

        
        
        p1_c1_list.append(p1_c1)
        p1_c2_list.append(p1_c2)
        
        
        norm_p1_c1.append(np.linalg.norm(p1_c1))
        p1_c1_proj = p1_c1 - np.dot(p1_c1,Nint)*Nint
        

        norm_p1_c2.append(np.linalg.norm(p1_c2))
        p1_c2_proj = p1_c2 - np.dot(p1_c2,Nint)*Nint
        
        p1_sum_avg.append((np.linalg.norm(p1_c1)+np.linalg.norm(p1_c2))/2)
        
        p1_diff_avg = (p1_c1_proj-p1_c2_proj)/2
        p1_diff_avg /= np.linalg.norm(p1_diff_avg)
        
        p1_diff_avg_notproj = (p1_c1-p1_c2)/2
        p1_diff_avg_notproj /= np.linalg.norm(p1_diff_avg_notproj)
        
        
        p1_dot_ydir.append(np.dot(p1_diff_avg, ydirtemp))
        p1_dot_sdir.append(np.cos(4*np.arccos(np.dot(p1_diff_avg, sdirtmp))))
        
        u1_cross_p1_diff_avg = np.cross(u1, p1_diff_avg)
        u1_cross_p1_diff_avg /= np.linalg.norm(u1_cross_p1_diff_avg)
        
        u1_cross_p1_diff_notproj = np.cross(u1,p1_diff_avg_notproj)
        
        r12_cross_p1mp2_norm.append(np.linalg.norm(u1_cross_p1_diff_notproj))
        
        u1_cross_p1_diff_notproj /= np.linalg.norm(u1_cross_p1_diff_notproj)
        
        omega_corr_p1_diff_avg.append(np.dot(n1[i], u1_cross_p1_diff_avg))
        omega_p1_diff_not_proj.append(np.dot(n1[i], u1_cross_p1_diff_notproj))
        
        p1_c1_alpha.append(F4.compute_alpha(p1_c1, u1))
        p1_c2_alpha.append(F4.compute_alpha(p1_c2, -u1))
        beta_tmp=F4.compute_beta2(p1_c1, p1_c2, u1)
        p1_beta.append(beta_tmp)
        
        
        theta_p1_c1, phi_p1_c1 = F4.get_angular_distribution_polarity(V1, T1, mrlc_values_cell1, u1, p1_c1)
        theta_p1_c2, phi_p1_c2 = F4.get_angular_distribution_polarity(V2, T2, mrlc_values_cell2, -u1, p1_c2)
        
        theta_phi_p1_c1 = np.zeros((Ntheta, Nphi))
        theta_phi_p1_c2 = np.zeros((Ntheta, Nphi))
        
        F4.get_angular_maps_interpolation(theta_p1_c1, phi_p1_c1, mrlc_values_cell1, theta_phi_p1_c1)
        F4.get_angular_maps_interpolation(theta_p1_c2, phi_p1_c2, mrlc_values_cell2, theta_phi_p1_c2)
        
        all_polar_maps[2*k].append(theta_phi_p1_c1)
        all_polar_maps[2*k+1].append(theta_phi_p1_c2)
        
        dg_c1, b_c1, int_signal_c1 = F4.get_tension_mod_fit(Ntheta, Nphi, theta_phi_p1_c1)
        dg_c2, b_c2, int_signal_c2 = F4.get_tension_mod_fit(Ntheta, Nphi, theta_phi_p1_c2)
        
        all_sigma_c1.append(F4.get_std_over_avg(Ntheta, Nphi, theta_phi_p1_c1))
        all_sigma_c2.append(F4.get_std_over_avg(Ntheta, Nphi, theta_phi_p1_c2))
        
        
        all_int_signal_c1.append(int_signal_c1)
        all_int_signal_c2.append(int_signal_c2)
        
        lammbda = int_signal_c1/int_signal_c2
        
        lam.append(lammbda)
        
        dgamma.append(dg_c1)
        dgamma.append(dg_c2)
        db.append(b_c1)
        db.append(b_c2)
        
        dgamma_mean.append((dg_c1+dg_c2)/2)
        b_mean.append(b_c1/2+b_c2/2)
        
        delta_signal_c1.append(F4.compute_delta_signal(V1, T1, mrlc_values_cell1))
        delta_signal_c2.append(F4.compute_delta_signal(V2, T2, mrlc_values_cell2))
        
        
        # compute r12_omega_maps
        
        theta_r12_omega_c1, phi_r12_omega_c1 = F4.get_angular_distribution_r12_omega(V1, T1, mrlc_values_cell1, u1, n1[i])
        theta_r12_omega_c2, phi_r12_omega_c2 = F4.get_angular_distribution_r12_omega(V2, T2, mrlc_values_cell2, -u1, n1[i])
        
        theta_phi_r12_omega_c1 = np.zeros((Ntheta, Nphi)) 
        theta_phi_r12_omega_c2 = np.zeros((Ntheta, Nphi)) 

        F4.get_angular_maps_interpolation(theta_r12_omega_c1, phi_r12_omega_c1, mrlc_values_cell1, theta_phi_r12_omega_c1)
        F4.get_angular_maps_interpolation(theta_r12_omega_c2, phi_r12_omega_c2, mrlc_values_cell2, theta_phi_r12_omega_c2)
        
        all_r12_omega_maps[2*k].append(theta_phi_r12_omega_c1)
        all_r12_omega_maps[2*k+1].append(theta_phi_r12_omega_c2)
        
        theta_phi_interface_c1 = np.zeros((Ntheta, Nphi))
        theta_phi_interface_c2 = np.zeros((Ntheta, Nphi))
        
        theta_int_c1, phi_int_c1 = F4.get_angular_distribution_r12_omega(V1, T1, interface_V1, u1, n1[i]) 
        theta_int_c2, phi_int_c2 = F4.get_angular_distribution_r12_omega(V2, T2, interface_V2, -u1, n1[i]) 

        F4.get_angular_maps_interpolation(theta_int_c1, phi_int_c1, interface_V1, theta_phi_interface_c1)
        F4.get_angular_maps_interpolation(theta_int_c2, phi_int_c2, interface_V2, theta_phi_interface_c2)
        
        all_r12_omega_interface_maps[2*k].append(theta_phi_interface_c1)
        all_r12_omega_interface_maps[2*k+1].append(theta_phi_interface_c2)
        
        i+=1
        
    p1_c1_array=np.array(p1_c1_list)
    p1_c2_array=np.array(p1_c2_list)
    
    p_tot=np.concatenate((p1_c1_array,p1_c2_array),axis=0)
    
    norm=np.sqrt(np.sum(p_tot**2,axis=1))
    p_totn=p_tot/norm[:,None]


    
    psum=p1_c1_array+p1_c2_array
    norm=np.sqrt(np.sum(psum**2,axis=1))
    psum=psum/norm[:,None]
    
    
    avg_r12_omega_interface_maps.append(np.sum(all_r12_omega_interface_maps[2*k], axis = 0)/i)
    avg_r12_omega_interface_maps.append(np.sum(all_r12_omega_interface_maps[2*k+1], axis = 0)/i)
    
    
    all_p1_dot_ydir += p1_dot_ydir
    all_p1_dot_sdir += p1_dot_sdir 
    all_omega_corr_p1_diff_avg += omega_corr_p1_diff_avg 
    all_omega_p1_diff_not_proj += omega_p1_diff_not_proj
    all_alpha_p1 += p1_c1_alpha + p1_c2_alpha
    all_beta_p1 += p1_beta
    all_delta_signal += delta_signal_c1 + delta_signal_c2
    all_p1_c1 += p1_c1_list
    all_p1_c2 += p1_c2_list
    all_r12_cross_p1mp2_norm += r12_cross_p1mp2_norm

    print('Number of doublet = ', int(len(all_alpha_p1)/2))
    


all_w=np.array(all_w)
all_p1_c1=np.array(all_p1_c1)
all_p1_c2=np.array(all_p1_c2)
all_rg=np.array(all_rg)
all_vg=np.array(all_vg)
all_r12=np.array(all_r12)


#Save the data generated thus far
with open('data_blur.pkl','wb') as f:
    list_data=[all_w,all_p1_c1,all_p1_c2,all_rg,all_r12,all_alpha_p1,all_beta_p1]
    list_data += [all_omega_corr_p1_diff_avg,all_omega_p1_diff_not_proj,all_p1_dot_ydir]
    list_data += [all_r12_omega_maps,avg_r12_omega_interface_maps,all_polar_maps,dQ,dQn,NdQ,Qs,Qsn,Theta1_binned,Theta2_binned,S2d]
    list_data += [p1_sum_avg,ang_velocity,yin_yang_height,all_int_signal_c1,all_int_signal_c2]
    list_data += [bowl_amplitude,all_sigma_c1,all_sigma_c2,saddle_node_amp,yin_yang_amp,pl_amp]
    list_data += [max_avg_ratio,inter_avg_ratio]
    pickle.dump(list_data,f)
    
#------------------------------------------------------------------------------
#Loop of analysis - END
#------------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
#This part generates all the plots of Figure 3 and Figure 4
#The bootstrapping statistical tests are commented out (or performed with a low
# number of samples) because they slow down the execution of the code, but they
# can be un-commented if needed.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#load the data from 'data_blur.pkl'
#------------------------------------------------------------------------------
    
with open('data_blur.pkl','rb') as f:  # Python 3: open(..., 'rb')
    all_w,all_p1_c1,all_p1_c2,all_rg,all_r12,all_alpha_p1,all_beta_p1, \
    all_omega_corr_p1_diff_avg,all_omega_p1_diff_not_proj,all_p1_dot_ydir, \
    all_r12_omega_maps,avg_r12_omega_interface_maps,all_polar_maps,dQ,dQn,NdQ,Qs,Qsn,Theta1_binned,Theta2_binned,S2d, \
    p1_sum_avg,ang_velocity,yin_yang_height,all_int_signal_c1,all_int_signal_c2,\
    bowl_amplitude,all_sigma_c1,all_sigma_c2,saddle_node_amp,yin_yang_amp,pl_amp,\
    max_avg_ratio,inter_avg_ratio = pickle.load(f)
    

#------------------------------------------------------------------------------
#Figure 3.i - Angles alpha,beta of the myosin polarity vectors for doublet 1
#------------------------------------------------------------------------------
    
all_alpha_p1=np.array(all_alpha_p1)
all_beta_p1=np.array(all_beta_p1)

alpha1=all_alpha_p1[0::2]
alpha2=all_alpha_p1[1::2]


doublet=1

time_array=np.linspace(0,(indices_cum[doublet+1]-indices_cum[doublet])*10,indices_cum[doublet+1]-indices_cum[doublet]+1)

fig,ax=plt.subplots(figsize=(4.2*cm,3.2*cm))
ax.plot(time_array[:-1],alpha1[indices_cum[doublet]:indices_cum[doublet+1]],color="#FFCE0A", linewidth = 1, markersize = 5)
ax.plot(time_array[:-1],alpha2[indices_cum[doublet]:indices_cum[doublet+1]],color="#FF0AE7", linewidth = 1, markersize = 5)
ax.plot(time_array[:-1],all_beta_p1[indices_cum[doublet]:indices_cum[doublet+1]],color="#1646DB", linewidth = 1, markersize = 5)
ax.axhline(np.pi,color='k',linestyle='--')
ax.axhline(np.pi/2,color='k',linestyle='--')
ax.set_ylim([0,2*np.pi])
ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax.set_xlabel("Time (min)",fontsize =7, labelpad = 2, fontname=fname)
ax.set_ylabel("", fontsize =7, labelpad = 2, fontname=fname)

lx = [0, 150, 300,450,600]
x = [str(e) for e in lx]

ax.set_xticks(lx)
ax.set_xticklabels(x)
ax.set_xlim([0,700])

# y = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$',r'$2\pi$']
# l = [0, np.pi/2, np.pi,3*np.pi/2,2*np.pi]

y = ['0', '90', '180', '270','360']
l = [0, np.pi/2, np.pi,3*np.pi/2,2*np.pi]

ax.set_yticks(l)
ax.set_yticklabels(y)

#------------------------------------------------------------------------------
#Figure 3.j - Distribution of the angle beta
#------------------------------------------------------------------------------

norm=np.sqrt(np.sum(all_w**2,axis=1))
wn=all_w/norm[:,None]
#rewrite polarities in the (omega,r12xomega)
norm=np.sqrt(np.sum(all_r12**2,axis=1))
ez=all_r12/norm[:,None]
ex=wn
ey=np.cross(ez,ex)

pc1=np.zeros(all_p1_c1.shape)
pc2=np.zeros(all_p1_c2.shape)

pc1[:,0]=np.sum(all_p1_c1*ex,axis=1)
pc1[:,1]=np.sum(all_p1_c1*ey,axis=1)
pc1[:,2]=np.sum(all_p1_c1*ez,axis=1)

pc2[:,0]=np.sum(all_p1_c2*ex,axis=1)
pc2[:,1]=np.sum(all_p1_c2*ey,axis=1)
pc2[:,2]=np.sum(all_p1_c2*ez,axis=1)

norm=np.sqrt(np.sum(pc1**2,axis=1))
pc1n=pc1/norm[:,None]
norm=np.sqrt(np.sum(pc2**2,axis=1))
pc2n=pc2/norm[:,None]

#phi1-phi2
c1_phi=np.arctan2(pc1n[:,1],pc1n[:,0])
c2_phi_same=np.arctan2(pc2n[:,1],pc2n[:,0])

beta=(c2_phi_same-c1_phi)%(2*np.pi)


vx=np.mean(np.cos(beta))
vy=np.mean(np.sin(beta))

beta_angle=np.arctan2(vy,vx)%(2*np.pi)

data=plt.hist(beta)

fig=F4.polar_hist(data,beta_angle)


#statistical test (not significantly different from 180 degrees)
#s,p=bootstrap_beta(beta,np.pi,True,1000)
#p
    
#------------------------------------------------------------------------------
#Figure 3.k - Distribution of alpha
#------------------------------------------------------------------------------

all_alpha_p1_deg = np.array(all_alpha_p1)*180/np.pi
all_beta_p1_deg = np.array(all_beta_p1)*180/np.pi

fig03, ax03 = plt.subplots(figsize=(2.8*cm,2.8*cm))

ax03.hist(all_alpha_p1_deg, int(len(all_alpha_p1_deg)/10))
ax03.axvline(np.mean(all_alpha_p1_deg),color ='red', lw = 1, alpha = 0.75, label = rf'$\langle\alpha\rangle$ = {np.mean(all_alpha_p1_deg):.1f} $\degree $')

ax03.set_xlim([0,180])

x = ['0', '90', '180']
lx = [0, 90, 180]

ax03.set_xticks(lx)
ax03.set_xticklabels(x)

ax03.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax03.set_ylabel(r'Count', fontsize =7, labelpad = 3, fontname=fname)
ax03.set_xlabel(r'$\alpha$', fontsize =7, labelpad = 2,  fontname=fname)
ax03.legend(prop=font)

#------------------------------------------------------------------------------
#Figure 3.l - Correlation of omega and r12x(p1-p2)
#------------------------------------------------------------------------------

fig05, ax05 = plt.subplots(figsize=(4.5*cm,3.28*cm))

ax05.hist(all_omega_corr_p1_diff_avg, int(len(all_omega_corr_p1_diff_avg)/10))
ax05.set_xlim([-1,1])
ax05.axvline(np.mean(all_omega_corr_p1_diff_avg),color ='red', lw = 1, alpha = 0.75, label = r'$\langle\vec{\omega}\cdot \frac{\vec{r}_{12} \times \vec{p}_1 + \vec{r}_{21} \times \vec{p}_2 }{2} \rangle$')
x = ['-1.0', '0.0', '1.0']
lx = [-1, 0.0, 1]

ax05.set_xticks(lx)
ax05.set_xticklabels(x)

ax05.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax05.set_ylabel(r'Count', fontsize =7, labelpad = 2, fontname=fname)
ax05.set_xlabel(r'$\vec{\omega}\cdot \frac{\vec{r}_{12} \times \vec{p}_1 + \vec{r}_{21} \times \vec{p}_2 }{2}$', fontsize =7, labelpad = 3,  fontname=fname)
ax05.legend(prop=font)


np.mean(all_omega_corr_p1_diff_avg)

#s,p=F.bootstrap(all_omega_corr_p1_diff_avg,True,100000)
#p

#------------------------------------------------------------------------------
#Figure 3.m Correlation of Y12 and p1-p2
#------------------------------------------------------------------------------

fig07, ax07 = plt.subplots(figsize=(4.5*cm,3.28*cm))

ax07.hist(all_p1_dot_ydir, int(len(all_p1_dot_ydir)/10))
ax07.set_xlim([-1,1])
ax07.axvline(np.mean(all_p1_dot_ydir),color ='red', lw = 1, alpha = 0.75, label = r'$\vec{Y}\cdot \frac{\vec{p}_1 - \vec{p}_2}{2}$')
x = ['-1.0', '0.0', '1.0']
lx = [-1, 0.0, 1]

ax07.set_xticks(lx)
ax07.set_xticklabels(x)

ax07.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax07.set_ylabel(r'Count', fontsize =7, labelpad = 2, fontname=fname)
ax07.set_xlabel(r'Polarity - Yin Yang direction', fontsize =7, labelpad = 3,  fontname=fname)
ax07.legend(prop=font)


np.mean(all_p1_dot_ydir)

#s,p=F.bootstrap(all_p1_dot_ydir,False,100000)
#p

#------------------------------------------------------------------------------
#Figure 3.n Map of myosin signal around Omega and r12
#------------------------------------------------------------------------------

final_map_r12_omega = F4.compute_avg_cell_maps(F4.compute_avg_time_maps(all_r12_omega_maps))
Nx, Ny = final_map_r12_omega.shape

final_interface_map_r12_omega = np.mean(avg_r12_omega_interface_maps, axis = 0)

level = np.argmax(final_interface_map_r12_omega<=0.5, axis = 0)

fig08, ax08 = plt.subplots(figsize=(7.0*cm,7.8*cm))

ar = 1.0
ax08.imshow(final_map_r12_omega, interpolation = 'nearest', cmap=plt.cm.gist_gray)#, extent = [0, Ny-1, 0, Nx-1], aspect = ar)
ax08.plot(np.arange(Ny), level*ar, '-r', lw = 1)
#ax08.plot(phi_p, theta_p/ar, '+r', markersize = 10)

x = [r'-$\pi$', '0', '0', '0', r'$\pi$']
lx = [0 , Ny/4, Ny/2, Ny*3/4, Ny-1]

ax08.set_xticks(lx)
ax08.set_xticklabels(x)

y = ['0', r'$\frac{\pi}{2}$', r'$\pi$']
ly = [0, Nx/2, Nx-1]
    
ax08.set_yticks(ly)
ax08.set_yticklabels(y)

ax08.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax08.set_ylabel(r'$\theta$', fontsize =7, labelpad = 2, fontname=fname)
ax08.set_xlabel(r'$\phi$', fontsize =7, labelpad = 3,  fontname=fname)


np.min(final_map_r12_omega)
np.max(final_map_r12_omega)

#------------------------------------------------------------------------------
#Figure 4.b Map of myosin signal around p1 and r12
#------------------------------------------------------------------------------

final_map_p1 = F4.compute_avg_cell_maps(F4.compute_avg_time_maps(all_polar_maps))

Nx, Ny = final_map_p1.shape

fig09, ax09 = plt.subplots(figsize=(6.0*cm,4*cm))

ax09.imshow(final_map_p1, interpolation = 'nearest', cmap=plt.cm.gist_gray)

x = [r'-$\pi$', '0', r'$\pi$']
lx = [0, Ny/2, Ny-1]

ax09.set_xticks(lx)
ax09.set_xticklabels(x)

y = ['0', r'$\pi/2$', r'$\pi$']
ly = [0, Nx/2, Nx-1]
    
ax09.set_yticks(ly)
ax09.set_yticklabels(y)

ax09.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax09.set_ylabel(r'$\theta$', fontsize =7, labelpad = 2, fontname=fname)
ax09.set_xlabel(r'$\phi$', fontsize =7, labelpad = 3,  fontname=fname)
# ax08.legend(prop=font)


np.min(final_map_p1)
np.max(final_map_p1)

#------------------------------------------------------------------------------
#We can also make maps for individual time points (or time averaged maps of 
#individual doublets), as shown on Extended Figure 7.
#
#We use the following doublets and time points:
#    - Panel a: Doublet 3, time 9
#    - Panel b: Doublet 2, time 8
#    - Panel d: Doublet 8, time 1
#    - Panel e: Doublet 4, time 0
#
#------------------------------------------------------------------------------

#Maps of individual time points (select below the doublet and the time point)
doublet=4
time_point=0

#first normalise each cell map individually
c1_map,=F4.normalize_theta_phi_signal([all_polar_maps[2*doublet][time_point]])
c2_map,=F4.normalize_theta_phi_signal([all_polar_maps[2*doublet+1][time_point]])

#determine min and max of intensity among both maps
Imin=min([np.min(c1_map),np.min(c2_map)])
Imax=max([np.max(c1_map),np.max(c2_map)])
print(f'min: {Imin}, max: {Imax}')


fig09, ax09 = plt.subplots(figsize=(4*cm,4*cm))

ax09.imshow(c1_map, interpolation = 'nearest', cmap=plt.cm.gist_gray, aspect=1.5, vmin=Imin, vmax=Imax)

x = [r'-$\pi$', '0', r'$\pi$']
lx = [0, Ny/2, Ny-1]

ax09.set_xticks(lx)
ax09.set_xticklabels(x)
y = ['0', r'$\pi/2$', r'$\pi$']
ly = [0, Nx/2, Nx-1]
ax09.set_yticks(ly)
ax09.set_yticklabels(y)
ax09.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax09.set_ylabel(r'$\theta$', fontsize =7, labelpad = 2, fontname=fname)
ax09.set_xlabel(r'$\phi$', fontsize =7, labelpad = 3,  fontname=fname)

fig09.savefig('map_individual_time_cell0.pdf')

fig09, ax09 = plt.subplots(figsize=(4*cm,4*cm))

ax09.imshow(c2_map, interpolation = 'nearest', cmap=plt.cm.gist_gray, aspect=1.5, vmin=Imin, vmax=Imax)

x = [r'-$\pi$', '0', r'$\pi$']
lx = [0, Ny/2, Ny-1]

ax09.set_xticks(lx)
ax09.set_xticklabels(x)
y = ['0', r'$\pi/2$', r'$\pi$']
ly = [0, Nx/2, Nx-1]
ax09.set_yticks(ly)
ax09.set_yticklabels(y)
ax09.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax09.set_ylabel(r'$\theta$', fontsize =7, labelpad = 2, fontname=fname)
ax09.set_xlabel(r'$\phi$', fontsize =7, labelpad = 3,  fontname=fname)

fig09.savefig('map_individual_time_cell1.pdf')

#Here we generate time averaged maps for a specific cell
doublet=4
time_normalised_1,=F4.compute_avg_time_single_map(all_polar_maps[2*doublet])
time_normalised_2,=F4.compute_avg_time_single_map(all_polar_maps[2*doublet+1])

Imin=min([np.min(time_normalised_1),np.min(time_normalised_2)])
Imax=max([np.max(time_normalised_1),np.max(time_normalised_2)])
print(f'min: {Imin}, max: {Imax}')

fig09, ax09 = plt.subplots(figsize=(4*cm,4*cm))
ax09.imshow(time_normalised_1, interpolation = 'nearest', cmap=plt.cm.gist_gray, aspect=1.5, vmin=Imin, vmax=Imax)
x = [r'-$\pi$', '0', r'$\pi$']
lx = [0, Ny/2, Ny-1]
ax09.set_xticks(lx)
ax09.set_xticklabels(x)
y = ['0', r'$\pi/2$', r'$\pi$']
ly = [0, Nx/2, Nx-1]
ax09.set_yticks(ly)
ax09.set_yticklabels(y)
ax09.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax09.set_ylabel(r'$\theta$', fontsize =7, labelpad = 2, fontname=fname)
ax09.set_xlabel(r'$\phi$', fontsize =7, labelpad = 3,  fontname=fname)

fig09.savefig('map_avgtime_cell0.pdf')

fig09, ax09 = plt.subplots(figsize=(4*cm,4*cm))
ax09.imshow(time_normalised_2, interpolation = 'nearest', cmap=plt.cm.gist_gray, aspect=1.5, vmin=Imin, vmax=Imax)
x = [r'-$\pi$', '0', r'$\pi$']
lx = [0, Ny/2, Ny-1]
ax09.set_xticks(lx)
ax09.set_xticklabels(x)
y = ['0', r'$\pi/2$', r'$\pi$']
ly = [0, Nx/2, Nx-1]
ax09.set_yticks(ly)
ax09.set_yticklabels(y)
ax09.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax09.set_ylabel(r'$\theta$', fontsize =7, labelpad = 2, fontname=fname)
ax09.set_xlabel(r'$\phi$', fontsize =7, labelpad = 3,  fontname=fname)

fig09.savefig('map_avgtime_cell1.pdf')


#------------------------------------------------------------------------------
#Figure 4.c Fit of myosin profiles around the polarity vector as a function of 
#theta (Then Extended Figure 8b, histogram of fit parameters)
#------------------------------------------------------------------------------

Ntheta=80
Nphi=160


avg_time_maps_norm_p1 = F4.compute_avg_time_maps(all_polar_maps)
final_map_p1 = F4.compute_avg_cell_maps(F4.compute_avg_time_maps(all_polar_maps))

avg_time_avg_along_phi = F4.mean_along_phi(avg_time_maps_norm_p1)

fig11, ax11 = plt.subplots(figsize = (5.5*cm, 3.5*cm))

theta = np.linspace(np.pi/(2*Ntheta) , np.pi-np.pi/(2*Ntheta), Ntheta)
tension_mods = []
bs = []
i = 0
for avg in avg_time_avg_along_phi:
    
    tension_mod, b, int_signal = F4.get_tension_mod_fit(Ntheta, Nphi, avg_time_maps_norm_p1[i])
    tension_mods.append(tension_mod)
    bs.append(b)
    ax11.plot(theta, avg, '.k', alpha = 0.2, markersize = 1)
    i+=1

tension_mod_mean, b_mean, int_signal_mean = F4.get_tension_mod_fit(Ntheta, Nphi, final_map_p1)
ax11.plot(theta, np.mean(avg_time_avg_along_phi,axis=0), '.b', markersize = 1) #, label = r'$\langle I_{myosin} \rangle$ for 12 doublets')
ax11.plot(theta, F4.func_fit(theta, tension_mod_mean, b_mean) , '-r', lw = 0.5) #, label = r'Fitted $I_{myosin}$')

x = ['0', r'$\pi/2$', r'$\pi$'] 
lx = [0, np.pi/2, np.pi]

ax11.set_xticks(lx)
ax11.set_xticklabels(x)

ly = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [str(e) for e in ly] 

ax11.set_yticks(ly)
ax11.set_yticklabels(y)

ax11.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax11.set_ylabel(r'Normalized $I_{myosin}$', fontsize =7, labelpad = 2, fontname=fname)
ax11.set_xlabel(r'$\theta$', fontsize =7, labelpad = 2,  fontname=fname)
ax11.set_ylim([0,4])


#Extended Figure 8b, histograms of tension_mod and b
#For b, one outlier point at b=9.22 is not shown
fig, ax = plt.subplots(figsize = (6*cm, 3*cm))
ax.hist(tension_mods)
ax.set_xlim([0,2.3])
ax.set_ylim([0,5])
ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax.set_ylabel('Count', fontsize =7, labelpad = 2, fontname=fname)
ax.set_xlabel(r'$\Delta I/\langle I \rangle$', fontsize =7, labelpad = 2,  fontname=fname)

fig, ax = plt.subplots(figsize = (6*cm, 3*cm))
ax.hist(bs,bins=18)
ax.set_xlim([0,5])
ax.set_ylim([0,16])
ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax.set_ylabel('Count', fontsize =7, labelpad = 2, fontname=fname)
ax.set_xlabel('b', fontsize =7, labelpad = 2,  fontname=fname)




#------------------------------------------------------------------------------
#Figure 4.h (Experimental part) - Rotation rate Omega as a function of the 
# myosin signal modulation (standard deviation over average)
#------------------------------------------------------------------------------

all_sigma_c1=np.array(all_sigma_c1)
all_sigma_c2=np.array(all_sigma_c2)


#Look for outlier data points with abnormally large values of sigma/I
#This can happen when the myosin signal is very weak for a given time point
ratio=np.sum(all_sigma_c1+all_sigma_c2<=5)/all_sigma_c1.shape[0]

print(f'{ratio*100}% of points lie in the 0 to 5 range for sigma/I')
ind_to_del=np.where(all_sigma_c1+all_sigma_c2>5)
print(f'Points with sigma/I>5 are considered outliers, {ind_to_del[0].shape[0]} points removed')

sum_sigma_del=np.delete(all_sigma_c1+all_sigma_c2,ind_to_del)
ang_velocity_del=np.delete(ang_velocity,ind_to_del)

#number of samples in boostrapping must be adjusted here to obtain a precise p-value
fig, ax = F4.plot_correlation(sum_sigma_del, ang_velocity_del, r'$\sigma_1/I_1 + \sigma_2/I_2$', r'$\omega$', 2000)
ax.set_xlim([0,4])
ax.set_ylim([0,1])
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])


#------------------------------------------------------------------------------
#Figure 4.i (Experimental part) - Interface deflection as a function of the 
# myosin signal modulation (standard deviation over average)
#------------------------------------------------------------------------------

#we remove the same outlier points as in the previous figure
yin_yang_height_del=np.delete(yin_yang_height,ind_to_del)

#number of samples in boostrapping must be adjusted here to obtain a precise p-value
fig, ax = F4.plot_correlation(sum_sigma_del, yin_yang_height_del, r'$\sigma_1/I_1 + \sigma_2/I_2$', r'$\sqrt{\langle H_{y-y}^2 \rangle}/R$', 1000)
ax.set_xlim([0,4])
ax.set_ylim([0,0.12])
ax.set_yticks([0,0.05,0.1])

#------------------------------------------------------------------------------
#Figure 4.l (Experimental part) - correlation of the orientation of the bowl 
#deformation mode with the difference of myosin intensity between the cells
#------------------------------------------------------------------------------

#extract and angle from sQpn times sgn delta I
all_int_signal_c1 = np.array(all_int_signal_c1)
all_int_signal_c2 = np.array(all_int_signal_c2)

delta_int = (all_int_signal_c1-all_int_signal_c2)/(all_int_signal_c1+all_int_signal_c2)

delta_int_sym = np.hstack((delta_int, -delta_int))

#look for outlier points where the intensity difference is abnormally large
#This can happen in problematic cases when the average myosin intensity is too 
#small on a given cell
ratio=np.sum(delta_int<-0.9)/delta_int.shape[0]
print(f'{ratio*100}% of points have a normalised intensity difference larger than -0.9')
print(f'It concerns the points {np.where(delta_int<-0.9)} which are considered to be outliers.')

bowl_amplitude_sym = np.hstack((np.array(bowl_amplitude), -np.array(bowl_amplitude)))

delta_int_del=np.delete(delta_int,[91,93,94])
bowl_amplitude_del=np.delete(bowl_amplitude,[91,93,94])
bowl_amplitude_sym_del = np.hstack((np.array(bowl_amplitude_del), -np.array(bowl_amplitude_del)))
delta_int_sym_del = np.hstack((delta_int_del, -delta_int_del))

#Adjust the number of samples to at least 100 000 to obtain an accurate p-value
fig, ax = F4.plot_correlation(delta_int_sym_del, bowl_amplitude_sym_del, r'$(\langle I_1\rangle - \langle I_2 \rangle)/ (\langle I_1\rangle + \langle I_2 \rangle)$', r'$B_{12}$', 500)
ax.axvline(0, color = 'k', linewidth = 0.5, alpha = 0.6)
ax.axhline(0, color = 'k', linewidth = 0.5, alpha = 0.6)
F4.set_xlim_ylim(ax, [-0.4, 0.4], [-0.2, 0.2])
F4.set_xticks_yticks(ax,[-0.4, -0.2, 0.0, 0.2, 0.4], [-0.2, -0.1, 0.0, 0.1, 0.2])

#------------------------------------------------------------------------------
#Figure 4.m - correlation of the orientation of the saddle-node deformation 
#mode with the nematic part of the myosin distribution
#------------------------------------------------------------------------------

Qsn=np.array(Qsn)
dQn=np.array(dQn)

dQn_Qsn=np.sum(np.sum(dQn*Qsn,axis=2),axis=1)
fig, ax = plt.subplots(figsize = (4.52*cm, 3.225*cm))
ax.hist(dQn_Qsn,bins=18)
ax.axvline(np.mean(dQn_Qsn),color='red',lw = 1, alpha = 0.75)
ax.set_xlabel(r'$(Q_1-Q_2)/|Q_1-Q_2|\cdot Q_S/|Q_S|$', fontsize =7, labelpad = 2, fontname=fname)
ax.set_ylabel('Count', fontsize =7, labelpad = 2, fontname=fname)
F4.set_xticks_yticks(ax,[-1.0,0.0,1.0],[0,5,10,15])
ax.tick_params(axis='both', which='major', labelsize=7, pad=2)

#statistical test
#s,p=F.bootstrap(dQn_Qsn,False,50000)
#p



