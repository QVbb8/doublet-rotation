#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:45:13 2024

@author: vagne
"""

#------------------------------------------------------------------------------
#After the files 'all_pz_single.txt' and 'all_pz_single_blur.txt' have been
#created using 'calibration_z_2min_single_cell.py', this script can be run to 
#generate the maps of myosin aligned around z in Supplementary Figure 1c, 1d
#
#First we analyse the single cells data and register myosin maps and polarity
#information for individual cells in different files
#
#Then, we load the generated data and aggregate all the myosin  maps together
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#Choose here whether to generate the map of Supplementary Figure 1c (not 
#blurred, False) or Supplementary Figure 1d (blurred, True)
#------------------------------------------------------------------------------
useblur=False


#------------------------------------------------------------------------------
#Import necessary packages and functions
#------------------------------------------------------------------------------
import os
import pickle
os.chdir('../Figure3_Figure4')
import useful_functions_interface as uf_int
import useful_functions_ply_files as uf
import function_final_interface as ff_int
os.chdir('../Supp_Fig1_single_cells')

import functions as func

import matplotlib.font_manager as font_manager
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cmap
from pyevtk.vtk import VtkTriangle, VtkQuad
from pyevtk.hl import unstructuredGridToVTK
import pyvista


def show_omega(Omegavector,time):
    
    pyvista.global_theme.font.color = 'black'
    ptc=pyvista.PolyData(Omegavector)
    pl=pyvista.Plotter(window_size=[1600,1600]) #plotter creation and initialize a movie
    #pl.open_movie('test.mp4')
    #actor=pl.add_mesh(ptc,point_size=10.0,render_points_as_spheres=True,scalars=np.linspace(0,1,len(Omegavector)))
    #pl.add_bounding_box()
    #norm=np.max(np.sqrt(np.sum(Omegavector**2,axis=1)))
    #arx=pyvista.Arrow(start=(0,0,0),direction=(1,0,0),scale=norm/10)
    #ary=pyvista.Arrow(start=(0,0,0),direction=(0,1,0),scale=norm/10)
    #arz=pyvista.Arrow(start=(0,0,0),direction=(0,0,1),scale=norm/10)
    
    #line going from point to point
    npoints=len(Omegavector)
   
    # Display the points
    pl.add_mesh(ptc,point_size=10.0,render_points_as_spheres=True,scalars=time,show_scalar_bar=False)
    #display lines
    for i in range(npoints-1):
        a0=[Omegavector[i,0],Omegavector[i,1],Omegavector[i,2]]
        a1=[Omegavector[i+1,0],Omegavector[i+1,1],Omegavector[i+1,2]]
        pl.add_lines(np.array([a0,a1]),color=cmap.viridis(i/(npoints-1)))
    #display a sphere
    Mnorm=np.max(np.sqrt(np.sum(Omegavector**2,axis=1)))
    sph=pyvista.Sphere(radius=Mnorm,center=(0,0,0))
    
    pl.add_mesh(sph,opacity=0.2)
    
    pl.set_background('white')
    pl.show_bounds(location='outer',font_size=10000000,grid=True)

    pl.show()
    
def save_mesh_vtu(xyz, T, data_fields, data_fields_name, savepath):
    
    x = np.copy(xyz[:,0])
    y = np.copy(xyz[:,1])
    z = np.copy(xyz[:,2])

    data_dict = dict([(data_fields_name[i], data_fields[i]) for i in range(len(data_fields))])    
    
    # Define connectivity or vertices that belongs to each element
    conn = T.flatten()

    offset = np.zeros(len(T))
    for i in range(len(offset)):
        offset[i] = 3*(i+1)
    # Define cell types

    ctype = np.zeros(len(offset))
    for i in range(len(ctype)):
        ctype[i] = VtkTriangle.tid
    
    cellData = dict([(data_fields_name[i], np.mean(data_fields[i][T],axis=1)) for i in range(len(data_fields))])   
    
    pointData = dict([(data_fields_name[i], data_fields[i]) for i in range(len(data_fields))])   
    
    unstructuredGridToVTK(savepath, x, y, z, connectivity = conn, offsets = offset, cell_types = ctype, cellData = cellData, pointData = pointData)#, comments = comments)

    return()


#------------------------------------------------------------------------------
#Loop on all single cells, analyse and save the data
#This will fill the folder 'single_cell_maps' or 'single_cell_maps_blur' with 
#individual myosin maps in the form of pickle files
#It will also save polarity information in 'data_single_nonblur.pickle' or 
#'data_single_blur.pickle'
#
#data_single_blur.pickle is used for the Extended Figure 1c to look at the
#temporal dynamics of polarity in single cells vs doublets. 
#
#------------------------------------------------------------------------------
name_path='./data_single_cells/Segmentation_'
Ncells=12
all_paths,all_img,name,scale_factors = func.generate_paths(name_path,Ncells,1/0.103,useblur)


#load z calibration data
if useblur==True:
    all_pz=np.loadtxt('all_pz_single_blur.txt')
else:
    all_pz=np.loadtxt('all_pz_single.txt')

Ntheta = 80
Nphi = 160
Polarities=[]
Polarities_random=[]
inter_avg_ratio=[]
max_avg_ratio=[]

for k,path in enumerate(all_paths):
    
    print(path)
    
    img_name = all_img[k]
    scale_factor = scale_factors[k]
    
    xy_pix = 5
    z_pix = 1
    
    
    img = imread(path+img_name)
    
    PATHS = [path + 'Cell_1']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    
    com_cell = func.get_series_com_single(PATHS)
        
    all_p_ex_maps = []
    all_p_ey_maps = []
    all_p_ez_maps = []
    
    all_z_maps=[]
    
    Pol_ind=np.zeros((endpoint+1-startpoint,3))
    
    i=0
    
    for t in range(startpoint, endpoint+1):
        print(f'Frame {t}/{endpoint}')
        mesh1, V1, T1 = uf.get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')

        im_mrlc = img[t-1,:,:,:]
        
        
        #rescale the myosin intensity using an exponential curve to compensate for the effect of a distance to the microscope.
        im_mrlc=func.exponential_calibration(im_mrlc,scale_factor,-1/all_pz[k])
        
        
        avg_inside_cell1 = func.get_signal_inside_cell(V1, im_mrlc, scale_factor)
                
        mrlc_values_cell1,mean_b,max_b = func.get_signal_vertices_single(V1, im_mrlc, xy_pix, z_pix, scale_factor, avg_inside_cell1)
        
        inter_avg_ratio.append(mean_b/avg_inside_cell1)
        max_avg_ratio.append(max_b/avg_inside_cell1)
        
        p1_c1,lol = func.polarity4(V1, T1, mrlc_values_cell1)
        Pol_ind[t-1,:]=p1_c1
        
        
        #First option with maximum of intensity
        theta_ex_c1, phi_ex_c1 = func.get_angular_distribution_p(V1, T1, mrlc_values_cell1,p1_c1,np.array([1,0,0]))
        theta_ey_c1, phi_ey_c1 = func.get_angular_distribution_p(V1, T1, mrlc_values_cell1,p1_c1,np.array([0,1,0]))
        theta_ez_c1, phi_ez_c1 = func.get_angular_distribution_p(V1, T1, mrlc_values_cell1,p1_c1,np.array([0,0,1]))
        
        #distribution around z for the nematic bias
        theta_z, phi_z = func.get_angular_distribution_z(V1,T1)
        
        theta_phi_ex_c1 = np.zeros((Ntheta, Nphi))
        theta_phi_ey_c1 = np.zeros((Ntheta, Nphi)) 
        theta_phi_ez_c1 = np.zeros((Ntheta, Nphi))
        theta_phi_z = np.zeros((Ntheta, Nphi))

        func.get_angular_maps_interpolation(theta_ex_c1, phi_ex_c1, mrlc_values_cell1, theta_phi_ex_c1)
        func.get_angular_maps_interpolation(theta_ey_c1, phi_ey_c1, mrlc_values_cell1, theta_phi_ey_c1)
        func.get_angular_maps_interpolation(theta_ez_c1, phi_ez_c1, mrlc_values_cell1, theta_phi_ez_c1)
        func.get_angular_maps_interpolation(theta_z, phi_z, mrlc_values_cell1, theta_phi_z)
        
        all_p_ex_maps.append(theta_phi_ex_c1)
        all_p_ey_maps.append(theta_phi_ey_c1)
        all_p_ez_maps.append(theta_phi_ez_c1)
        
        all_z_maps.append(theta_phi_z)
        
        i+=1
        
    Polarities.append(Pol_ind)
        

    if useblur==True:
        prefix='./single_cell_maps_blur/'
    else:
        prefix='./single_cell_maps/'

    with open(prefix+f'maps_pure_z_{name[k]}.pickle','wb') as f:
        pickle.dump(all_z_maps,f)

if useblur==False:
    with open(f'data_single_nonblur.pickle','wb') as f:
        list_data=[inter_avg_ratio,max_avg_ratio,Polarities]
        pickle.dump(list_data,f)
else:
    with open(f'data_single_blur.pickle','wb') as f:
        list_data=[inter_avg_ratio,max_avg_ratio,Polarities]
        pickle.dump(list_data,f)
        
#------------------------------------------------------------------------------
#In this part we load all the individual maps and combine them to create 
#Supplementary Figure 1.c (non blurred) or 1.d (blurred)
#
#A few cells are excluded in the blured case because the myosin signal on the
#membrane is not strong enough compared to the signal inside the cell, leading
#to negative average signal on the membrane
#------------------------------------------------------------------------------
cm = 1/2.54
fname = 'Arial'
font = font_manager.FontProperties(family=fname,
                                    weight='normal',
                                    style='normal', size=5)

if useblur==True:
    setoffiles = ['./single_cell_maps_blur/maps_pure_z_'+str(i)+'.pickle' for i in [1,2,3,4,5,6,8,9,11,12]]
else:
    setoffiles = ['./single_cell_maps/maps_pure_z_'+str(i)+'.pickle' for i in range(1,Ncells+1)]

#setoffiles = ['./single_cell_maps_blur/maps_pure_z_'+str(i)+'.pickle' for i in range(1,Ncells+1)]


avg_time_avg_along_phi=[]
avg_time_maps_norm_p1=[]
for fil in setoffiles:

    with open(fil,'rb') as f:
        maps=pickle.load(f)
        
        
    avg_map=np.mean(maps,axis=0)
    avg_map_norm=func.normalize_theta_phi_signal(avg_map)
    
    vmc1=np.min(avg_map_norm)
    vMc1=np.max(avg_map_norm)
    #plt.figure()
    #plt.title(fil)
    #im=plt.imshow(avg_map_norm,vmin=vmc1,vmax=vMc1)
    #plt.colorbar(im)
    
    avg_phi=np.mean(avg_map_norm,axis=1)
    
    avg_time_avg_along_phi.append(avg_phi)
    avg_time_maps_norm_p1.append(avg_map_norm)
    
avg_phi_mean_over_cells= np.mean(avg_time_avg_along_phi,axis=0)


final_map_p1=np.mean(avg_time_maps_norm_p1,axis=0)

Nx,Ny = final_map_p1.shape

np.min(final_map_p1)
np.max(final_map_p1)

fig,ax = plt.subplots(figsize=(7.0*cm,7.8*cm))
im=ax.imshow(final_map_p1,cmap=plt.cm.gist_gray)
#fig.colorbar(im)

lx = [0 , Ny/4, Ny/2, Ny*3/4, Ny-1]
ax.set_xticks(lx)
ly = [0, Nx/2, Nx-1]
ax.set_yticks(ly)
ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
