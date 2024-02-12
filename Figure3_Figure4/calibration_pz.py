#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:08:11 2023

@author: vagne
"""


#This generates the file 'all_pz_blur.txt' that contains the corrections to 
#apply to the myosin signal for each doublet

#It also generates the following plots:
# - Extended Figure 5.d : Profile of myosin along x,y,z for one doublet.
# - Extended Figure 5.e : Myosin signal decrease for all doublets, along x,y,z.

from skimage.io import imread
import matplotlib.font_manager as font_manager
import functions_doublet as F
import numpy as np
import os
import useful_functions_interface as uf_int
import useful_functions_ply_files as uf
import function_final_interface as ff_int
from scipy.spatial import Delaunay
from numba import jit
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

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
def get_signal_vertices(vertices, img, n_pix_xy, n_pix_z, scale_factor, avg_signal_cell, avg_signal_other_cell, interface_v):
    
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
        if interface_v[k] == 0:
            diff = value_signal - avg_signal_cell
            values[k] = diff
        else :
            diff = (value_signal - (avg_signal_cell+avg_signal_other_cell)/2)/2
            values[k] = diff
           
    return(values)
    

cm = 1/2.54

fname = 'Arial'
font = font_manager.FontProperties(family=fname,
                                    weight='normal',
                                    style='normal', size=5)


##############################################################################
########################### all files ########################################
##############################################################################

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

#FIRST PART: COMPUTE Zmin and Zmax, also xmin,xmax,ymin,ymax
#----------------------------------------------------------------------------
all_zM=[]
all_zm= []

all_xM=[]
all_xm= []

all_yM=[]
all_ym= []

#loop on doublets (individual calibration for each doublet)
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
    
    #now loop on time points of the doublet to accumulate data
    i = 0
    #we don't skip the last time point
    for t in range(startpoint, endpoint+1): 
        print('Timepoint =', t)
        mesh1, V1, T1 = uf.get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')
        mesh2, V2, T2 = uf.get_vertices_triangles(PATHS[1]+'/time'+str(t)+'_cell_2.ply')
        im_mrlc = img[t-1,:,0,:,:] ### ECAD is 1
        
        dist1, dist2 = ff_int.compute_distance_between_two_clouds(V1, V2)
        dist_threshold = int(np.floor(scale_factor) + 1) 
        outer_shell_cloud = np.vstack((V1[dist1>=dist_threshold],V2[dist2>=dist_threshold]))
        
        #zxys=get_pixels_inside(V1,V2,im_mrlc, scale_factor)
        
        
        #avg_inside_cell1 = get_signal_inside_cell(V1, im_mrlc, scale_factor)
        #avg_inside_cell2 = get_signal_inside_cell(V2,im_mrlc, scale_factor)
                
        #mrlc_values_cell1 = get_signal_vertices(V1, im_mrlc, xy_pix, z_pix, scale_factor, avg_inside_cell1, avg_inside_cell2, interface_V1)
        #mrlc_values_cell2 = get_signal_vertices(V2, im_mrlc, xy_pix, z_pix, scale_factor, avg_inside_cell2, avg_inside_cell1, interface_V2)
        
        #we get the signal only on the interface cloud
        #mrlc_values_cloud=np.concatenate((mrlc_values_cell1[dist1>=dist_threshold],mrlc_values_cell2[dist2>=dist_threshold]))
        
        #F.show_doublet_outer(outer_shell_cloud,mrlc_values_cloud)
        
        #first we only accumulate the zmin,zmax of the doublet for all times
        if t==startpoint:
            zm=np.min(outer_shell_cloud[:,2])
            zM=np.max(outer_shell_cloud[:,2])
        else:
            if np.min(outer_shell_cloud[:,2]) < zm:
                zm=np.min(outer_shell_cloud[:,2])
            if np.max(outer_shell_cloud[:,2]) > zM:
                zM=np.max(outer_shell_cloud[:,2])
                
        if t==startpoint:
            xm=np.min(outer_shell_cloud[:,0])
            xM=np.max(outer_shell_cloud[:,0])
        else:
            if np.min(outer_shell_cloud[:,0]) < xm:
                xm=np.min(outer_shell_cloud[:,0])
            if np.max(outer_shell_cloud[:,0]) > xM:
                xM=np.max(outer_shell_cloud[:,0])
                
        if t==startpoint:
            ym=np.min(outer_shell_cloud[:,1])
            yM=np.max(outer_shell_cloud[:,1])
        else:
            if np.min(outer_shell_cloud[:,1]) < ym:
                ym=np.min(outer_shell_cloud[:,1])
            if np.max(outer_shell_cloud[:,1]) > yM:
                yM=np.max(outer_shell_cloud[:,1])
        i+=1
    all_zm.append(zm)
    all_zM.append(zM)
    all_xm.append(xm)
    all_xM.append(xM)
    all_ym.append(ym)
    all_yM.append(yM)
    
#SECOND PART: histograms (requires all_zm,all_zM,all_xm,all_xM,all_ym,all_yM
#----------------------------------------------------------------------------


all_px=[]
all_py=[]
all_pz=[]

all_px_um=[]
all_py_um=[]
all_pz_um=[]

#loop on doublets (individual calibration for each doublet)
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
    
    nbin=20
    signal_binned_z=np.zeros((endpoint-startpoint+1,nbin))
    signal_timeavg_z=np.zeros(nbin)
    zbins=np.linspace(all_zm[k],all_zM[k],nbin+1)
    zmid=(zbins[1:]+zbins[:-1])/2
    
    signal_binned_x=np.zeros((endpoint-startpoint+1,nbin))
    signal_timeavg_x=np.zeros(nbin)
    xbins=np.linspace(all_xm[k],all_xM[k],nbin+1)
    xmid=(xbins[1:]+xbins[:-1])/2
    
    signal_binned_y=np.zeros((endpoint-startpoint+1,nbin))
    signal_timeavg_y=np.zeros(nbin)
    ybins=np.linspace(all_ym[k],all_yM[k],nbin+1)
    ymid=(ybins[1:]+ybins[:-1])/2
    
    #now loop on time points of the doublet to accumulate data
    i = 0
    #we don't skip the last time point
    for t in range(startpoint, endpoint+1): 
        print('Timepoint =', t)
        
        mesh1, V1, T1 = uf.get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')
        
        mesh2, V2, T2 = uf.get_vertices_triangles(PATHS[1]+'/time'+str(t)+'_cell_2.ply')
        im_mrlc = img[t-1,:,0,:,:] ### ECAD is 1
        
        dist1, dist2 = ff_int.compute_distance_between_two_clouds(V1, V2)
        dist_threshold = int(np.floor(scale_factor) + 1) 
        outer_shell_cloud = np.vstack((V1[dist1>=dist_threshold],V2[dist2>=dist_threshold]))
        
        interface_V1 = np.zeros(len(V1))
        interface_V2 = np.zeros(len(V2))
        
        interface_V1[dist1<dist_threshold] = 1
        interface_V2[dist2<dist_threshold] = 1
        interface_V1 = interface_V1.astype(np.uint)
        interface_V2 = interface_V2.astype(np.uint)
        
        #avg_inside_cell1 = get_signal_inside_cell(V1, im_mrlc, scale_factor)
        #avg_inside_cell2 = get_signal_inside_cell(V2,im_mrlc, scale_factor)
        
        #here we DONT substract the average inside the cell because the raw signal is the one to work with.
        mrlc_values_cell1 = get_signal_vertices(V1, im_mrlc, xy_pix, z_pix, scale_factor, 0, 0, interface_V1)
        mrlc_values_cell2 = get_signal_vertices(V2, im_mrlc, xy_pix, z_pix, scale_factor, 0, 0, interface_V2)
        
        #we get the signal only on the interface cloud
        mrlc_values_cloud=np.concatenate((mrlc_values_cell1[dist1>=dist_threshold],mrlc_values_cell2[dist2>=dist_threshold]))
        
        #F.show_doublet_outer(outer_shell_cloud,mrlc_values_cloud)
        
        #put signal into bins
        for z in range(nbin):
            indices=np.logical_and(zbins[z] <= outer_shell_cloud[:,2],outer_shell_cloud[:,2] <= zbins[z+1])
            signal_binned_z[i][z] = np.mean(mrlc_values_cloud[indices])
            
        for x in range(nbin):
            indices=np.logical_and(xbins[x] <= outer_shell_cloud[:,0],outer_shell_cloud[:,0] <= xbins[x+1])
            signal_binned_x[i][x] = np.mean(mrlc_values_cloud[indices])
            
        for y in range(nbin):
            indices=np.logical_and(ybins[y] <= outer_shell_cloud[:,1],outer_shell_cloud[:,1] <= ybins[y+1])
            signal_binned_y[i][y] = np.mean(mrlc_values_cloud[indices])
        
        
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
    
    #--------------------------------------------------------------------------
    # Extended Figure 5.d : Profile of myosin along x,y,z for the first doublet.
    #--------------------------------------------------------------------------
    if k==0: 
        fig,ax=plt.subplots(3,figsize=(5*cm,9*cm))
        fig.tight_layout()
        
        ax[0].plot(xmid/scale_factor,signal_binned_x[0],'ko',alpha=0.2,markersize=1,label='individual time points')
        for i in range(1,endpoint-startpoint+1):
            ax[0].plot(xmid/scale_factor,signal_binned_x[i],'ko',alpha=0.2,markersize=1)
        ax[0].plot(xmid/scale_factor,signal_timeavg_x,'b-',label='time average')
        ax[0].plot(xmid/scale_factor,np.exp(px[1])*np.exp(px[0]*xmid),'r-',label='exponential fit')
        ax[0].set_xlim([all_xm[k]/scale_factor,all_xM[k]/scale_factor])
        ax[0].set_ylim([min_yaxis,max_yaxis])
        ax[0].set_xlabel(r'x ($\mu$m)',fontsize =7, labelpad = 3, fontname=fname)
        ax[0].set_ylabel('myosin signal',fontsize =7, labelpad = 3, fontname=fname)
        #plt.title(f'doublet {k}')
        ax[0].legend(prop=font)
        ax[0].tick_params(axis='both', which='major', labelsize=7, pad=2)
        
    
        for i in range(endpoint-startpoint+1):
            ax[1].plot(ymid/scale_factor,signal_binned_y[i],'ko',alpha=0.2,markersize=1)
        ax[1].plot(ymid/scale_factor,signal_timeavg_y,'b-')
        ax[1].plot(ymid/scale_factor,np.exp(py[1])*np.exp(py[0]*ymid),'r-')
        ax[1].set_xlim([all_ym[k]/scale_factor,all_yM[k]/scale_factor])
        ax[1].set_ylim([min_yaxis,max_yaxis])
        ax[1].set_xlabel(r'y ($\mu$m)',fontsize =7, labelpad = 3, fontname=fname)
        ax[1].set_ylabel('myosin signal',fontsize =7, labelpad = 3, fontname=fname)
        #plt.title(f'doublet {k}')
        ax[1].tick_params(axis='both', which='major', labelsize=7, pad=2)
        
        
        for i in range(endpoint-startpoint+1):
            ax[2].plot(zmid/scale_factor,signal_binned_z[i],'ko',alpha=0.2,markersize=1)
        ax[2].plot(zmid/scale_factor,signal_timeavg_z,'b-')
        ax[2].plot(zmid/scale_factor,np.exp(pz[1])*np.exp(pz[0]*zmid),'r-')
        ax[2].set_xlim([all_zm[k]/scale_factor,all_zM[k]/scale_factor])
        ax[2].set_ylim([min_yaxis,max_yaxis])
        ax[2].set_xlabel(r'z ($\mu$m)',fontsize =7, labelpad = 3, fontname=fname)
        ax[2].set_ylabel('myosin signal',fontsize =7, labelpad = 3, fontname=fname)
        #plt.title(f'doublet {k}')
        ax[2].tick_params(axis='both', which='major', labelsize=7, pad=2)
        
        
        plt.show()
    
    

#--------------------------------------------------------------------------
# Extended Figure 5.e : Lengthscale of myosin decrease along x,y,z for all doublets
#--------------------------------------------------------------------------

fig,ax=plt.subplots(3)
ax[0].hist(all_px)
ax[0].set_xlabel('slope along x')
ax[0].set_xlim([-0.004,0.004])
ax[1].hist(all_py)
ax[1].set_xlabel('slope along y')
ax[1].set_xlim([-0.004,0.004])
ax[2].hist(all_pz)
ax[2].set_xlabel('slope along z')
ax[2].set_xlim([-0.004,0.004])
ax[0].set_title('myosin')

fig,ax=plt.subplots(3,figsize=(5*cm,9*cm))
fig.tight_layout()
ax[0].hist(all_px_um,bins=np.linspace(-0.04,0.04,45))
ax[0].axvline(np.mean(all_px_um),color ='red', lw = 1, alpha = 0.75)
ax[0].tick_params(axis='both', which='major', labelsize=7, pad=2)
ax[0].set_xlabel(r'$\lambda_x$ ($\mu m^{-1}$)',fontsize =7, labelpad = 3, fontname=fname)
ax[0].set_xlim([-0.04,0.04])
ax[1].hist(all_py_um,bins=np.linspace(-0.04,0.04,45))
ax[1].axvline(np.mean(all_py_um),color ='red', lw = 1, alpha = 0.75)
ax[1].tick_params(axis='both', which='major', labelsize=7, pad=2)
ax[1].set_xlabel(r'$\lambda_y$ ($\mu m^{-1}$)',fontsize =7, labelpad = 3, fontname=fname)
ax[1].set_xlim([-0.04,0.04])
ax[2].hist(all_pz_um,bins=np.linspace(-0.04,0.04,45))
ax[2].axvline(np.mean(all_pz_um),color ='red', lw = 1, alpha = 0.75)
ax[2].set_xlabel(r'$\lambda_z$ ($\mu m^{-1}$)',fontsize =7, labelpad = 3, fontname=fname)
ax[2].set_xlim([-0.04,0.04])
ax[2].tick_params(axis='both', which='major', labelsize=7, pad=2)

#Statistical bootstrap analysis of the sign of the mean of each length scales
#along x,y and z 
#for x,y: 10 replicates with 1000 samples is enough because the p-values are 
#high (non significant)
#for z we do one test with 50 000 samples showing that p<1e-4
p_arr=[]
for i in range(10):
    s,p=F.bootstrap(all_px,True,1000)
    p_arr.append(p)
    
plt.figure()  
plt.hist(p_arr)

#extract 95% confidence interval from the 10 replicates
s,p=F.bootstrap(p_arr,True,1000)
a,b,c=plt.hist(s,bins=40,cumulative=True)
m=b[np.argmax(a>=25)]
M=b[np.argmax(a>=975)]
print(f'{(M+m)/2} p/m {(M-m)/2}')

p_arr=[]
for i in range(10):
    s,p=F.bootstrap(all_py,True,1000)
    p_arr.append(p)
    
plt.figure()  
plt.hist(p_arr)

#extract 95% confidence interval from the 10 replicates
s,p=F.bootstrap(p_arr,True,1000)
a,b,c=plt.hist(s,bins=40,cumulative=True)
m=b[np.argmax(a>=25)]
M=b[np.argmax(a>=975)]
print(f'{(M+m)/2} p/m {(M-m)/2}')


s,p=F.bootstrap(all_pz,False,50000)
p

#save txt file that will be loaded by the 'plots_Fig3_Fig4.py' script to remove the z dependency on the
#myosin signal
np.savetxt('all_pz_blur.txt',all_pz)

