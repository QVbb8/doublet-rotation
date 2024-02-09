# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 19:45:41 2022

@author: Riveline LAB
"""

import os
import numpy as np

import read_ply

import useful_functions_ply_files as uf

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
import pyntcloud

from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL

# path_import = 'E:/Rotation/Data_analysis_Figure_Article/Interface'
# os.chdir(path_import)
import useful_functions_interface as uf_int


import basal_maps as bm

# %matplotlib inline
# plt.close('all')

def cells_to_cloud_points(path, PATHS, time):
    
    "warning : treat a single time point "
    
    #create two images one for each cell 
    os.chdir(path)
    
    for p in PATHS :
        #print(p)
        os.chdir(p)
        A = np.array(os.listdir())
        B = np.array([file.endswith(".ply") for file in A])
        A = A[B]            
        for file in A:
            name, ext = os.path.splitext(file)
            t = name.split('time')[1].split('_')[0] 
            if t == str(time):
                #print(time)
                #label = int(name.split('_')[1])
                mesh,V,T = uf.get_vertices_triangles(file)
                if 'cell_1' in name:
                    cell1 = V
                else:
                    cell2 = V
    return(cell1,cell2)

def plot_cloud(point_clouds):
    '''point_clouds is a list of point clouds to plot'''
    
    list_to_plot = []
    
    colors = [[0,0,1], [1,0,0], [0,1,0]]
    i = 0
    for point_cloud in point_clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.paint_uniform_color(colors[i])
        list_to_plot.append(pcd)
        i+=1
    o3d.visualization.draw_geometries(list_to_plot)
    return()

def compute_distance_between_two_clouds(cloud1, cloud2):
    
    '''return the minimal distance between points of cloud1 and cloud 2 (dist1)
    and the same for points of cloud2 and cloud1 (dist2)'''
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(cloud1)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(cloud2)
    
    distance1 = pcd1.compute_point_cloud_distance(pcd2)
    distance2 = pcd2.compute_point_cloud_distance(pcd1)
    dist1 = np.asarray(distance1)
    dist2 = np.asarray(distance2)
    return(dist1, dist2)

def compute_center_cloud(cloud):
    
    xc = np.mean(cloud[:,0])
    yc = np.mean(cloud[:,1])
    zc = np.mean(cloud[:,2])
    return(xc,yc,zc)

def compute_distance_to_center(cloud):
    
    xc,yc,zc = compute_center_cloud(cloud)
    x, y, z = cloud[:,0], cloud[:,1], cloud[:,2]
    dist = np.sqrt((x-xc)**2+(y-yc)**2+(z-zc)**2)
    return(dist)

def outter_interface(cloud):
    
    d = compute_distance_to_center(cloud)
    out_int = cloud[d/np.max(d)>0.6]
    return(out_int)

def select_with_ecad_levels(out_int, scale_factor, img,n_pix):

    out_int_img = np.zeros(img.shape)

    cloud_pix = uf.convert_pos_to_pix(out_int,scale_factor)
    cloud_pix = cloud_pix[:, [2, 1, 0]]
    x,y,z = cloud_pix[:,0], cloud_pix[:,1], cloud_pix[:,2]
    
    points_to_keep = [[],[],[]]
    
    V = []
    threshold = np.mean(img)*1.0
    print(threshold)
    for k in range(len(cloud_pix)):
        v=[]
        for i in range(-n_pix,n_pix+1):
            for j in range(-n_pix,n_pix+1):
                x0 = x[k] + i
                y0 = y[k] + j
                z0 = z[k]
                v.append(img[z0][x0][y0])
        value_signal = np.mean(v)
        V.append(value_signal)
        if value_signal > threshold :
            points_to_keep[0].append(out_int[k,0])
            points_to_keep[1].append(out_int[k,1])
            points_to_keep[2].append(out_int[k,2])
    plt.figure()
    plt.hist(V-threshold,bins = 100)
    int_select = np.vstack((np.array(points_to_keep[0]),np.array(points_to_keep[1]),np.array(points_to_keep[2])))
    return(int_select.T)

def save_snapshot_with_interface(path, PATHS,img,time, name, scale_factor):
    
    img_t = img[time-1]
    
    ##### compute interface cloud #####
    cell1, cell2 = cells_to_cloud_points(path, PATHS, time)
    dist1, dist2 = compute_distance_between_two_clouds(cell1, cell2)
    interface_cloud = np.vstack((cell1[dist1<threshold],cell2[dist2<threshold]))
    
    d = compute_distance_to_center(interface_cloud)
    out_int = interface_cloud[d/np.max(d)>0.75]
    int_int = interface_cloud[d/np.max(d)<=0.75]
    
    p = select_with_ecad_levels(out_int, scale_factor, img[time-1],2)
    
    interface_cloud = np.vstack((int_int,p))
    ###################################
   
    ###### convert cloud pos to img pos ######
    interface_img = np.zeros(img_t.shape)
    V_used = uf.convert_pos_to_pix(interface_cloud,scale_factor)
    interface_img = uf_int.label_region(interface_img,V_used,0,1)
    
    uf_int.save_img_two_channels_single_timepoint(path,img_t,interface_img,'new_interface_points_t'+str(time)+'.tif',[1/scale_factor,1/scale_factor],10*60)

    return()



def naive_fit_plane(interface_vertices): 
    
    Cint=np.mean(interface_vertices,axis=0)
    svd=np.linalg.svd((interface_vertices-Cint).T)
    #the last vector is the normal vector of the plane
    Nint=svd[0][:,-1]
    
    #we use the first and second vectors as ex and ey axis in the plane
    #they are normalized already, so let's compute coordinates and height
    ex=svd[0][:,0]
    ey=svd[0][:,1]
    H=np.dot(interface_vertices-Cint,Nint)
    X=np.dot(interface_vertices-Cint,ex)
    Y=np.dot(interface_vertices-Cint,ey)
    
    return([ex, ey, Nint], [X,Y,H])

def plane_cloud(interface_cloud):
    
    
    [ex, ey, Nint], [X,Y,H] = naive_fit_plane(interface_cloud)
    x,y,z = interface_cloud[:,0],interface_cloud[:,1],interface_cloud[:,2]

    n_scalar_p = Nint[0]*(x-np.mean(x)) +Nint[1]*(y-np.mean(y))+Nint[2]*(z-np.mean(z))
    
    x_plane = x - n_scalar_p*Nint[0]
    y_plane = y - n_scalar_p*Nint[1]
    z_plane = z - n_scalar_p*Nint[2]
    
    return(np.vstack((x_plane,y_plane,z_plane)).T)

def interface_polynomial3_fit(interface_cloud):
    
    [ex, ey, Nint], [X,Y,H] = naive_fit_plane(interface_cloud)
    #polynomial fit to interface
    A=np.array([X*0+1,X,Y,X**2,Y**2,X*Y,X**3,X**2*Y,X*Y**2,Y**3]).T
    coeff, r, rank,s=np.linalg.lstsq(A,H)
    
    Hfit=coeff[0]+X*coeff[1]+Y*coeff[2]+X**2*coeff[3]+Y**2*coeff[4]+X*Y*coeff[5]
    Hfit=Hfit+coeff[6]*X**3+coeff[7]*X**2*Y+coeff[8]*X*Y**2+coeff[9]*Y**3
    
    return(X,Y,Hfit,H)


def interface_polynomial3_fit_coeff(interface_cloud):
    
    [ex, ey, Nint], [X,Y,H] = naive_fit_plane(interface_cloud)
    #polynomial fit to interface
    A=np.array([X*0+1,X,Y,X**2,Y**2,X*Y,X**3,X**2*Y,X*Y**2,Y**3]).T
    coeff, r, rank,s=np.linalg.lstsq(A,H)
    
    H0 = coeff[0]
    H1 = X*coeff[1] +Y*coeff[2]
    H2 = X**2*coeff[3]+Y**2*coeff[4]+X*Y*coeff[5]
    H3 = coeff[6]*X**3+coeff[7]*X**2*Y+coeff[8]*X*Y**2+coeff[9]*Y**3
    Hfit=H0+H1+H2 + H3
        
    return(X,Y,Hfit,H1,H2,H3)

def plot_component_fit(interface_cloud,t, Vmin, Vmax):
    
    X,Y,Hfit,H1,H2,H3 = interface_polynomial3_fit_coeff(interface_cloud)
    
    xmin = -70
    ymin = -70
    xmax = 70
    ymax = 70
    
    fig, axs = plt.subplots(3, 2)
    axs[0,0].scatter(X,Y, c= Hfit, vmin = Vmin , vmax = Vmax)
    axs[0, 0].set_aspect('equal', 'box')
    axs[0, 0].set_xlim([xmin, xmax])
    axs[0, 0].set_ylim([ymin, ymax])
    axs[0,1].scatter(X,Y, c= H1, vmin =  Vmin , vmax = Vmax)
    axs[0, 1].set_aspect('equal', 'box')
    axs[0, 1].set_xlim([xmin, xmax])
    axs[0, 1].set_ylim([ymin, ymax])
    axs[1,0].scatter(X,Y, c= H2, vmin =  Vmin , vmax = Vmax)
    axs[1, 0].set_aspect('equal', 'box')
    axs[1, 0].set_xlim([xmin, xmax])
    axs[1, 0].set_ylim([ymin, ymax])
    axs[1,1].scatter(X,Y, c= H3, vmin =  Vmin , vmax =  Vmax)
    axs[1, 1].set_aspect('equal', 'box')
    axs[1, 1].set_xlim([xmin, xmax])
    axs[1, 1].set_ylim([ymin, ymax])
    axs[2,0].scatter(X,Y, c= H1+H3, vmin = Vmin , vmax = Vmax)
    axs[2, 0].set_aspect('equal', 'box')
    axs[2, 0].set_xlim([xmin, xmax])
    axs[2, 0].set_ylim([ymin, ymax])
    axs[2, 1].set_xlim([xmin, xmax])
    axs[2, 1].set_ylim([ymin, ymax])
    
    fig.suptitle('time = '+str(t))

    return()


def fit_cloud_degree3(interface_cloud):
    
    
    [ex, ey, Nint], [X,Y,H] = naive_fit_plane(interface_cloud)
    #polynomial fit to interface
    A=np.array([X*0+1,X,Y,X**2,Y**2,X*Y,X**3,X**2*Y,X*Y**2,Y**3]).T
    coeff, r, rank,s=np.linalg.lstsq(A,H)
    # print(coeff)
    Hfit=coeff[0]+X*coeff[1]+Y*coeff[2]+X**2*coeff[3]+Y**2*coeff[4]+X*Y*coeff[5]
    Hfit=Hfit+coeff[6]*X**3+coeff[7]*X**2*Y+coeff[8]*X*Y**2+coeff[9]*Y**3
    
    x,y,z = interface_cloud[:,0],interface_cloud[:,1],interface_cloud[:,2]

    n_scalar_p = Nint[0]*(x-np.mean(x)) +Nint[1]*(y-np.mean(y))+Nint[2]*(z-np.mean(z))
    
    #mean plane + Hfit * normal_vector
    x_plane_fit = x - n_scalar_p*Nint[0] + Hfit*Nint[0]
    y_plane_fit = y - n_scalar_p*Nint[1] + Hfit*Nint[1]
    z_plane_fit = z - n_scalar_p*Nint[2] + Hfit*Nint[2]
    
    return(np.vstack((x_plane_fit,y_plane_fit,z_plane_fit)).T)
    


def fit_cloud_degree4(interface_cloud):
    
    
    [ex, ey, Nint], [X,Y,H] = naive_fit_plane(interface_cloud)
    #polynomial fit to interface
    A=np.array([X*0+1,X,Y,X**2,Y**2,X*Y,X**3,X**2*Y,X*Y**2,Y**3, X**4, X**3*Y, X**2*Y**2,X*Y**3, Y**4]).T
    coeff, r, rank,s=np.linalg.lstsq(A,H)
    
    Hfit=coeff[0]+X*coeff[1]+Y*coeff[2]+X**2*coeff[3]+Y**2*coeff[4]+X*Y*coeff[5]
    Hfit=Hfit+coeff[6]*X**3+coeff[7]*X**2*Y+coeff[8]*X*Y**2+coeff[9]*Y**3
    Hfit = Hfit+coeff[10]*X**4 + coeff[11]*X**3*Y + coeff[12]*X**2*Y**2 + coeff[13]*X*Y**3 + coeff[14]*Y**4
    
    x,y,z = interface_cloud[:,0],interface_cloud[:,1],interface_cloud[:,2]

    n_scalar_p = Nint[0]*(x-np.mean(x)) +Nint[1]*(y-np.mean(y))+Nint[2]*(z-np.mean(z))
    
    #mean plane + Hfit * normal_vector
    x_plane_fit = x - n_scalar_p*Nint[0] + Hfit*Nint[0]
    y_plane_fit = y - n_scalar_p*Nint[1] + Hfit*Nint[1]
    z_plane_fit = z - n_scalar_p*Nint[2] + Hfit*Nint[2]
    
    return(np.vstack((x_plane_fit,y_plane_fit,z_plane_fit)).T)


def compute_Vmin_Vmax_series(PATHS, startpoint,endpoint, threshold):
    
    Vmin = []
    Vmax = []

    for t in range(startpoint, endpoint+1):
        print(t)
        timepoint = t
        cell1, cell2 = bm.get_cells_cloud_time(timepoint, PATHS)
        dist1, dist2 = compute_distance_between_two_clouds(cell1, cell2)
        interface_cloud = np.vstack((cell1[dist1<threshold],cell2[dist2<threshold]))
        X,Y,Hfit,H1,H2,H3 = interface_polynomial3_fit_coeff(interface_cloud)
        
        Vmin.append(np.min(H1+H3))
        Vmax.append(np.max(H1+H3))
    
    Vmin = np.min(Vmin)
    Vmax = np.max(Vmax)
    return(Vmin,Vmax)

def plot_interface_series(PATHS, threshold):
    
    
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])

    Vmin, Vmax = compute_Vmin_Vmax_series(PATHS, startpoint,endpoint, threshold)

    for t in range(startpoint, endpoint+1):
        print(t)
        timepoint = t
    
        cell1, cell2 = bm.get_cells_cloud_time(timepoint, PATHS)
        dist1, dist2 = compute_distance_between_two_clouds(cell1, cell2)
        interface_cloud = np.vstack((cell1[dist1<threshold],cell2[dist2<threshold]))
        
        xfit,yfit,Hfit,H = interface_polynomial3_fit(interface_cloud)
        
        plot_component_fit(interface_cloud,t, Vmin, Vmax)
    
    return()

##############################################################################
#########################  plot intensities ##################################
##############################################################################

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

def plot_intensity_maps(interface_points, interface_intensity):#, name,save_path, save_plot=False):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(interface_points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=50.0,
                                             ransac_n=3,
                                             num_iterations=100000)
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    #### compute distance to the plane ####
    x = interface_points[:,0]
    y = interface_points[:,1]
    z = interface_points[:,2]
    
    H = (a*x+b*y+c*z+d)/np.sqrt(a**2+b**2+c**2)
    ah=np.mean(H)
    sh2=np.sqrt(np.mean(H*H))
    # print(f'sqrt(<h^2>)={sh2}')
    # print(f"Mean distance from the plane = {ah}")
    
    
    plane_fit = a*x+b*y+c*z+d;
    
    planar_x = x - plane_fit*a;
    planar_y = y - plane_fit*b;
    planar_z = z - plane_fit*c;
    
    n = np.array([a,b,c])
    e1 = np.array([0,0,1])
    e1 = e1 - np.dot(e1, n) * n
    e1/= np.sqrt((e1**2).sum())
    e2 = np.cross(n, e1)
    
    x_plane = np.copy(x)
    y_plane = np.copy(y)
    for i,ix in enumerate(x):
        A = np.array([planar_x[i], planar_y[i], planar_z[i]])
        x_plane[i] = np.dot(A, e1)
        y_plane[i] = np.dot(A, e2)
    
    xc,yc = compute_xc_yc(x_plane,y_plane)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x_plane-xc,y_plane-yc,c=H, vmin = -25, vmax=25)
    
    ax.set_aspect('equal', adjustable='box')
    plt.ylim([-80,80])
    plt.xlim([-80,80])
    plt.colorbar()
    plt.title(name+' - n = '+str(len(interface_points))+' points')
    
    if save_plot == True:
        time = int(name.split('t = ')[-1])
        print(save_path)
        print(save_path+'interface_proj_saved/'+'interface_proj_t'+str(time)+'.jpg')
        fig.savefig(save_path+'interface_proj_t'+str(time)+'.jpg')
    #return(H/sh2)
    return()

def save_x_y_h_quantities(interface_cloud, save_path, filename):
    
    x,y,Hfit,H = interface_polynomial3_fit(interface_cloud)
    
    data_array = np.vstack((x,y,Hfit)).T
    np.save(save_path+filename+'.npy', data_array)
    
    
    return(data_array)

def save_plane_vectors(interface_cloud, save_path, filename): 
    
    [ex, ey, Nint], [X,Y,H] = naive_fit_plane(interface_cloud)
    data = np.vstack((ex,ey,Nint)).T
    np.save(save_path+filename+'.npy', data)
    return(data)


# path = 'E:/Rotation/Data_analysis_Figure_Article/1_cell_to_2_cells/b2/Segmentation_6_ecad/'
# path_cell_1 = path +'Cell_1/'
# path_cell_2 = path +'Cell_2/' 
# PATHS = [path_cell_1,path_cell_2]
# img_name = '6_ecad.tif'
# img = imread(path + img_name)
# scale_factor = 1/0.206

# time = 75

# cell1, cell2 = cells_to_cloud_points(path, PATHS, time)
# dist1, dist2 = compute_distance_between_two_clouds(cell1, cell2)

# threshold = 5
# l = [cell1[dist1<threshold],cell2[dist2<threshold]]
# int_cloud = np.vstack((cell1[dist1<threshold],cell2[dist2<threshold]))
# d = compute_distance_to_center(int_cloud)
# out_int = int_cloud[d/np.max(d)>0.75]
# int_int = int_cloud[d/np.max(d)<=0.75]

# p = select_with_ecad_levels(out_int, scale_factor, img[time-1],2)

# interface_points = np.vstack((int_int,p))

# int1 = cell1[dist1<threshold]

# int2 = cell2[dist2<threshold]


# # i_hull, i_points, i_bspline, interface_img = uf_int.label_interface_complete(path,PATHS,img_name,time,4,[1/scale_factor,1/scale_factor],plot=False, save_img = False, save_plot = False, save_final_plot = False)
# plot_cloud([int1,int2])
# name = 'f'
# save_snapshot_with_interface(path, PATHS,img,time, name, scale_factor)