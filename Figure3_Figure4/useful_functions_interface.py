# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:55:40 2022

@author: Tristan Guyomar
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


def find_startpoint_endpoint(cell_path):
    
    a = os.listdir(cell_path)
    #added line for macos problem with DS_store files
    b = [int(e.split('time')[1].split('_')[0]) for e in a if not e=='.DS_Store']
    startpoint = min(b)
    endpoint = max(b)
    return(startpoint,endpoint)



def label_region(img_seg, pixel_positions,n_pix,label):
    
    if type(pixel_positions)==tuple:
        #print('tuple type for pixel_positions')
        ##it is the case when you give a np.where(img==value) where img is an array of size(nz,nx,ny)
        pixel_positions = np.array([row for row in pixel_positions if sum(row)!=0]).T
    else :
        #print('vertices type for pixel_positions')
        # it is the case when label_region is used with the vertex of the mesh as pixel_positions
        pixel_positions = np.array([row for row in pixel_positions if sum(row)!=0])
    for i in range(-n_pix,n_pix+1):
        for j in range(-n_pix,n_pix+1):
            x = pixel_positions[:, 1] + i
            y = pixel_positions[:, 2] + j
            z = pixel_positions[:, 0]
            img_seg[z, x, y] = label
            
    return(img_seg)


def segment_single_cell(img_seg, mesh, V, T, scale_factor, n_pix,label):
    
    V_used = uf.convert_pos_to_pix(V,scale_factor)
    img_seg = label_region(img_seg,V_used,n_pix,label)
    
    return

def label_objects(path,PATHS, img_shape, time, pixel_dims, n_pix):
    
    # return the segmented image based on a single ply file present in ply_path
    # img_name is the name of the image present in the folder ply_path
    # scale_factor is the factor 
    
    img = imread(img_name)
    img_seg_tot = np.zeros((1,*img_shape), dtype=int)
    
    for p in PATHS :
        #print(p)
        os.chdir(p)
        A = np.array(os.listdir())
        B = np.array([file.endswith(".ply") for file in A])
        A = A[B]
        
        for file in A:
            name, ext = os.path.splitext(file)
            if name.split('_')[-1] ==str(time):
                #print(name)
                #label = int(name.split('_')[1])
                mesh,V,T = uf.get_vertices_triangles(file)
                segment_single_cell(img_seg_tot[0], mesh, V, T, 1.0/pixel_dims[0], n_pix, 1)
    I_holes = img_seg_tot.copy()
    for i in range(len(I_holes[0])):
        I_holes[0,i] = ndimage.binary_fill_holes(I_holes[0,i]).astype(int)
    I_holes[0] -= img_seg_tot[0]
    img_seg_tot[0][np.where(I_holes[0] == 1)] +=2
    return(img_seg_tot)


def segment_two_cells(path, PATHS, img_shape, time, n_pix,pixel_dims):
    
    "warning : treat a single time point "
    
    #create two images one for each cell 
    os.chdir(path)
    if np.size(img_shape) == 3 :
        cell1 = np.zeros(img_shape, dtype=int)
        cell2 = np.zeros(img_shape, dtype=int)
        
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
                        segment_single_cell(cell1, mesh, V, T, 1.0/pixel_dims[0], n_pix, 1)
                    else:
                        segment_single_cell(cell2, mesh, V, T, 1.0/pixel_dims[0], n_pix, 1) 
    else :
        print('you did not give a single timepoint to segment')
    return(cell1,cell2)

def get_cells_to_plot(path,PATHS,img_shape,time,n_pix,pixel_dims):

    cell1,cell2 = segment_two_cells(path,PATHS,img_shape,time, n_pix,pixel_dims)
    cell1, cell2 = np.where(cell1!=0), np.where(cell2!=0)
    
    cell1_points = np.array([[cell1[1][i],cell1[2][i],cell1[0][i]/pixel_dims[0]] for i in range(0,len(cell1[0]))])
    cell2_points = np.array([[cell2[1][i],cell2[2][i],cell2[0][i]/pixel_dims[0]] for i in range(0,len(cell2[0]))])
    
    return(cell1_points,cell2_points)


def compute_convex_hull_cells(cell1,cell2):
    
    #retrieve the points corresponding to the cells from which we compute the 3D convex hull
    filt_c1 = np.where(cell1==1)
    filt_c2 = np.where(cell2==1)
    x = np.hstack((filt_c1[0],filt_c2[0]))
    y = np.hstack((filt_c1[1],filt_c2[1]))
    z = np.hstack((filt_c1[2],filt_c2[2]))
    
    
    points_to_hull = np.vstack((x,y,z)).T
    
    #compute the 3D convexHull
    hull = ConvexHull(points_to_hull)
    #get the points that are on the hull
    points_boundary_hull = points_to_hull[hull.vertices]
    
    return(points_to_hull,points_boundary_hull)

def compute_min_distance(points_boundary_hull,points_to_hull):
    
    #for each points in points_to_hull it compute the minimal distance to the points in the convex hull
    min_dist = np.zeros(points_to_hull.shape[0])
    for i,p in enumerate(points_to_hull):
        dist = np.sqrt((p[0]-points_boundary_hull[:,0])**2+(p[1]-points_boundary_hull[:,1])**2+(p[2]-points_boundary_hull[:,2])**2)
        min_dist[i] = np.min(dist[dist>0])
        
    return(min_dist)

def select_interface(points_to_hull, min_dist,threshold) :
    
    return(points_to_hull[np.where(min_dist>threshold)])


def label_interface_with_cell_hull(path,PATHS,img_shape,time,n_pix,pixel_dims):
    "label cell doublet interface for a single timepoint "
    
    #extract segmented cells 
    cell1,cell2 = segment_two_cells(path,PATHS,img_shape,time, n_pix,pixel_dims)
    
    points_to_hull, points_boundary_hull = compute_convex_hull_cells(cell1,cell2)
    threshold = 8
    min_dist = compute_min_distance(points_boundary_hull,points_to_hull)
    
    interface_hull = select_interface(points_to_hull, min_dist,threshold)
    
    interface = np.zeros(cell1.shape)

    for p in interface_hull:

        interface[p[0]][p[1]][p[2]] = 255
    
    I = np.where(interface!=0)
    interface_points = np.array([[I[1][i],I[2][i],I[0][i]/pixel_dims[0]] for i in range(0,len(I[0]))])
    
    # #fill img_interface with the info from the two other cells
    # filt = cell1+cell2>1
    # print(filt.shape)
    # interface1 = np.where(filt==False, 0, cell1_0)
    # interface2 = np.where(filt==False,0,cell2_0)
    # interface3 = interface1 + interface2
    # interface4 = np.where(interface3>=1, 255,interface3)
    
    
    return(interface_points)


def label_interface_with_convex_hull_to_plot(path,PATHS,img_shape,time,n_pix,pixel_dims):
    
    #extract segmented cells 
    cell1,cell2 = segment_two_cells(path,PATHS,img_shape,time, n_pix,pixel_dims)

    cell1_0, cell2_0 = segment_two_cells(path,PATHS,img_shape, time, 0,pixel_dims)
    
    #fill img_interface with the info from the two other cells
    filt = cell1+cell2>1
    interface1 = np.where(filt==False, 0, cell1_0)
    interface2 = np.where(filt==False,0,cell2_0)
    interface3 = interface1 + interface2
    interface = np.where(interface3>=1, 1,interface3)
    
    ## here we look for points inside the convex hull of the interface
    filt = np.where(interface==1)
    points_to_hull = np.vstack((filt[0],filt[1],filt[2])).T
    #compute the 3D convexHull
    
    hull = ConvexHull(points_to_hull)
    deln = Delaunay(points_to_hull[hull.vertices])
    idx = np.stack(np.indices(cell1.shape), axis = -1)
    inside_convex_hull = np.nonzero(deln.find_simplex(idx) + 1)
    z = inside_convex_hull[0]
    x = inside_convex_hull[1]
    y = inside_convex_hull[2]
    
    interface = np.zeros(cell1.shape)
    cells = cell1+cell2
    for i,e in enumerate(z) :
        if (cells[e][x[i]][y[i]]>=1) :
            interface[e][x[i]][y[i]] = 1
    
    #interface_out = np.where(interface>=1)
    
    I = np.where(interface!=0)
    interface_points = np.array([[I[1][i],I[2][i],I[0][i]/pixel_dims[0]] for i in range(0,len(I[0]))])
        
    
    # z_int = np.unique(interface_points[:,1])
    # print(z_int)
    # int_points = [[],[],[]]
    # for z in z_int:
    #     interface_points1 = interface_points[interface_points[:,1]==z]
    #     y_int = np.unique(interface_points1[:,2])
    #     for y in y_int:
    #         int_points[0].append(np.mean(interface_points1[interface_points1[:,2]==y][:,0]))
    #         int_points[1].append(z)
    #         int_points[2].append(y)
    
    
    # z_int = np.unique(interface_points[:,0])
    # print(z_int)
    # int_points1 = [[],[],[]]
    # for z in z_int:
    #     interface_points1 = interface_points[interface_points[:,0]==z]
    #     y_int = np.unique(interface_points1[:,2])
    #     for y in y_int:
    #         int_points1[1].append(np.mean(interface_points1[interface_points1[:,2]==y][:,1]))
    #         int_points1[0].append(z)
    #         int_points1[2].append(y)
            

    # z_int = np.unique(interface_points[:,2])
    # print(z_int)
    # int_points2 = [[],[],[]]
    # for z in z_int:
    #     interface_points1 = interface_points[interface_points[:,2]==z]
    #     y_int = np.unique(interface_points1[:,1])
    #     for y in y_int:
    #         int_points2[0].append(np.mean(interface_points1[interface_points1[:,1]==y][:,0]))
    #         int_points2[2].append(z)
    #         int_points2[1].append(y)
    
    
    # int_points = np.array(int_points).T
    # int_points1 = np.array(int_points1).T
    # int_points2 = np.array(int_points2).T
    
    # print('along y direction',len(int_points))
    # print('along x direction',len(int_points1))
    # print('along z direction',len(int_points2))
    
    # if (len(int_points1)>len(int_points) and len(int_points1)>len(int_points2)):
    #     interface_points = int_points1
    # elif (len(int_points2)>len(int_points)):
    #     interface_points = int_points2
    # else : 
    #     interface_points = int_points
        
    return(interface_points)


def label_interface_with_convex_hull(path,PATHS,img_shape,time,n_pix,pixel_dims):
    
    #extract segmented cells 
    cell1,cell2 = segment_two_cells(path,PATHS,img_shape,time, n_pix,pixel_dims)

    cell1_0, cell2_0 = segment_two_cells(path,PATHS,img_shape, time, 0,pixel_dims)
    
    #fill img_interface with the info from the two other cells
    filt = cell1+cell2>1
    interface1 = np.where(filt==False, 0, cell1_0)
    interface2 = np.where(filt==False,0,cell2_0)
    interface3 = interface1 + interface2
    interface = np.where(interface3>=1, 1,interface3)
    
    ## here we look for points inside the convex hull of the interface
    filt = np.where(interface==1)
    points_to_hull = np.vstack((filt[0],filt[1],filt[2])).T
    #compute the 3D convexHull
    
    hull = ConvexHull(points_to_hull)
    deln = Delaunay(points_to_hull[hull.vertices])
    idx = np.stack(np.indices(cell1.shape), axis = -1)
    inside_convex_hull = np.nonzero(deln.find_simplex(idx) + 1)
    z = inside_convex_hull[0]
    x = inside_convex_hull[1]
    y = inside_convex_hull[2]
    
    interface = np.zeros(cell1.shape)
    cells = cell1+cell2
    for i,e in enumerate(z) :
        if (cells[e][x[i]][y[i]]>=1) :
            interface[e][x[i]][y[i]] = 1
    
    #interface_out = np.where(interface>=1)
    
    
    
    I = np.where(interface!=0)
    interface_points = np.array([[I[1][i],I[2][i],I[0][i]] for i in range(0,len(I[0]))])
    
    ##########################################################################
    ################## select the interface ##################################
    ##########################################################################
    
    
    x_int = np.unique(interface_points[:,0])
    y_int = np.unique(interface_points[:,1])
    z_int = np.unique(interface_points[:,2])
    
    int_points = [[],[],[]]
    for y in y_int:
        interface_points1 = interface_points[interface_points[:,1]==y]
        z_unique = np.unique(interface_points1[:,2])
        for z in z_unique:
            int_points[0].append(np.mean(interface_points1[interface_points1[:,2]==z][:,0]))
            int_points[1].append(y)
            int_points[2].append(z)
    
    interface_points_y_along_z = np.array(int_points).T
    
    int_points = [[],[],[]]
    
    for x in x_int:
        interface_points1 = interface_points[interface_points[:,0]==x]
        y_unique = np.unique(interface_points1[:,1])
        for y in y_unique:
            int_points[2].append(np.mean(interface_points1[interface_points1[:,1]==y][:,2]))
            int_points[1].append(y)
            int_points[0].append(x)
    
    interface_points_x_along_y = np.array(int_points).T
    
    int_points = [[],[],[]]
    
    for x in x_int:
        interface_points1 = interface_points[interface_points[:,0]==x]
        z_unique = np.unique(interface_points1[:,2])
        for z in z_unique:
            int_points[1].append(np.mean(interface_points1[interface_points1[:,2]==z][:,1]))
            int_points[0].append(x)
            int_points[2].append(z)
    
    interface_points_x_along_z = np.array(int_points).T
    
    
    ##########################################################################
    ############ data to plot ################################################
    ##########################################################################
    
    interface_img = np.zeros((3,*cell1.shape))
    other = np.zeros(cell1.shape)
    for j,interface_points in enumerate([interface_points_x_along_y, interface_points_x_along_z, interface_points_y_along_z]) :
        
        #interface_points = interface_points_x_along_z
        x = interface_points[:,0]
        y = interface_points[:,1]
        z = interface_points[:,2]
        
        for i,e in enumerate(z) :
            #print(e,x[i],y[i])
            interface_img[j][int(e)][int(x[i])][int(y[i])] = 1
            
        
    return(interface_img,other)



def label_interface_complete(path,PATHS,img_name,time,n_pix,pixel_dims,plot=False, save_img = False, save_plot = False, save_final_plot = False):
    
    #get image and infos
    
    img = imread(path + img_name)
    img1 = img[time-1]
    img_shape = img1.shape
    
    save_path = path
    
    #extract segmented cells 
    cell1,cell2 = segment_two_cells(path,PATHS,img_shape,time, n_pix,pixel_dims)

    cell1_0, cell2_0 = segment_two_cells(path,PATHS,img_shape, time, 0,pixel_dims)
    
    #fill img_interface with the info from the two other cells
    filt = cell1+cell2>1
    interface1 = np.where(filt==False, 0, cell1_0)
    interface2 = np.where(filt==False,0,cell2_0)
    interface3 = interface1 + interface2
    interface = np.where(interface3>=1, 1,interface3)
    
    ## here we look for points inside the convex hull of the interface
    filt = np.where(interface==1)
    points_to_hull = np.vstack((filt[0],filt[1],filt[2])).T
    #compute the 3D convexHull
    
    hull = ConvexHull(points_to_hull)
    deln = Delaunay(points_to_hull[hull.vertices])
    idx = np.stack(np.indices(cell1.shape), axis = -1)
    inside_convex_hull = np.nonzero(deln.find_simplex(idx) + 1)
    z = inside_convex_hull[0]
    x = inside_convex_hull[1]
    y = inside_convex_hull[2]
    
    interface = np.zeros(cell1.shape)
    cells = cell1+cell2
    for i,e in enumerate(z) :
        if (cells[e][x[i]][y[i]]>=1) :
            interface[e][x[i]][y[i]] = 1
    
    I = np.where(interface!=0)
    interface_points = np.array([[I[1][i],I[2][i],I[0][i]/pixel_dims[0]] for i in range(0,len(I[0]))])
    
    if plot==True:
        print('lol')
        plot_x_plane_y_plane_H(interface_points,'Interface from convex hull : set of points B',save_path, save_plot)
        # interface_points_bspline = interface_from_NURBS(interface_points, path, 0.01, 'z')
        
        # plot_x_plane_y_plane_H(interface_points_bspline,'Final interface after bspline - t = '+str(time),save_path, save_plot)
    
    full_interface_points = np.copy(interface_points)
    
    ##########################################################################
    ##### compute the interface from the convex hull of cell cell contact ####
    ##########################################################################
    
    x_int = np.unique(interface_points[:,0])
    y_int = np.unique(interface_points[:,1])
    z_int = np.unique(interface_points[:,2])
    
    int_points = [[],[],[]]
    for y in y_int:
        interface_points1 = interface_points[interface_points[:,1]==y]
        z_unique = np.unique(interface_points1[:,2])
        for z in z_unique:
            int_points[0].append(np.mean(interface_points1[interface_points1[:,2]==z][:,0]))
            int_points[1].append(y)
            int_points[2].append(z)
    
    interface_points_y_along_z = np.array(int_points).T
    
    int_points = [[],[],[]]
    
    for x in x_int:
        interface_points1 = interface_points[interface_points[:,0]==x]
        y_unique = np.unique(interface_points1[:,1])
        for y in y_unique:
            int_points[2].append(np.mean(interface_points1[interface_points1[:,1]==y][:,2]))
            int_points[1].append(y)
            int_points[0].append(x)
    
    interface_points_x_along_y = np.array(int_points).T
    
    int_points = [[],[],[]]
    
    for x in x_int:
        interface_points1 = interface_points[interface_points[:,0]==x]
        z_unique = np.unique(interface_points1[:,2])
        for z in z_unique:
            int_points[1].append(np.mean(interface_points1[interface_points1[:,2]==z][:,1]))
            int_points[0].append(x)
            int_points[2].append(z)
    
    interface_points_x_along_z = np.array(int_points).T

    
    ##########################################################################
    ############## compute the interface from cell cell contact ##############
    ##########################################################################
    
    
    interface,other = label_interface1(path,PATHS,img_shape,time,4,pixel_dims)
    I = np.where(interface!=0)
    interface_points_cell_touch = np.array([[I[1][i],I[2][i],I[0][i]/pixel_dims[0]] for i in range(0,len(I[0]))])
    
    

    if plot==True:
        print('lol')
        # interface_points_bspline = interface_from_NURBS(interface_points_cell_touch, path, 0.01,'z')    
        plot_x_plane_y_plane_H(interface_points_cell_touch,'Set of points A: cell-cell contact- t = '+str(time),save_path, save_plot)
        # plot_x_plane_y_plane_H(interface_points_bspline,'Final interface after bspline - t = '+str(time),save_path, save_plot)
        plot_x_plane_y_plane_H(interface_points_y_along_z,'interface_points_y_along_z- t = '+str(time),save_path, save_plot)
        plot_x_plane_y_plane_H(interface_points_x_along_y,'interface_points_x_along_y- t = '+str(time),save_path, save_plot)
        plot_x_plane_y_plane_H(interface_points_x_along_z,'interface_points_x_along_z- t = '+str(time),save_path, save_plot)


    interface_selected,ind  = select_direction_of_proj(interface_points_y_along_z, interface_points_x_along_y, interface_points_x_along_z, interface_points_cell_touch)
    interface_selected = np.vstack((interface_points_cell_touch,interface_selected))
    
    
    if plot==True:
        
        plot_x_plane_y_plane_H(interface_selected,'Set A and selected set C - t = '+str(time),save_path, save_plot)
        # interface_points_bspline = interface_from_NURBS(interface_selected, path, 0.01,'z')    
        # plot_x_plane_y_plane_H(interface_points_bspline,'Final interface after bspline - t = '+str(time),save_path, save_plot)
    
    ##########################################################################
    ##### compute the interface from the previous selection ##################
    ##########################################################################
    
    interface_points = np.copy(interface_selected)
    x_int = np.unique(interface_points[:,0])
    y_int = np.unique(interface_points[:,1])
    z_int = np.unique(interface_points[:,2])
    
    int_points = [[],[],[]]
    for y in y_int:
        interface_points1 = interface_points[interface_points[:,1]==y]
        z_unique = np.unique(interface_points1[:,2])
        for z in z_unique:
            int_points[0].append(np.mean(interface_points1[interface_points1[:,2]==z][:,0]))
            int_points[1].append(y)
            int_points[2].append(z)
    
    interface_points_y_along_z = np.array(int_points).T
    
    int_points = [[],[],[]]
    
    for x in x_int:
        interface_points1 = interface_points[interface_points[:,0]==x]
        y_unique = np.unique(interface_points1[:,1])
        for y in y_unique:
            int_points[2].append(np.mean(interface_points1[interface_points1[:,1]==y][:,2]))
            int_points[1].append(y)
            int_points[0].append(x)
    
    interface_points_x_along_y = np.array(int_points).T
    
    int_points = [[],[],[]]
    
    for x in x_int:
        interface_points1 = interface_points[interface_points[:,0]==x]
        z_unique = np.unique(interface_points1[:,2])
        for z in z_unique:
            int_points[1].append(np.mean(interface_points1[interface_points1[:,2]==z][:,1]))
            int_points[0].append(x)
            int_points[2].append(z)

    
    interface_points_x_along_z = np.array(int_points).T
    
    interfaces = [interface_points_y_along_z,interface_points_x_along_y,interface_points_x_along_z]
    
    interface_final = interfaces[ind]
    
    #print(interface_final.shape)
    
    if plot==True:
        plot_x_plane_y_plane_H(interface_final,'Final interface before bspline - t = '+str(time), save_plot, save_path)
        
    
    ##########################################################################
    ############### NURBS algorithm to fit interpolate #######################
    ##########################################################################
    

    interface_points_bspline_x = interface_from_NURBS(interface_final, path, 0.01,'x')    
    interface_points_bspline_y = interface_from_NURBS(interface_final, path, 0.01,'y') 
    interface_points_bspline_z = interface_from_NURBS(interface_final, path, 0.01,'z')
    interfaces = [interface_points_bspline_x,interface_points_bspline_y,interface_points_bspline_z]
    sizes=[]
    for i in interfaces :
        a = np.unique(i[:,0])
        a1 = np.unique(i[:,1])
        a2 = np.unique(i[:,2])
        sizes.append(len(a)*len(a1)*len(a2))
    
    interface_points_bspline = interfaces[np.argmax(sizes)]
    
    
    if save_final_plot==True:
        os.chdir(path)        
        if os.path.isdir('interface_proj_saved')==False:
        
            # Directory
            directory = "interface_proj_saved"
            # Parent Directory path
            parent_dir = path
      
            # Path
            save_path = os.path.join(parent_dir, directory)  
            os.mkdir(save_path)
        
        save_path = path+'interface_proj_saved/'
        plot_x_plane_y_plane_H(interface_points_bspline,'Final interface - t = '+str(time),save_path, save_final_plot)
        
    if plot == True:
        plot_x_plane_y_plane_H(interface_points_bspline,'Final interface after bspline - t = '+str(time), save_plot, save_path)
        # plot_x_plane_y_plane_H(interface_points_bspline_x,'Final interface after bspline - t = '+str(time), save_plot, save_path)
        # plot_x_plane_y_plane_H(interface_points_bspline_y,'Final interface after bspline - t = '+str(time), save_plot, save_path)
        # plot_x_plane_y_plane_H(interface_points_bspline_z,'Final interface after bspline - t = '+str(time), save_plot, save_path)

        
    ##########################################################################
    ############ data to plot ################################################
    ##########################################################################
    
    interface_img = np.zeros(cell1.shape)
    other = np.zeros(cell1.shape)
    
    x = interface_final[:,0]
    y = interface_final[:,1]
    z = interface_final[:,2]*pixel_dims[0]
        
    for i,e in enumerate(z) :
        interface_img[int(e)][int(x[i])][int(y[i])] = 1
    
    if save_img == True:
        
        img = imread(path + img_name)
        os.chdir(path)        
        if os.path.isdir('interface_saved')==False:
        
            # Directory
            directory = "interface_saved"
            # Parent Directory path
            parent_dir = path
      
            # Path
            save_path = os.path.join(parent_dir, directory)  
            os.mkdir(save_path)
        save_path = path+'interface_saved'
        save_img_two_channels_single_timepoint(save_path,img[t-1],interface_img,'interface_points_t'+str(t)+'.tif',pixel_dims,time_int)
    
    return(interface_points_cell_touch,interface_final,interface_points_bspline, interface_img)




def prepare_points_to_NURBS(interface_points,direction_to_use):
    
    x_int = np.unique(interface_points[:,0])
    y_int = np.unique(interface_points[:,1])
    z_int = np.unique(interface_points[:,2])
    
    if direction_to_use == 'x':
        ctrlpts_x = []
        L = []
        for x in x_int:
            
            points = interface_points[interface_points[:,0]==x]
            p = [list(e) for e in points]
            ctrlpts_x.append(p)
            L.append(len(p))
    
        for i,l in enumerate(ctrlpts_x):
            if len(l)<max(L):
                number_of_points_to_add = max(L)-len(l)
                new_l = [l[0] for i in range(number_of_points_to_add)]+l
                ctrlpts_x[i] = new_l
        ctrlpts = ctrlpts_x
    
    
    if direction_to_use == 'y':
        ctrlpts_y = []
        L = []
        for y in y_int:
            
            points = interface_points[interface_points[:,1]==y]
            p = [list(e) for e in points]
            ctrlpts_y.append(p)
            L.append(len(p))
    
        for i,l in enumerate(ctrlpts_y):
            if len(l)<max(L):
                number_of_points_to_add = max(L)-len(l)
                new_l = [l[0] for i in range(number_of_points_to_add)]+l
                ctrlpts_y[i] = new_l
        ctrlpts = ctrlpts_y
        
        
    if direction_to_use == 'z':
        ctrlpts_z = []
        L = []
        for z in z_int:
            
            points = interface_points[interface_points[:,2]==z]
            p = [list(e) for e in points]
            ctrlpts_z.append(p)
            L.append(len(p))
    
        for i,l in enumerate(ctrlpts_z):
            if len(l)<max(L):
                number_of_points_to_add = max(L)-len(l)
                new_l = [l[0] for i in range(number_of_points_to_add)]+l
                ctrlpts_z[i] = new_l
        ctrlpts = ctrlpts_z
    
    
    
    return(ctrlpts)



def select_direction_of_proj(interface1,interface2,interface3, interface_compare):
    
    x_plane,y_plane = compute_x_plane_y_plane_H(interface1)
    x_plane1,y_plane1 = compute_x_plane_y_plane_H(interface2)
    x_plane2,y_plane2 = compute_x_plane_y_plane_H(interface3)
    x_plane3,y_plane3 = compute_x_plane_y_plane_H(interface_compare)

    xx = [x_plane,x_plane1,x_plane2,x_plane3]
    yy = [y_plane,y_plane1,y_plane2,y_plane3]
    
    center_points = [compute_xc_yc(xx[i],yy[i]) for i in range(0,len(xx))]
    d = []
    for p in center_points:
        d.append(np.sqrt((center_points[3][0]-p[0])**2+(center_points[3][1]-p[1])**2))
    
    interfaces = [interface1,interface2,interface3]
    
    areas = [convex_area(xx[i],yy[i]) for i in range(0,len(xx))]
    d_areas = []
    for a in areas:
        d_areas.append(areas[3]-a)
    
    prod = np.array(d)*np.array(d_areas)
   
    
    interface = interfaces[np.argmin(prod[:-1])]
    ind = np.argmin(prod[:-1])
    
    print(ind)
    return(interface, ind)

def interface_from_NURBS(interface_points, path, delta, direction_to_use):
    
    ctrlpts = prepare_points_to_NURBS(interface_points, direction_to_use)
    
    # Create a BSpline surface
    surf = BSpline.Surface()

    # Set degrees
    surf.degree_u = 3
    surf.degree_v = 3

    # Set control points
    surf.ctrlpts2d = ctrlpts
    
    # Set knot vectors
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
    
    # Set evaluation delta
    surf.delta = delta

    # Evaluate surface points
    surf.evaluate()
    
    os.chdir(path)
    exchange.export_csv(surf,"surface_points.csv", two_dimensional=False)
    
    interface_points_bspline = np.genfromtxt('surface_points.csv', delimiter=',')
    interface_points_bspline = interface_points_bspline[1:]
    
    return(interface_points_bspline)



def label_interface1(path,PATHS,img_shape,time,n_pix,pixel_dims):
    "label cell doublet interface for a single timepoint "
    
    #extract segmented cells 
    cell1,cell2 = segment_two_cells(path,PATHS,img_shape,time, n_pix,pixel_dims)

    cell1_0, cell2_0 = segment_two_cells(path,PATHS,img_shape, time, 0,pixel_dims)
    
    #fill img_interface with the info from the two other cells
    filt = cell1+cell2>1
    print(filt.shape)
    interface1 = np.where(filt==False, 0, cell1_0)
    interface2 = np.where(filt==False,0,cell2_0)
    interface3 = interface1 + interface2
    interface4 = np.where(interface3>=1, 255,interface3)
    
    filt_other = ~((cell1+cell2==0) + filt)
    other1 = np.where(filt_other==False, 0, cell1_0)
    other2 = np.where(filt_other==False,0,cell2_0)
    other3 = other1 + other2
    other = np.where(other3>=1, 255,other3)
        
    return(interface4,other)






def label_interface2(path,PATHS,img_shape,time,n_pix,pixel_dims):
    "label cell doublet interface for a single timepoint "
    
    #extract segmented cells 
    cell1,cell2 = segment_two_cells(path,PATHS,img_shape,time, n_pix,pixel_dims)

    cell1_0, cell2_0 = segment_two_cells(path,PATHS,img_shape, time, 0,pixel_dims)
    #fill img_interface with the info from the two other cells
    
    filt = cell1+cell2>1
    interface1 = np.where(filt==False, 0, cell1_0)
    interface2 = np.where(filt==False,0,cell2_0)
    interface3 = interface1 + interface2
    interface4 = np.where(interface3>=1, 255,interface3)
    
    filt_other = ~((cell1+cell2==0) + filt)
    other1 = np.where(filt_other==False, 0, cell1_0)
    other2 = np.where(filt_other==False,0,cell2_0)
    other3 = other1 + other2
    other = np.where(other3>=1, 255,other3)
    
    #fill up lumen part by first finding the contour of the lumen, then building a convex hull out of it and fill the interface with the corresponding data from RAW cell1_0,cell2_0
    
    start_npix_other = 5
    end_n_pix_other = 6
    step_npix_other = 1
    n_pixs_other =[]
    z_of_interests =[]
    for n_pix_other in range(start_npix_other, end_n_pix_other,step_npix_other):
        
        #print(n_pix_other)
        other_filled = np.copy(other)
        other_filled = label_region(other_filled,np.where(other==255), n_pix_other,255)
        z_of_interest=[]
        points_to_hull = np.array([])
        
        for z in range(0,other_filled.shape[0]):
            contours=measure.find_contours(other_filled[z],255/2)
            if len(contours)>=3:
                length_of_contours = (np.argsort([e.shape[0] for e in contours])[::-1])
                #the third contour corresponds to the lumen
                c = contours[length_of_contours[2]]
                p = np.vstack((z*np.ones(len(c)),c.T)).T
                z_of_interest.append(z)
                if len(points_to_hull)==0:
                    points_to_hull = p
                else:
                    points_to_hull = np.vstack((points_to_hull,p))   
        
        #z_of_interests.append(len(z_of_interest))
        # z_of_interests+=z_of_interest
        # n_pixs_other+=[n_pix_other]*len(z_of_interest)
    
    ### here we set a condition such that we compute the convex hull and modify the interface
    ### only if there are more than 3 zsteps involved
    
    if len(z_of_interest)>3:
        hull = ConvexHull(points_to_hull)
        deln = Delaunay(points_to_hull[hull.vertices]) 
        idx = np.stack(np.indices(other_filled.shape), axis = -1)
        lumen_idx = np.nonzero(deln.find_simplex(idx) + 1)
        lumen_img = np.zeros(other_filled.shape)
        lumen_img[lumen_idx] = True
        
        interface5 = np.where(lumen_img==True, (cell1_0+cell2_0)*255,interface4)
        interface = np.where(interface5>255,255,interface5)
    else:
        interface = interface4
        
    return(interface,other)
    #return(n_pixs_other,z_of_interests)
    #return(z_of_interests)

def label_interface_all_timepoints1(path,PATHS,img_shape,start_timepoint,n_pix_seg,n_pix_vis,pixel_dims):
    
    os.chdir(path)
    timepoints = img_shape[0]
    print('There are ', timepoints,'timepoints to segment !')

    img_shape = img_shape[1:]
    interface_tot = []
    other_tot = []
    for timepoint in range(10,20):
        print(timepoint)
        interface,other = label_interface1(path,PATHS,img_shape,timepoint+start_timepoint,n_pix_seg,pixel_dims)
        interface_tot.append(label_region(interface,np.where(interface==255), n_pix_vis,255))
        other_tot.append(label_region(other,np.where(other==255), n_pix_vis,255))
    
    return(np.array(interface_tot),np.array(other_tot))

def label_interface_all_timepoints2(path,PATHS,img_shape,start_timepoint,n_pix_seg,n_pix_vis,pixel_dims):
    
    os.chdir(path)
    timepoints = img_shape[0]
    print('There are ', timepoints,'timepoints to segment !')

    img_shape = img_shape[1:]
    interface_tot = []
    other_tot = []
    for timepoint in range(1,timepoints-1):
        print(timepoint)
        interface,other = label_interface2(path,PATHS,img_shape,timepoint+start_timepoint,n_pix_seg,pixel_dims)
        interface_tot.append(label_region(interface,np.where(interface==255), n_pix_vis,255))
        other_tot.append(label_region(other,np.where(other==255), n_pix_vis,255))
    
    return(np.array(interface_tot),np.array(other_tot))

def compute_interface_curvature_all_timepoints1(path,PATHS,img_shape,start_timepoint,n_pix_seg,n_pix_vis,pixel_dims):
    
    
    C=[]
    P=[]
    os.chdir(path)
    timepoints = img_shape[0]
    
    print('There are ', timepoints,'timepoints to segment !')

    img_shape = img_shape[1:]

    for timepoint in range(0,timepoints):
        print(timepoint)
        interface,other = label_interface1(path,PATHS,img_shape,timepoint+start_timepoint,n_pix_seg,pixel_dims)
        
        I = np.where(interface!=0)
        interface_points = np.array([[I[1][i],I[2][i],I[0][i]/pixel_dims[0]] for i in range(0,len(I[0]))])
    
        interface_pd = pd.DataFrame(interface_points, columns=['x','y','z'])
        interface_cloud = PyntCloud(interface_pd)
        
        k_neighbors = interface_cloud.get_neighbors(k=int(len(interface_pd)/5))
        eigenvalues = interface_cloud.add_scalar_field(
            "eigen_values", 
            k_neighbors=k_neighbors)
    
    
        curvature = interface_cloud.add_scalar_field("curvature", ev=eigenvalues)
        curv = interface_cloud.points[curvature]
        curv1 = np.std(curv)
        
        k_neighbors = interface_cloud.get_neighbors(k=int(len(interface_pd)/20))
        eigenvalues = interface_cloud.add_scalar_field(
            "eigen_values", 
            k_neighbors=k_neighbors)
        planarity = interface_cloud.add_scalar_field("planarity", ev=eigenvalues)
        planar = interface_cloud.points[planarity]
        planar1 = np.std(planar)
        curv1=np.mean(curv[planar<0.4])
        
        P.append(len(curv[planar<0.4])/len(curv))
        C.append(curv1)
    
    return(np.array(C),np.array(P))
    
def save_img(path,img_seg_tot,name,pixel_dims,time_int) :
    
    #img_seg_tot = np.swapaxes(img_seg_tot,0,1)
    # for labelling need int8 
    img_seg_tot = np.asarray(img_seg_tot,dtype = np.int16)
    #img_seg_tot = np.asarray(img_seg_tot,dtype = np.float32)
    img_seg_tot = np.reshape(img_seg_tot,(1,*img_seg_tot.shape))
    print(img_seg_tot.shape)
    img_seg_tot = np.swapaxes(img_seg_tot,0,1)
    img_seg_tot = np.swapaxes(img_seg_tot,1,2)
    os.chdir(path)
    #imwrite(name, img_seg_tot,imagej =True) gives the helix when 3D stack in Fiji :')
    imwrite(name, img_seg_tot, imagej=True, resolution=(1./pixel_dims[0], 1./pixel_dims[1]),metadata={'spacing': 1.0, 'unit': 'um', 'finterval': time_int,'axes': 'TZCYX'})
    #imsave(name, img_seg_tot, compress=0, resolution=[1/pixel_dims[0]*(2.54*10000), 1/pixel_dims[1]*(2.54*10000)], metadata={'axes': 'TZCYX','spacing': 1.0 },plugin='tifffile')
    print('I have saved the file '+name)
    return


def save_img_two_channels(path,img_seg_tot_1,img_seg_tot_2,name,pixel_dims,time_int) :
    
    
    img_seg_tot = np.concatenate((img_seg_tot_1,img_seg_tot_2))
    #img_seg_tot = np.swapaxes(img_seg_tot,0,1)
    # for labelling need int8 
    img_seg_tot = np.asarray(img_seg_tot,dtype = np.int16)
    #img_seg_tot = np.asarray(img_seg_tot,dtype = np.float32)
    img_seg_tot = np.reshape(img_seg_tot,(2,*img_seg_tot_1.shape))
    img_seg_tot = np.swapaxes(img_seg_tot,0,1)
    img_seg_tot = np.swapaxes(img_seg_tot,1,2)
    os.chdir(path)
    #imwrite(name, img_seg_tot,imagej =True) gives the helix when 3D stack in Fiji :')
    imwrite(name, img_seg_tot, imagej=True, resolution=(1./pixel_dims[0], 1./pixel_dims[1]),metadata={'spacing': 1.0, 'unit': 'um', 'finterval': time_int,'axes': 'TZCYX'})
    #imsave(name, img_seg_tot, compress=0, resolution=[1/pixel_dims[0]*(2.54*10000), 1/pixel_dims[1]*(2.54*10000)], metadata={'axes': 'TZCYX','spacing': 1.0 },plugin='tifffile')
    print('I have saved the file '+name)
    return

def save_img_two_channels_single_timepoint(path,img_seg_tot_1,img_seg_tot_2,name,pixel_dims,time_int) :
    
    
    img_seg_tot = np.concatenate((img_seg_tot_1,img_seg_tot_2))
    
    #img_seg_tot = np.swapaxes(img_seg_tot,0,1)
    # for labelling need int8 
    img_seg_tot = np.asarray(img_seg_tot,dtype = np.int16)
    #img_seg_tot = np.asarray(img_seg_tot,dtype = np.float32)
    img_seg_tot = np.reshape(img_seg_tot,(2,*img_seg_tot_1.shape))
    img_seg_tot = np.swapaxes(img_seg_tot,0,1)
    t_0 = np.zeros((1,*img_seg_tot.shape))
    t_0[0] = img_seg_tot
    t_0 = np.asarray(t_0,dtype = np.int16)
    os.chdir(path)
    #imwrite(name, img_seg_tot,imagej =True) gives the helix when 3D stack in Fiji :')
    imwrite(name, t_0, imagej=True, resolution=(1./pixel_dims[0], 1./pixel_dims[1]),metadata={'spacing': 1.0, 'unit': 'um', 'finterval': time_int,'axes': 'TZCYX'})
    #imsave(name, img_seg_tot, compress=0, resolution=[1/pixel_dims[0]*(2.54*10000), 1/pixel_dims[1]*(2.54*10000)], metadata={'axes': 'TZCYX','spacing': 1.0 },plugin='tifffile')
    print('I have saved the file '+name)
    return


def save_img_three_channels_single_timepoint(path,img_seg_tot,name,pixel_dims,time_int) :
    
    
    #img_seg_tot = np.concatenate((img_seg_tot_1,img_seg_tot_2))
    
    #img_seg_tot = np.swapaxes(img_seg_tot,0,1)
    # for labelling need int8 
    img_seg_tot = np.asarray(img_seg_tot,dtype = np.int16)
    #img_seg_tot = np.asarray(img_seg_tot,dtype = np.float32)
    #img_seg_tot = np.reshape(img_seg_tot,(2,*img_seg_tot_1.shape))
    img_seg_tot = np.swapaxes(img_seg_tot,0,1)
    t_0 = np.zeros((1,*img_seg_tot.shape))
    t_0[0] = img_seg_tot
    t_0 = np.asarray(t_0,dtype = np.int16)
    os.chdir(path)
    #imwrite(name, img_seg_tot,imagej =True) gives the helix when 3D stack in Fiji :')
    imwrite(name, t_0, imagej=True, resolution=(1./pixel_dims[0], 1./pixel_dims[1]),metadata={'spacing': 1.0, 'unit': 'um', 'finterval': time_int,'axes': 'TZCYX'})
    #imsave(name, img_seg_tot, compress=0, resolution=[1/pixel_dims[0]*(2.54*10000), 1/pixel_dims[1]*(2.54*10000)], metadata={'axes': 'TZCYX','spacing': 1.0 },plugin='tifffile')
    print('I have saved the file '+name)
    return


def compute_normals_to_cloud(cloud, portion):
    
    k_neighbors = cloud.get_neighbors(k=int(len(cloud.points['x'])/portion))
    eigenvalues = cloud.add_scalar_field(
    "eigen_values", 
    k_neighbors=k_neighbors)

    normals = cloud.add_scalar_field("normals", k_neighbors=k_neighbors)
    
    return(cloud,normals)

def find_center_of_cloud(interface_cloud, portion):
    
    xpoints = interface_cloud.points['x']
    ypoints = interface_cloud.points['y']
    zpoints = interface_cloud.points['z']
    xc,yc,zc=np.mean(xpoints),np.mean(ypoints),np.mean(zpoints)
    
    d = np.sqrt((xpoints-xc)**2+(ypoints-yc)**2+(zpoints-zc)**2)
    
    index = np.argsort(d)[0:int(len(xpoints)/portion)]
    
    return(index)

def scalar_product_normal(interface_cloud,portion):
    
    index = find_center_of_cloud(interface_cloud, 100)
    print('To compute center orientation npoints =', len(index))
    cloud_normals,normals = compute_normals_to_cloud(interface_cloud,portion)
    
    nx = cloud_normals.points[normals[0]]
    ny = cloud_normals.points[normals[1]]
    nz = cloud_normals.points[normals[2]]
    
    a = np.array([nx,ny,nz])
    
    #avg_normal_center = np.array([np.mean(nx[index]),np.mean(ny[index]),np.mean(nz[index])])
    avg_normal_center = np.array([nx[index[0]],ny[index[0]],nz[index[0]]])
    dot_product = np.abs(np.array([np.dot(a.T[i],avg_normal_center.T) for i in range(len(a.T))]))
    return(dot_product)

def generate_interface_cloud(interface,pixel_dims) : 
    
    I = np.where(interface!=0)
    interface_points = np.array([[I[1][i],I[2][i],I[0][i]/pixel_dims[0]] for i in range(0,len(I[0]))])
    if len(interface_points)>0:
        interface_pd = pd.DataFrame(interface_points, columns=['x','y','z'])
        interface_cloud = PyntCloud(interface_pd)
    else : 
        interface_cloud = []
    return(interface_cloud)

def compute_shape_index(path,PATHS,img_shape,timepoint,n_pix,pixel_dims):
    
    interface,other = label_interface1(path,PATHS,img_shape,timepoint,n_pix,pixel_dims)
    interface_cloud = generate_interface_cloud(interface, pixel_dims)
    if type(interface_cloud) == pyntcloud.core_class.PyntCloud:
        dot_product = scalar_product_normal(interface_cloud,30)
        if len(dot_product)>600 :
            a =  np.mean(1 - dot_product)
        else : 
            a = 0
    else:
        a = 0
    return(a)
        

##############################################################################

########### Compute shape of the interface ###################################

##############################################################################


def compute_x_plane_y_plane_H(interface_points):
    
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

    return(x_plane,y_plane)

def compute_x_plane_y_plane_H1(interface_points):
    
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

    return(x_plane,y_plane,H)

def plot_x_plane_y_plane_H(interface_points,name,save_path, save_plot=False):
    
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
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    
    fig, ax = plt.subplots(figsize=(19.20,10.80))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.scatter(x_plane-xc,y_plane-yc,c=H ,s = 60, vmin = -10, vmax=10)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-60,60])
    ax.set_ylim([-60,60])
    clb = fig.colorbar(im, cax=cax, orientation='vertical')
    clb.ax.tick_params(labelsize=8) 
    # clb.set_label('Your Label',fontsize=35, rotation=270)
    clb.ax.xaxis.set_tick_params(pad=130)

    # ax.set_xlabel(r'||$\vec{\omega}$|| (rotation.h$^{-1}$)', fontsize =35, labelpad = 12)
    # ax.set_ylabel(r'Average interface height ($\mu$m)', fontsize =35, labelpad = 12)
    ax.tick_params(axis='both', which='major', labelsize=35, pad=10)
    
    if save_plot == True:
        time = int(name.split('t = ')[-1])
        print(save_path)
        print(save_path+'interface_proj_saved/'+'interface_proj_t'+str(time)+'.jpg')
        fig.savefig(save_path+'interface_proj_t'+str(time)+'.jpg')
    #return(H/sh2)
    return()


def convex_area(x,y):
    
    points_to_hull = np.vstack((x,y)).T
    hull = ConvexHull(points_to_hull)
   
    return(hull.area)

def compute_xc_yc(x,y):
    
    points_to_hull = np.vstack((x,y)).T
    hull = ConvexHull(points_to_hull)
    cx = np.mean(hull.points[hull.vertices,0])
    cy = np.mean(hull.points[hull.vertices,1])
    
    return((cx,cy))

##############################################################################
#############             PLOT FUNCTIONS                 #####################
##############################################################################
def set_proper_aspect_ratio(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def get_colors(cloud, use_as_color, cmap):
    try:
        colors = cloud.points[use_as_color].values
    except KeyError:
        colors = None
    if use_as_color != ["red", "green", "blue"] and colors is not None:
        import matplotlib.pyplot as plt

        s_m = plt.cm.ScalarMappable(cmap=cmap)
        colors = s_m.to_rgba(colors)[:, :-1] * 255
    elif colors is None:
        # default color orange
        colors = np.repeat([[255, 125, 0]], cloud.xyz.shape[0], axis=0)
    return colors.astype(np.uint8)


def plot_cloud(cloud, table ,**kwargs):

    #colors = get_colors(cloud, kwargs["use_as_color"], kwargs["cmap"])
    colors = table
    
    ptp = cloud.xyz.ptp()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    
    if kwargs["lim_colorbar"] == 'limit_colorbar' :
        p = ax.scatter(
            cloud.xyz[:, 0],
            cloud.xyz[:, 1],
            cloud.xyz[:, 2],
            marker="D",
            c=colors,
            zdir="z",
            depthshade=True,
            s= ptp / 10, vmin=0, vmax=1,cmap=plt.cm.gist_rainbow)
    else :
       p = ax.scatter(
            cloud.xyz[:, 0],
            cloud.xyz[:, 1],
            cloud.xyz[:, 2],
            marker="D",
            c=colors ,
            zdir="z",
            depthshade=True,
            s= ptp / 10)
        
    set_proper_aspect_ratio(ax)
    fig.colorbar(p, ax=ax)


    return plt.show()