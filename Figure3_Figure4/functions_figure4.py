#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:12:57 2023

@author: vagne
"""

from numba import jit,njit
from numba.types import bool_
import useful_functions_interface as uf_int
import numpy as np
import os
import useful_functions_ply_files as uf
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from pyevtk.vtk import VtkTriangle, VtkQuad
from pyevtk.hl import unstructuredGridToVTK

import matplotlib.pyplot as plt 

import matplotlib.font_manager as font_manager

import functions_doublet as F

@njit
def bool_array_from_index_list(nvert,indices):
    result=np.zeros(nvert,dtype=bool_)
    for i in indices:
        result[i]=True
    return result


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

def create_cylinder(center,direction,L,n):
    #2*n+2  points
    V=np.zeros((2*n+2,3))
    #normalize direction
    ez=direction/np.linalg.norm(direction)
    #create (ex,ey,ez) with ez the direction of the cylinder
    #ex is creating by doing ez x u with u =[1,0,0] or [0,1,0] or [0,0,1] depending on ez
    u=np.zeros(3)
    u[np.argmin(np.abs(ez))]=1
    ex=np.cross(ez,u)
    ey=np.cross(ez,ex)
    #array of angles theta
    Theta=np.linspace(0,(n-1)*2*np.pi/n,n)
    #create vertices coordinates
    V[0:n,:] = center[None,:] + L/2*ez[None,:] + np.outer(np.cos(Theta),ex)+np.outer(np.sin(Theta),ey)
    V[n:(2*n),:] = center[None,:] - L/2*ez[None,:] + np.outer(np.cos(Theta),ex)+np.outer(np.sin(Theta),ey)
    V[2*n,:] = center+L/2*ez
    V[2*n+1,:] = center-L/2*ez
    #create triangles (2*n for sides, 2*n for caps)
    T=np.zeros((4*n,3),dtype=int)
    for i in range(n):
        T[i,0]=i
        T[i,1]=(i+1)%n
        T[i,2]=n+(i+1)%n
        T[n+i,0]=i
        T[n+i,1]=n+i
        T[n+i,2]=n+(i+1)%n
        #upper cap
        T[2*n+i,0]=2*n
        T[2*n+i,1]=i
        T[2*n+i,2]=(i+1)%n
        #lower cap
        T[3*n+i,0]=2*n+1
        T[3*n+i,1]=n+i
        T[3*n+i,2]=n+(i+1)%n
    return V,T
    
    
    

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
    mean_before=0
    max_before=0
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
        
        mean_before = mean_before + value_signal
        if value_signal > max_before:
            max_before=value_signal
        
        # value_signal = v/n
        if interface_v[k] == 0:
            diff = value_signal - avg_signal_cell
            values[k] = diff
        else :
            diff = (value_signal - (avg_signal_cell+avg_signal_other_cell)/2)/2
            values[k] = diff
    mean_before = mean_before / len(pix)
           
    return(values,mean_before,max_before)

def polar_hist(data,angle):
    
    N = data[0].shape[0]
    bottom = 1
    max_height = 4
    
    theta = (data[1][1:]+data[1][:-1])/2 #bins centers in beta
    radii = max_height*data[0]/np.max(data[0]) #beta counts
    width = (2*np.pi) / N
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    bars = ax.bar(theta, radii, width=width, bottom=bottom)
    
    ax.plot(np.array([0,angle]),np.array([0,max_height]),'r-')
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=5, pad=-1)
    return fig
    
    

def exponential_calibration(img,scale_factor,l):
    #applies an exponential correction exp(z/l) to the raw values in img
    img_resc=np.zeros(img.shape)
    
    # reorder pixel positions to be [z, rows, cols]
    # pix are the  real coordinates of the vertices XYZ in units of ImageJ 
    # but WARNING img = imread(img_name) you give youi img as [z][y][x] 
    
    z=np.arange(0,img.shape[0])*scale_factor
    expcor=np.exp(z/l)
    
    img_resc=img*expcor[:,None,None] #term by term multiplication,first dim is z
    
    return img_resc

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

@njit
def make_histo(xyz_inside, im_mrlc):
    N=xyz_inside.shape[0]
    histo=np.zeros(N)
    for k in range(N):
        z = int(xyz_inside[k,0])
        x = int(xyz_inside[k,1])
        y = int(xyz_inside[k,2])
        histo[k] = im_mrlc[z][x][y]
    return histo

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
    xyz_inside = xyz_cell[distances>3*scale_factor] #was originally at 20 pixel (regardless of scale factor), now we want 3um consistently.
    #print('detected pixels at more than 3um:',xyz_inside.shape[0])
    avg_inside_signal = avg_signal(xyz_inside, im_mrlc)
    #histogram inside the cell
    #histo = make_histo(xyz_inside,im_mrlc)
    
    return(avg_inside_signal)

    

def get_names_ply(path2):

    A = np.array(os.listdir(path=path2))
    B = np.array([file.endswith(".ply") for file in A])
    A = A[B]
    return(A)

def get_tricentres(vertices,triangles):
    u1 = vertices[triangles[:, 0], :]
    u2 = vertices[triangles[:, 1], :]
    u3 = vertices[triangles[:, 2], :]
        
    return (u1+u2+u3)/3.0
    

def calc_centre_of_mass(vertices,triangles):
    
    u1 = vertices[triangles[:, 0], :]
    u2 = vertices[triangles[:, 1], :]
    u3 = vertices[triangles[:, 2], :]
        
    tri_centres = (u1+u2+u3)/3.0
        
    # calculate area of each triangle
    v1 = u2 - u1
    v2 = u3 - u1
    cross = np.cross(v1, v2)
    areas = 0.5*np.sqrt(cross[:, 0]**2+cross[:, 1]**2+cross[:, 2]**2)
    total_area = areas.sum()
        
    # calculate sum of triangle centres, weighted by area, divided by total area
    weighted_centres = np.zeros_like(tri_centres)
    weighted_centres[:, 0] = tri_centres[:, 0]*areas
    weighted_centres[:, 1] = tri_centres[:, 1]*areas
    weighted_centres[:, 2] = tri_centres[:, 2]*areas
    com = weighted_centres.sum(axis=0)/total_area
    
    return(com)

def get_series_com(cells_path):
    
    startpoint,endpoint = uf_int.find_startpoint_endpoint(cells_path[0])

    com_cell1 = np.zeros((endpoint-startpoint+1,3))
    com_cell2 = np.zeros((endpoint-startpoint+1,3))
    for path in cells_path:
        filenames = get_names_ply(path)
        for file in filenames:
            t = int(file.split('time')[1].split('_')[0])
            mesh,V,T = uf.get_vertices_triangles(path+file)
            com_cell = uf.calc_centre_of_mass(V,T)
            
            if 'cell_1' in file: 
                com_cell1[t-startpoint,:] = com_cell
                
            else :
                com_cell2[t-startpoint,:] = com_cell
                    
    return(np.array(com_cell1),np.array(com_cell2))

def calc_volume(com, vertices, triangles):
    
    centre = calc_centre_of_mass(vertices, triangles)
    volume = 0.0
        
    # calculate volume of each tetrahedron
    a = vertices[triangles[:, 0], :]-centre
    b = vertices[triangles[:, 1], :]-centre
    c = vertices[triangles[:, 2], :]-centre
    b_cross_c = np.cross(b, c)
    triple_prod_abc = a[:, 0]*b_cross_c[:, 0] + a[:, 1]*b_cross_c[:, 1] + a[:, 2]*b_cross_c[:, 2]
    tetra_vol = np.abs(triple_prod_abc)/6.0

    volume = tetra_vol.sum()
    
    return(volume)

def get_series_volumes(cells_path):
    
    startpoint,endpoint = uf_int.find_startpoint_endpoint(cells_path[0])

    vol_cell1 = np.zeros((endpoint-startpoint+1))
    vol_cell2 = np.zeros((endpoint-startpoint+1))
    for path in cells_path:
        filenames = get_names_ply(path)
        for file in filenames:
            t = int(file.split('time')[1].split('_')[0])
            mesh,V,T = uf.get_vertices_triangles(path+file)
            com_cell = uf.calc_centre_of_mass(V,T)
            
            if 'cell_1' in file: 
                vol_cell1[t-startpoint] = calc_volume(com_cell, V, T)
                
            else :
                vol_cell2[t-startpoint] = calc_volume(com_cell, V, T)
                    
    return(np.array(vol_cell1),np.array(vol_cell2))




def get_xyz_polarity(path,fname):
    x_y_z = np.genfromtxt(path+fname, delimiter=',')
    x_y_z = x_y_z[1:,1:-1] 
    
    return(x_y_z)

def normalize_vector(vec):
    
    norm_vec = np.copy(vec)
    for i,v in enumerate(vec):
        norm_vec[i] = v/np.linalg.norm(v)
    return(norm_vec)

def scalar_product(a,b):
    scal = np.zeros(len(a))
    for i,e in enumerate(a):
        scal[i] = np.dot(e,b[i])
        
    return(scal)

def get_angular_distribution_saddle(vertices,center,signal_value,S12,N12):
    #center is the center of the interface
    ex=np.copy(S12)
    ex/= np.linalg.norm(ex)
    
    ez=np.copy(N12)
    ez/= np.linalg.norm(ez)
    
    ey = np.cross(ez, ex)
    
    v = vertices - center
    v = normalize_vector(v)
    
    theta = np.arccos(np.dot(v,ez))
    temp = np.dot(v,ez)[:,None]*ez[None,:]
    vproj = v - temp
    
    x = np.dot(vproj, ex)
    y = np.dot(vproj, ey)
    
    
    phi = np.arctan2(y,x)

    
    return(theta,phi)


def get_angular_distribution_polarity(vertices,triangles, signal_value, r12, polarity):
    
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    polarity_norm = polarity/np.linalg.norm(polarity) 
    ex = np.cross(polarity_norm, r12)
    ex/= np.linalg.norm(ex)
    
    ey = np.cross(polarity_norm, ex)
    ez = np.copy(polarity_norm)
    
    v = vertices - com_cell
    v = normalize_vector(v)
    
    theta = np.arccos(np.dot(v,ez))
    temp = np.dot(v,ez)[:,None]*ez[None,:]
    vproj = v - temp
    
    x = np.dot(vproj, ex)
    y = np.dot(vproj, ey)
    
    
    phi = np.arctan2(y,x)

    
    return(theta,phi)


def get_angular_distribution_r12_omega(vertices,triangles, signal_value, r12, n1):
    
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    ez = np.copy(r12)
    ez /= np.linalg.norm(ez)
    
    ex = np.copy(n1)
    ex /= np.linalg.norm(ex)
    
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    
    v = vertices - com_cell
    v = normalize_vector(v)
    
    theta = np.arccos(np.dot(v,ez))
    temp = np.dot(v,ez)[:,None]*ez[None,:]
    vproj = v - temp
    
    x = np.dot(vproj, ex)
    y = np.dot(vproj, ey)
    
    
    phi = np.arctan2(y,x)

    
    return(theta,phi)




def get_angular_distribution_yin_yang(vertices,triangles, signal_value, polarity, r12):
    
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    polarity /= np.linalg.norm(polarity) 
    ex = np.cross(polarity, r12)
    ex/= np.linalg.norm(ex)
    
    ey = np.cross(polarity, ex)
    ez = np.copy(polarity)
    
    v = vertices - com_cell
    v = normalize_vector(v)
    
    theta = np.arccos(np.dot(v,ez))
    temp = np.dot(v,ez)[:,None]*ez[None,:]
    vproj = v - temp
    
    x = np.dot(vproj, ex)
    y = np.dot(vproj, ey)
    
    
    phi = np.arctan2(y,x)

    
    return(theta,phi)

def get_angular_maps_interpolation(theta_v, phi_v, signal_values, theta_phi_final):
    
    ### signal_values is an array as vertices with intensity of a given signal of interest
    
    
    Ntheta = theta_phi_final.shape[0]
    Nphi = theta_phi_final.shape[1]
    
    dtheta = np.pi/Ntheta
    dphi = 2*np.pi/Nphi
    
    theta = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    x = np.outer(np.sin(theta), np.cos(phi)).flatten()
    y = np.outer(np.sin(theta), np.sin(phi)).flatten()
    z = np.outer(np.cos(theta), np.ones(np.size(phi))).flatten()
    
    x_v = np.sin(theta_v)*np.cos(phi_v)
    y_v = np.sin(theta_v)*np.sin(phi_v)
    z_v = np.cos(theta_v)
    
    xyz = np.stack((x,y,z), axis = 1)
    xyz_v = np.stack((x_v,y_v,z_v), axis = 1)
    
    tree = cKDTree(xyz_v)
    distances,indices = tree.query(xyz, k = 1)
    
    ind_theta = np.outer(np.arange(0,Ntheta), np.ones(np.size(phi))).flatten().astype(int)
    ind_phi = np.outer(np.ones(np.size(theta)),np.arange(0,Nphi)).flatten().astype(int)

    
    for i in range(len(ind_theta)):

        theta_phi_final[ind_theta[i], ind_phi[i]] += signal_values[indices[i]]
    
    
    return()

@jit
def bin_angles(theta_int, phi_int, signal_value, theta_phi_final, avg_signal):
    
    Ntheta = theta_phi_final.shape[0]
    Nphi = theta_phi_final.shape[1]
    
    dtheta = np.pi/Ntheta
    dphi = 2*np.pi/Nphi
    count = 0
    for i in range(Ntheta):
        for j in range(Nphi):
            
            ind_t = np.logical_and(theta_int>=i*dtheta, theta_int<(i+1)*dtheta)
            ind_p = np.logical_and(phi_int>=-np.pi+j*dphi, phi_int<-np.pi+(j+1)*dphi)
            ind = np.logical_and(ind_t, ind_p)
            
            if np.sum(ind) > 0.0:
                signal_ind = np.mean(signal_value[ind])
                
                theta_phi_final[i,j] += signal_ind
            else :
                theta_phi_final[i,j] += avg_signal
                count+=1
    print(count/Ntheta/Nphi)
    return()


def fill_integral_triangles(t1, t2, a, b, u1) : 
    
    int_triangle = 1/24*(2*t1[:,a]*t1[:,b] + t1[:,b]*t2[:,a] + 4*u1[:,b]*(t1[:,a]+ t2[:,a]) + t1[:,a]*t2[:,b] + 2*t2[:,a]*t2[:,b] + 4*u1[:,a]*(3*u1[:,b] + t1[:,b] + t2[:,b]))
    
    
    ds = np.linalg.norm(np.cross(t1,t2))
    
    return(ds * int_triangle)
    
def Q_tensor3d_myosin_triangles(vertices, triangles, signal):


    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    

    X = vertices[:,0]
    Y = vertices[:,1]
    Z = vertices[:,2]
    
    Xc=X-com_cell[0]
    Yc=Y-com_cell[1]
    Zc=Z-com_cell[2]
    
    norm = np.sqrt((Xc**2+Yc**2+Zc**2))
    
    Xc=Xc/norm
    Yc=Yc/norm
    Zc=Zc/norm
    
    new_vertices = np.stack((Xc,Yc,Zc), axis = 1)
    
    u1 = new_vertices[triangles[:, 0], :]
    u2 = new_vertices[triangles[:, 1], :]
    u3 = new_vertices[triangles[:, 2], :]

    tri_centres = (u1+u2+u3)/3.0

    signal_tri_avg = (signal[triangles[:, 0]] + signal[triangles[:, 1]] + signal[triangles[:, 2]])/3.0
    
    t1 = u2-u1
    t2 = u3-u1
    
    # calculate area of each triangle

    cross = np.cross(t1, t2)
    areas = 0.5*np.sqrt(cross[:, 0]**2+cross[:, 1]**2+cross[:, 2]**2)
    
    integrated_signal = np.sum(areas*signal_tri_avg)
    
    int_triangle = np.zeros((len(t1), 3, 3))
    
    int_triangle[:, 0, 0] = fill_integral_triangles(t1, t2, 0, 0, u1)
    int_triangle[:, 0, 1] = fill_integral_triangles(t1, t2, 0, 1, u1)
    int_triangle[:, 0, 2] = fill_integral_triangles(t1, t2, 0, 2, u1)
    int_triangle[:, 1, 1] = fill_integral_triangles(t1, t2, 1, 1, u1)
    int_triangle[:, 1, 2] = fill_integral_triangles(t1, t2, 1, 2, u1)
    int_triangle[:, 2, 2] = fill_integral_triangles(t1, t2, 2, 2, u1)
    
    Q=np.zeros((3,3))
        
    Q[0,0] = np.sum(int_triangle[:,0,0]*signal_tri_avg)/integrated_signal
    Q[0,1] = np.sum(int_triangle[:,0,1]*signal_tri_avg)/integrated_signal
    Q[0,2] = np.sum(int_triangle[:,0,2]*signal_tri_avg)/integrated_signal
    Q[1,1] = np.sum(int_triangle[:,1,1]*signal_tri_avg)/integrated_signal
    Q[1,2] = np.sum(int_triangle[:,1,2]*signal_tri_avg)/integrated_signal
    Q[2,2] = np.sum(int_triangle[:,2,2]*signal_tri_avg)/integrated_signal
    
    tr=Q[0,0]+Q[1,1]+Q[2,2]
    Q[0,0]=Q[0,0]-tr/3.0
    Q[1,1]=Q[1,1]-tr/3.0
    Q[2,2]=Q[2,2]-tr/3.0

    Q[1,0]=Q[0,1]
    Q[2,0]=Q[0,2]
    Q[2,1]=Q[1,2]

    return (Q)

def Q_tensor3d_myosin_simple(vertices, triangles, signal):


    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    

    X = vertices[:,0]
    Y = vertices[:,1]
    Z = vertices[:,2]
    
    Xc=X-com_cell[0]
    Yc=Y-com_cell[1]
    Zc=Z-com_cell[2]
    
    norm = np.sqrt((Xc**2+Yc**2+Zc**2))
    
    Xc=Xc/norm
    Yc=Yc/norm
    Zc=Zc/norm
    
    new_vertices = np.stack((Xc,Yc,Zc), axis = 1)
    
    u1 = new_vertices[triangles[:, 0], :]
    u2 = new_vertices[triangles[:, 1], :]
    u3 = new_vertices[triangles[:, 2], :]

    tri_centres = (u1+u2+u3)/3.0

    signal_tri_avg = (signal[triangles[:, 0]] + signal[triangles[:, 1]] + signal[triangles[:, 2]])/3.0
    
    t1 = u2-u1
    t2 = u3-u1
    
    # calculate area of each triangle

    cross = np.cross(t1, t2)
    areas = 0.5*np.sqrt(cross[:, 0]**2+cross[:, 1]**2+cross[:, 2]**2)
    
    integrated_signal = np.sum(areas*signal_tri_avg)
    
    
    Q=np.zeros((3,3))
        
    Q[0,0] = np.sum(tri_centres[:,0]*tri_centres[:,0]*signal_tri_avg*areas)/integrated_signal
    Q[0,1] = np.sum(tri_centres[:,0]*tri_centres[:,1]*signal_tri_avg*areas)/integrated_signal
    Q[0,2] = np.sum(tri_centres[:,0]*tri_centres[:,2]*signal_tri_avg*areas)/integrated_signal
    Q[1,1] = np.sum(tri_centres[:,1]*tri_centres[:,1]*signal_tri_avg*areas)/integrated_signal
    Q[1,2] = np.sum(tri_centres[:,1]*tri_centres[:,2]*signal_tri_avg*areas)/integrated_signal
    Q[2,2] = np.sum(tri_centres[:,2]*tri_centres[:,2]*signal_tri_avg*areas)/integrated_signal

    
    tr=Q[0,0]+Q[1,1]+Q[2,2]
    Q[0,0]=Q[0,0]-tr/3.0
    Q[1,1]=Q[1,1]-tr/3.0
    Q[2,2]=Q[2,2]-tr/3.0

    Q[1,0]=Q[0,1]
    Q[2,0]=Q[0,2]
    Q[2,1]=Q[1,2]

    return (Q)

def Q_2d(theta_bin):
    #create 2D Q tensor from an array of signal on a circle (bins from -pi to pi)
    nbin=theta_bin.shape[0]
    tharray=np.linspace(-np.pi+np.pi/nbin,np.pi-np.pi/nbin,nbin)
    X=np.cos(tharray)
    Y=np.sin(tharray)
    
    int_signal=np.sum(theta_bin)
    
    Q=np.zeros((2,2))
    
    Q[0,0]=np.sum(X*X*theta_bin)/int_signal
    Q[1,1]=np.sum(Y*Y*theta_bin)/int_signal
    Q[0,1]=np.sum(X*Y*theta_bin)/int_signal
    Q[1,0]=Q[0,1]
    tr=Q[0,0]+Q[1,1]
    Q[0,0]=Q[0,0]-tr/2.0
    Q[1,1]=Q[1,1]-tr/2.0
    
    return Q

def Q_2d_vec(v):
    Q=np.zeros((2,2))
    
    Q[0,0]=v[0]*v[0]
    Q[1,1]=v[1]*v[1]
    Q[0,1]=v[0]*v[1]
    Q[1,0]=Q[0,1]
    tr=Q[0,0]+Q[1,1]
    Q[0,0]=Q[0,0]-tr/2.0
    Q[1,1]=Q[1,1]-tr/2.0
    
    return Q

def Q_2d_2vec(v1,v2):
    Q=np.zeros((2,2))
    
    Q[0,0]=v1[0]*v2[0]+v2[0]*v1[0]
    Q[1,1]=v1[1]*v2[1]+v2[1]*v1[1]
    Q[0,1]=v1[0]*v2[1]+v2[0]*v1[1]
    Q[1,0]=Q[0,1]
    tr=Q[0,0]+Q[1,1]
    Q[0,0]=Q[0,0]-tr/2.0
    Q[1,1]=Q[1,1]-tr/2.0
    
    return Q

def pol_2d(theta_bin):
    #create polarity
    nbin=theta_bin.shape[0]
    tharray=np.linspace(-np.pi+np.pi/nbin,np.pi-np.pi/nbin,nbin)
    X=np.cos(tharray)
    Y=np.sin(tharray)
    
    int_signal=np.sum(theta_bin)
    
    p=np.zeros(2)
    p[0]=np.sum(X*theta_bin)/int_signal
    p[1]=np.sum(Y*theta_bin)/int_signal
    
    return p

def create_Q_from_vec(vec):
    
    Q=np.zeros((3,3))
    
    norm = np.linalg.norm(vec)
    
    Xc = vec[0]/norm
    Yc = vec[1]/norm
    Zc = vec[2]/norm
    
    Q[0,0] = Xc*Xc
    Q[0,1] = Xc*Yc
    Q[0,2] = Xc*Zc
    Q[1,1] = Yc*Yc
    Q[1,2] = Yc*Zc
    Q[2,2] = Zc*Zc
    
    tr=Q[0,0]+Q[1,1]+Q[2,2]
    Q[0,0]=Q[0,0]-tr/3.0
    Q[1,1]=Q[1,1]-tr/3.0
    Q[2,2]=Q[2,2]-tr/3.0

    Q[1,0]=Q[0,1]
    Q[2,0]=Q[0,2]
    Q[2,1]=Q[1,2]
    
    return(Q*np.sqrt(3/2))

def create_Qs_from_2vec(vp,vn):
    #vp,vn are two unit orthogonal vectors in the plane of the interface
    #vp is the one pointing towards a "positive" part of the saddle node (closer to cell 2)
    #vn is pointing towards a "negative" part (closer to cell 1)
    
    norm=np.linalg.norm(vp)
    X=np.zeros(3)
    X=vp/norm
    
    norm=np.linalg.norm(vn)
    Y=np.zeros(3)
    Y=vn/norm
    
    Q=np.zeros((3,3))
    Q[0,0]=X[0]*X[0]-Y[0]*Y[0]
    Q[0,1]=X[0]*X[1]-Y[0]*Y[1]
    Q[0,2]=X[0]*X[2]-Y[0]*Y[2]
    Q[1,1]=X[1]*X[1]-Y[1]*Y[1]
    Q[1,2]=X[1]*X[2]-Y[1]*Y[2]
    Q[2,2]=X[2]*X[2]-Y[2]*Y[2]
    #already traceless by definition and if X and Y are units and orhtogonal, the norm is sqrt(2)
    Q[1,0]=Q[0,1]
    Q[2,0]=Q[0,2]
    Q[2,1]=Q[1,2]
    return (Q/np.sqrt(2))


def plot_on_a_sphere(theta_phi, vmin, vmax): 
    
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    Ntheta = theta_phi.shape[0]
    Nphi = theta_phi.shape[1]
    
    
    v = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)
    u = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    # create the sphere surface
    x=10 * np.outer(np.cos(u), np.sin(v))
    y=10 * np.outer(np.sin(u), np.sin(v))
    z=10 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # simulate heat pattern (striped)
    # vmin = np.min(theta_phi.flatten())
    # vmax = np.max(theta_phi.flatten())
    myheatmap = (np.copy(theta_phi.T)-vmin)/(vmax-vmin)
    
    ax.plot_surface(x, y, z, cstride=1, rstride=1, facecolors=cmap.hot(myheatmap))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
            
    return()

def plot_on_a_sphere_complete(theta_phi, vmin, vmax):
    
    
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # fig = plt.subplots(figsize=(4.5*cm,3.28*cm))
    # ax = fig.gca(projection='3d')

    r=1 # radius of sphere
    
    theta = np.linspace(0, np.pi, Ntheta+1)
    phi = np.linspace(-np.pi, np.pi, Nphi+1)
    
    myheatmap = (np.copy(theta_phi).T-vmin)/(vmax-vmin)


    verts2 = []
    cols = []
    for i  in range(len(phi)-1):
        for j in range(len(theta)-1):
            
            cp0= r*np.cos(phi[i])
            cp1= r*np.cos(phi[i+1])
            sp0= r*np.sin(phi[i])
            sp1= r*np.sin(phi[i+1])

            ct0= np.cos(theta[j])
            ct1= np.cos(theta[j+1])

            st0= r*np.sin(theta[j])
            st1= r*np.sin(theta[j+1])

            verts=[]
            verts.append((cp0*st0, sp0*st0, ct0))
            verts.append((cp1*st0, sp1*st0, ct0))
            verts.append((cp1*st1, sp1*st1, ct1))
            verts.append((cp0*st1, sp0*st1, ct1))
            verts2.append(verts)
            
            cols.append(cmap.gist_gray(myheatmap[i,j]))
            
    poly3= Poly3DCollection(verts2, facecolor= cols, edgecolors=cols)  

    poly3.set_alpha(1.0)
    
    ax.add_collection3d(poly3)
    ax.set_xlabel('X')
    ax.set_xlim3d(-1, 1)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, 1)
    ax.set_zlabel('Z')
    ax.set_zlim3d(-1, 1)
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=25., azim=45)

    figname = 'plot_on_a_sphere_polar_map'
    #fig.savefig(save_path+figname+polarity_name)
    
    return()
    


def plot_maps(maps1, maps2):
    
    Nx, Ny = maps1.shape
    
    fig, ax = plt.subplots(1,2)
    
    ax[0].imshow(maps1, interpolation = 'nearest', cmap=plt.cm.gist_gray)
    ax[0].contour(maps2>0.5, 8, colors = 'red')
    
    x = [r'-$\pi$', '0', r'$\pi$']
    lx = [0, Ny/2, Ny-1]
    
    ax[0].set_xticks(lx)
    ax[0].set_xticklabels(x)
    
    y = ['0', r'$\frac{\pi}{2}$', r'$\pi$']
    ly = [0, Nx/2, Nx-1]
    
    ax[0].set_yticks(ly)
    ax[0].set_yticklabels(y)
    
    ax[0].set_xlabel(r'$\phi$', fontsize = 14)
    ax[0].set_ylabel(r'$\theta$', fontsize = 14)
    ax[0].tick_params(axis='both', which='major', labelsize=14, pad=2)
    
    ax[1].imshow(maps2)
    
    return()


def func_fit(theta, tension_mod, b):
    
    return(1.0 + tension_mod*((b*np.exp(b*np.cos(theta))-np.sinh(b))/(b*np.exp(b)-np.sinh(b))))

def get_tension_mod_fit(Ntheta, Nphi, theta_phi):
    
    theta = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    int_phi = np.sum(theta_phi, axis = 1)*2*np.pi/Nphi
    
    int_signal = np.sum(np.sin(theta)*int_phi)*np.pi/Ntheta
    
    popt, pcov = curve_fit(func_fit, theta, 2*int_phi/int_signal, bounds=([0,0.001], [10,100]))
    
    return(popt[0],popt[1], int_signal)

def get_std_over_avg(Ntheta, Nphi, theta_phi):
    #we compute the standard deviation of the signal divided by the average
    #ASSUMES positive average
    #the formula is sqrt(<x^2>/(<x>^2)-1)
    
    theta = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    int_phi = np.sum(theta_phi, axis = 1)*2*np.pi/Nphi
    int_phi2 = np.sum(theta_phi**2, axis = 1)*2*np.pi/Nphi
    
    int_signal = np.sum(np.sin(theta)*int_phi)*np.pi/Ntheta
    int_signal2 = np.sum(np.sin(theta)*int_phi2)*np.pi/Ntheta
    
    if int_signal <= 0:
        print("get_std_over_avg: warning: negative average for signal")
    
    avg_s=int_signal/(4*np.pi)
    avg_s2=int_signal2/(4*np.pi)
    
    restmp=avg_s2/(avg_s**2)
    
    if restmp < 1.0:
        return 0
    else:
        return np.sqrt(restmp-1.0)
    
    
    
    


def normalize_theta_phi_signal(all_theta_phi):
    
    Ntheta, Nphi = all_theta_phi[0].shape
    ang = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    all_theta_phi_norm = []
        
    for theta_phi in all_theta_phi :
        
        int_phi = np.sum(theta_phi, axis = 1)*2*np.pi/Nphi
        
        int_signal = np.sum(np.sin(ang)*int_phi)*np.pi/Ntheta
        
        if int_signal < 0 :
            print('''C'EST PAS BIEN ! FALLAIT PAS LE FAIRE !''')
    
        all_theta_phi_norm.append(4*np.pi*theta_phi/int_signal)
        
    return(all_theta_phi_norm)

def mean_along_phi(all_theta_phi):
    
    Ntheta, Nphi = all_theta_phi[0].shape
    ang = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
        
    avg_along_phi = []
    
    for theta_phi in all_theta_phi :
        
        int_phi = np.sum(theta_phi, axis = 1)*2*np.pi/Nphi
        
        int_signal = np.sum(np.sin(ang)*int_phi)*np.pi/Ntheta
        
        if int_signal < 0 :
            print('''C'EST PAS BIEN ! FALLAIT PAS LE FAIRE !''')
            
        avg_along_phi.append(2*int_phi/int_signal)
        
    return(avg_along_phi)

def polarity_from_sphere(ang_map_theta_phi):
    
    Ntheta, Nphi = ang_map_theta_phi.shape
    
    dtheta = np.pi/Ntheta
    dphi = 2*np.pi/Nphi
    
    theta = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    int_phi = np.sum(ang_map_theta_phi, axis = 1)*2*np.pi/Nphi
    
    int_signal = np.sum(np.sin(theta)*int_phi)*np.pi/Ntheta
    
    
    px_phi = np.sum(np.cos(phi)[None,:]*ang_map_theta_phi, axis = 1)*dphi
    py_phi = np.sum(np.sin(phi)[None,:]*ang_map_theta_phi, axis = 1)*dphi 
    pz_phi = np.sum(ang_map_theta_phi, axis = 1)*dphi 
    
    #CAREFUL: There is a sin(theta) Or cos(theta) coming from the vector Delta r / | Delta r |
    #but there is also sin(theta) from the surface element 
    px = np.sum(np.sin(theta)*px_phi*np.sin(theta))*dtheta/int_signal
    py = np.sum(np.sin(theta)*py_phi*np.sin(theta))*dtheta/int_signal
    pz = np.sum(np.cos(theta)*pz_phi*np.sin(theta))*dtheta/int_signal
    
    p = np.array([px, py, pz])
    p = p/np.linalg.norm(p)
    
    theta_p = np.arccos(p[2])/dtheta
    phi_p = (np.pi + np.arctan2(p[1],p[0]))/dphi
    
    p = np.array([0,-1,0])
    
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones(np.size(phi)))
    
    ct = x*p[0][None, None]+y*p[1][None, None] + z*p[2][None, None]
        
    ac = np.arccos(ct)
    ac = ac.flatten()
    values = np.copy(ang_map_theta_phi).flatten()
    I_theta = np.copy(theta)
    for i in range(Ntheta):
        ind_t = np.logical_and(ac>=i*dtheta, ac<(i+1)*dtheta)
        
        I_theta[i] = np.mean(values[ind_t])
    
    return(I_theta, theta_p, phi_p)
    
def naive_fit_plane_oriented(interface_vertices, u1):
    
    ## return [ex, ey, Nint], [X, Y, H] in a set of coordinates where Nint is 
    ## same direction as u1 (vector from cell com1 to cell com2)
    
    Cint=np.mean(interface_vertices,axis=0)
    svd=np.linalg.svd((interface_vertices-Cint).T)
    #the last vector is the normal vector of the plane
    Nint=svd[0][:,-1]
    
    #we use the first and second vectors as ex and ey axis in the plane
    #they are normalized already, so let's compute coordinates and height
    ex=svd[0][:,0]
    ey=svd[0][:,1]
    
    
    sign = np.dot(Nint, u1)
    
    if sign < 0 :
        Nint = -Nint
        ex = -ex
    
    H=np.dot(interface_vertices-Cint,Nint)
    X=np.dot(interface_vertices-Cint,ex)
    Y=np.dot(interface_vertices-Cint,ey)
    
    A=np.array([X*0+1,X,Y,X**2,Y**2,X*Y,X**3,X**2*Y,X*Y**2,Y**3]).T
    coeff, r, rank,s=np.linalg.lstsq(A,H)
    
    Hfit=coeff[0]+X*coeff[1]+Y*coeff[2]+X**2*coeff[3]+Y**2*coeff[4]+X*Y*coeff[5]
    Hfit=Hfit+coeff[6]*X**3+coeff[7]*X**2*Y+coeff[8]*X*Y**2+coeff[9]*Y**3
    
    
    return([ex, ey, Nint], [X,Y,Hfit])



def plot_polar_map(all_polar_maps, doublet_number, cell_number, timepoint):
    
    
    '''
    - timepoint like the ones in Fiji (starting from 1)
    - doublet number from 1 to 12
    - cell number is either 1 or 2 
    '''
    
    angular_map_pol = all_polar_maps[2*(doublet_number-1)+(cell_number-1)]
    Ntheta, Nphi = angular_map_pol[0].shape
    angular_map_pol_norm, avg_along_phi_pol_ = normalize_theta_phi_signal(Ntheta, Nphi, angular_map_pol)
    # angular_map_pol_mean = np.mean(angular_map_pol_norm, axis = 0)
    
    polar_map = angular_map_pol_norm[timepoint-1]
    Nx, Ny = polar_map.shape

    
    #compute vmin, vmax of the doublet:
    maps_doublet = all_polar_maps[2*(doublet_number-1)]+all_polar_maps[2*(doublet_number-1)+1]
    maps_doublet_norm, maps_doublet_avg_along_phi = normalize_theta_phi_signal(Ntheta, Nphi, maps_doublet)
    min_s = np.min(maps_doublet_norm)
    max_s = np.max(maps_doublet_norm)
    print(min_s, max_s)
    fig09, ax09 = plt.subplots(figsize=(6.0*cm,4*cm))

    ax09.imshow(polar_map, interpolation = 'nearest', cmap=plt.cm.gist_gray, vmin = min_s, vmax = max_s)

    # CS = ax08.contour(angular_map_interface_mean>0.5, 0, colors = 'red',linewidths=2, alpha = 1)
    x = [r'-$\pi$', '0', r'$\pi$']
    lx = [0, Ny/2, Ny-1]

    ax09.set_xticks(lx)
    ax09.set_xticklabels(x)

    y = ['0', r'$\frac{\pi}{2}$', r'$\pi$']
    ly = [0, Nx/2, Nx-1]
        
    ax09.set_yticks(ly)
    ax09.set_yticklabels(y)

    ax09.tick_params(axis='both', which='major', labelsize=7, pad=2)
    ax09.set_ylabel(r'$\theta$', fontsize =7, labelpad = 2, fontname=fname)
    ax09.set_xlabel(r'$\phi$', fontsize =7, labelpad = 3,  fontname=fname)
    # ax08.legend(prop=font)

    figname = 'polarity_map_doublet_'+str(doublet_number)+'cell_'+str(cell_number)+'_t_'+str(timepoint)+''
    fig09.savefig(save_path+figname+polarity_name)
    
    plot_on_a_sphere_complete(polar_map, min_s, max_s)
    return()



def calc_centre_of_mass_signal1(vertices,triangles, signal_values):
    
    ### signal_values is an array as vertices with intensity of a given signal of interest
    
    u1 = vertices[triangles[:, 0], :]
    u2 = vertices[triangles[:, 1], :]
    u3 = vertices[triangles[:, 2], :]
        
    tri_centres = (u1+u2+u3)/3.0

    signal_tri_avg = (signal_values[triangles[:, 0]] + signal_values[triangles[:, 1]] + signal_values[triangles[:, 2]])/3.0
    
    min_signal_avg = np.min(signal_tri_avg)
    max_signal_avg = np.max(signal_tri_avg)
    
    signal_tri_avg = (signal_tri_avg - min_signal_avg)/(max_signal_avg-min_signal_avg)
    
    # calculate area of each triangle
    v1 = u2 - u1
    v2 = u3 - u1
    cross = np.cross(v1, v2)
    areas = 0.5*np.sqrt(cross[:, 0]**2+cross[:, 1]**2+cross[:, 2]**2)
    
    integrated_signal = np.sum(areas*signal_tri_avg)
        
    # calculate sum of triangle centres, weighted by area and signal, divided by integrated signal 
    weighted_centres = np.zeros_like(tri_centres)
    weighted_centres[:, 0] = tri_centres[:, 0]*areas * signal_tri_avg
    weighted_centres[:, 1] = tri_centres[:, 1]*areas * signal_tri_avg
    weighted_centres[:, 2] = tri_centres[:, 2]*areas * signal_tri_avg
    
    com = weighted_centres.sum(axis=0)/integrated_signal
    
    return(com)

def polarity1(vertices, triangles, signal_values):
    
    com_signal1 = calc_centre_of_mass_signal1(vertices, triangles, signal_values)
    com_cell = calc_centre_of_mass(vertices, triangles)
    vol = calc_volume(com_cell, vertices, triangles)
    r = (3*vol/4/np.pi)**(1/3)
    
    p = com_signal1 - com_cell
    p /= r
    
    return(p)



def polarity2(vertices,triangles, signal_values):
    
    ### compute a polarity vector whose norm is between 0 and 1
    ### 1 represents a situation with a single spot on a cell
    
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    u1 = vertices[triangles[:, 0], :] - com_cell
    u2 = vertices[triangles[:, 1], :] - com_cell
    u3 = vertices[triangles[:, 2], :] - com_cell
        
    tri_centres = (u1+u2+u3)/3.0
    
    norm_of_tri_centres = np.sqrt(tri_centres[:,0]**2 + tri_centres[:,1]**2 + tri_centres[:,2]**2)
    
    signal_tri_avg = (signal_values[triangles[:, 0]] + signal_values[triangles[:, 1]] + signal_values[triangles[:, 2]])/3.0
    
    
    # min_signal_avg = np.min(signal_tri_avg)
    # max_signal_avg = np.max(signal_tri_avg)
    
    # signal_tri_avg = (signal_tri_avg - min_signal_avg)/(max_signal_avg-min_signal_avg)
    
    # calculate area of each triangle
    v1 = u2 - u1
    v2 = u3 - u1
    cross = np.cross(v1, v2)
    areas = 0.5*np.sqrt(cross[:, 0]**2+cross[:, 1]**2+cross[:, 2]**2)
          
    weighted_centres = np.zeros_like(tri_centres)
    
    weighted_centres[:, 0] = tri_centres[:, 0]*areas*signal_tri_avg
    weighted_centres[:, 1] = tri_centres[:, 1]*areas*signal_tri_avg
    weighted_centres[:, 2] = tri_centres[:, 2]*areas*signal_tri_avg
    
    p = weighted_centres.sum(axis=0)/np.sum(areas*signal_tri_avg*norm_of_tri_centres, axis = 0)
    
    
    return(p)



def polarity3(vertices, triangles, signal_values): 
    
    ### compute a polarity vector whose norm is between 0 and 1
    ### each point on the surface has a weight prop to its signal but not to its distance to the com_cell
    
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    u1 = vertices[triangles[:, 0], :] - com_cell
    u2 = vertices[triangles[:, 1], :] - com_cell
    u3 = vertices[triangles[:, 2], :] - com_cell
        
    tri_centres = (u1+u2+u3)/3.0
    
    norm_of_tri_centres = np.sqrt(tri_centres[:,0]**2 + tri_centres[:,1]**2 + tri_centres[:,2]**2)
    
    signal_tri_avg = (signal_values[triangles[:, 0]] + signal_values[triangles[:, 1]] + signal_values[triangles[:, 2]])/3.0
    
    #THIS is an addition to track spots
    # min_sig = np.min(signal_tri_avg)
    # max_sig = np.max(signal_tri_avg)
    
    # indhigh=signal_tri_avg >= min_sig + 0.5*(max_sig-min_sig)
    
    # signal_tri_avg[indhigh]=1
    # signal_tri_avg[~indhigh]=0
    
    #signal_tri_avg = (signal_tri_avg - min_signal_avg)/(max_signal_avg-min_signal_avg)
    
    # calculate area of each triangle
    
    v1 = u2 - u1
    v2 = u3 - u1
    cross = np.cross(v1, v2)
    areas = 0.5*np.sqrt(cross[:, 0]**2+cross[:, 1]**2+cross[:, 2]**2)
          
    weighted_centres = np.zeros_like(tri_centres)
    
    weighted_centres[:, 0] = tri_centres[:, 0]*areas*signal_tri_avg/norm_of_tri_centres
    weighted_centres[:, 1] = tri_centres[:, 1]*areas*signal_tri_avg/norm_of_tri_centres
    weighted_centres[:, 2] = tri_centres[:, 2]*areas*signal_tri_avg/norm_of_tri_centres
    
    p = weighted_centres.sum(axis=0)/np.sum(areas*signal_tri_avg, axis = 0)
    
    return(p)

@jit
def fast_binning(ind_theta,ind_phi,dtheta,dphi,signal_tri_avg,indices,Ntheta,Nphi,theta_t,phi_t):
    signal_on_sphere=np.zeros((Ntheta,Nphi))
    for index in range(len(ind_theta)):
        
        #if bin is empty, we use the closest triangle.
        #otherwise we average the signal in the bin
        
        #get i,j the bin indice in theta,phi
        i=ind_theta[index]
        j=ind_phi[index]
        #check number of vertices in bin
        indth = np.logical_and(i*dtheta <= theta_t, theta_t <= (i+1)*dtheta)
        indphi= np.logical_and(-np.pi+j*dphi <= phi_t, phi_t <= -np.pi+(j+1)*dphi)
        ind_in = np.logical_and(indth,indphi)
        if np.sum(ind_in)>0:
            #things in bin so we take the mean
            signal_on_sphere[i,j]=np.mean(signal_tri_avg[ind_in])
        else:
            #bin empty, we take the signal of nearest vertex
            signal_on_sphere[i,j]=signal_tri_avg[indices[index]]
    return signal_on_sphere

def polarity4(vertices, triangles, signal_values):
    #compute a polarity vector based on integrating on a unit sphere along theta,phi
    #so supposedly not perturbed by elongations of the shape
    
    #get triangle centers, normalized
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    u1 = vertices[triangles[:, 0], :] - com_cell
    u2 = vertices[triangles[:, 1], :] - com_cell
    u3 = vertices[triangles[:, 2], :] - com_cell
        
    tri_centres = (u1+u2+u3)/3.0
    
    norm_of_tri_centres = np.sqrt(tri_centres[:,0]**2 + tri_centres[:,1]**2 + tri_centres[:,2]**2)
    
    tri_centres=tri_centres / norm_of_tri_centres[:,None]
    
    #signal on triangles
    signal_tri_avg = (signal_values[triangles[:, 0]] + signal_values[triangles[:, 1]] + signal_values[triangles[:, 2]])/3.0
    
    # max_sig=np.max(signal_tri_avg)
    # min_sig=np.min(signal_tri_avg)
    # #proceed to non-linear transformation of signal
    # b=(max_sig-min_sig)/((max_sig-min_sig)**2)
    # signal_tri_avg=min_sig+b*(signal_tri_avg-min_sig)**2
    
    
    # indhigh=signal_tri_avg >= min_sig + 0.8*(max_sig-min_sig)
    
    # signal_tri_avg[indhigh]=1
    # signal_tri_avg[~indhigh]=0
    
    
    #get theta,phi angles of each triangle center
    theta_t=np.arccos(tri_centres[:,2])
    phi_t=np.arctan2(tri_centres[:,1],tri_centres[:,0])
    
    #bins for integration of sphere, use number of triangles = number of bins
    # so nt = Nphi*Ntheta = 2*Ntheta**2 
    ntri=tri_centres.shape[0]
    Ntheta = int(np.sqrt(ntri/2))
    Nphi = 2*Ntheta
    
    dtheta = np.pi/Ntheta
    dphi = 2*np.pi/Nphi
    
    theta = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    x = np.outer(np.sin(theta), np.cos(phi)).flatten()
    y = np.outer(np.sin(theta), np.sin(phi)).flatten()
    z = np.outer(np.cos(theta), np.ones(np.size(phi))).flatten()
    
    x_t = np.sin(theta_t)*np.cos(phi_t)
    y_t = np.sin(theta_t)*np.sin(phi_t)
    z_t = np.cos(theta_t)
    
    xyz = np.stack((x,y,z), axis = 1)
    xyz_t = np.stack((x_t,y_t,z_t), axis = 1)
    
    tree = cKDTree(xyz_t)
    distances,indices = tree.query(xyz, k = 1)
    
    ind_theta = np.outer(np.arange(0,Ntheta), np.ones(np.size(phi))).flatten().astype(int)
    ind_phi = np.outer(np.ones(np.size(theta)),np.arange(0,Nphi)).flatten().astype(int)
    
    signal_on_sphere=fast_binning(ind_theta,ind_phi,dtheta,dphi,signal_tri_avg,indices,Ntheta,Nphi,theta_t,phi_t)
            
    #last thing to do, integrate on the sphere Delta r/ ||Delta r|| x signal and just signal
    Intvec_x_phi = np.sum(signal_on_sphere*np.cos(phi)[None,:], axis = 1)*2*np.pi/Nphi
    Intvec_y_phi = np.sum(signal_on_sphere*np.sin(phi)[None,:], axis = 1)*2*np.pi/Nphi
    Intvec_z_phi = np.sum(signal_on_sphere, axis = 1)*2*np.pi/Nphi
    Intsig_phi = np.sum(signal_on_sphere, axis = 1)*2*np.pi/Nphi
    
    Intsig = np.sum(Intsig_phi*np.sin(theta))*np.pi/Ntheta
    
    p=np.zeros(3)
    
    
    #Attention, there is the sin(theta) coming from element of surface
    p[0] = np.sum(Intvec_x_phi*np.sin(theta)*np.sin(theta))*np.pi/Ntheta/Intsig
    p[1] = np.sum(Intvec_y_phi*np.sin(theta)*np.sin(theta))*np.pi/Ntheta/Intsig
    p[2] = np.sum(Intvec_z_phi*np.cos(theta)*np.sin(theta))*np.pi/Ntheta/Intsig
    
    return p,signal_on_sphere


def polarity_norm_from_sphere(ang_map_theta_phi):
    
    Ntheta, Nphi = ang_map_theta_phi.shape
    
    dtheta = np.pi/Ntheta
    dphi = 2*np.pi/Nphi
    
    theta = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    int_phi = np.sum(ang_map_theta_phi, axis = 1)*2*np.pi/Nphi
    
    int_signal = np.sum(np.sin(theta)*int_phi)*np.pi/Ntheta
    
    px_phi = np.sum(np.cos(phi)[None,:]*ang_map_theta_phi, axis = 1)*dphi
    py_phi = np.sum(np.sin(phi)[None,:]*ang_map_theta_phi, axis = 1)*dphi 
    pz_phi = np.sum(ang_map_theta_phi, axis = 1)*dphi 
    
    px = np.sum(np.sin(theta)*px_phi)*dtheta/int_signal
    py = np.sum(np.sin(theta)*py_phi)*dtheta/int_signal
    pz = np.sum(np.cos(theta)*pz_phi)*dtheta/int_signal
    
    p = np.array([px, py, pz])

    
    return(int_signal/4/np.pi, np.linalg.norm(p))

def calc_centre_of_mass_signal_simple(vertices,triangles, signal_values):
    
    ### signal_values is an array as vertices with intensity of a given signal of interest
    
    u1 = vertices[triangles[:, 0], :]
    u2 = vertices[triangles[:, 1], :]
    u3 = vertices[triangles[:, 2], :]
        
    tri_centres = (u1+u2+u3)/3.0

    signal_tri_avg = (signal_values[triangles[:, 0]] + signal_values[triangles[:, 1]] + signal_values[triangles[:, 2]])/3.0
    
    ind_max = np.argmax(signal_tri_avg)
        
    com = tri_centres[ind_max]
    
    return(com)



def compute_alpha(p1, u1):
    
    alpha = np.arccos(np.dot(p1/np.linalg.norm(p1), u1/np.linalg.norm(u1)))
    
    return(alpha)

def compute_beta(p1,p2,u1):
    #u1 needs to be normalised to 1
    u1_norm = u1/np.linalg.norm(u1)
    p1_proj = p1 - np.dot(p1,u1_norm)*u1_norm
    p2_proj = p2 - np.dot(p2,u1_norm)*u1_norm

    beta = np.arccos(np.dot(p1_proj/np.linalg.norm(p1_proj), p2_proj/np.linalg.norm(p2_proj)))
    return(beta)

def compute_beta2(p1,p2,u1):
    #other definition of beta between 0 and 2*pi
    #u1 needs to be normalised to 1
    u1_norm = u1/np.linalg.norm(u1)
    p1_proj = p1 - np.dot(p1,u1_norm)*u1_norm
    p2_proj = p2 - np.dot(p2,u1_norm)*u1_norm
    
    #basis in the plane perpendicular to r12 : ex=-p1_proj, ey=r12xex
    ex=-p1_proj/np.linalg.norm(p1_proj)
    ey=np.cross(u1_norm,ex)
    
    x=np.dot(p2_proj,ex)
    y=np.dot(p2_proj,ey)
    
    return np.pi+np.arctan2(y,x)
    

def compute_delta_signal(vertices, triangles, signal_values):
    
    ### signal_values is an array as vertices with intensity of a given signal of interest
    ### delta_signal = max_over_triangles(signal_values)/mean_over_triangles(signal_values)
    
    u1 = vertices[triangles[:, 0], :]
    u2 = vertices[triangles[:, 1], :]
    u3 = vertices[triangles[:, 2], :]
        
    tri_centres = (u1+u2+u3)/3.0

    signal_tri_avg = (signal_values[triangles[:, 0]] + signal_values[triangles[:, 1]] + signal_values[triangles[:, 2]])/3.0

    # calculate area of each triangle
    v1 = u2 - u1
    v2 = u3 - u1
    cross = np.cross(v1, v2)
    areas = 0.5*np.sqrt(cross[:, 0]**2+cross[:, 1]**2+cross[:, 2]**2)
    
    integrated_signal = np.sum(areas*signal_tri_avg)
    integrated_areas = np.sum(areas)
    mean = integrated_signal/integrated_areas
    max_s = np.max(signal_tri_avg)
    
    return(max_s/mean)


def compute_avg_time_maps(all_maps):
    
    all_sum_maps = []
    for cell_maps in all_maps:
        sum_time_cell_map = np.zeros(cell_maps[0].shape)
        for time_map in cell_maps:
            sum_time_cell_map += time_map
        all_sum_maps.append(sum_time_cell_map)
    
    all_normalised_maps = normalize_theta_phi_signal(all_sum_maps)
    
    return(all_normalised_maps)

def compute_avg_time_single_map(maps):
    sum_time_cell_map = np.zeros(maps[0].shape)
    for time_map in maps:
        sum_time_cell_map += time_map
    
    normalised_map = normalize_theta_phi_signal([sum_time_cell_map])
    return normalised_map

def compute_avg_cell_maps(all_normalised_maps):
    
    return(np.mean(all_normalised_maps, axis = 0))


def bin_plot(x, y, nbins):
    
    x = np.array(x)
    y = np.array(y)
    
    y_mean = np.zeros(nbins+1)
    y_std = np.zeros(nbins+1)
    nvalues = [0]
    for i in range(nbins):
        y_tmp = y[np.logical_and(x>=i*np.max(x)/nbins, x<=((i+1)*np.max(x)/nbins))]
        y_mean[i+1] = np.mean(y_tmp)
        y_std[i+1] = 1.96*np.std(y_tmp)/np.sqrt(len(y_tmp))
        nvalues.append(len(y_tmp))
    x_bin = np.linspace(0.5*np.max(x)/nbins,np.max(x)-0.5*np.max(x)/nbins,nbins)
    x_bins = np.array([0]+list(x_bin))
    
    return(x_bins, y_mean, y_std, np.array(nvalues))



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

def set_xticks_yticks(ax, lx, ly):
    
    x = [str(e) for e in lx]
    y = [str(e) for e in ly]
    ax.set_xticks(lx)
    ax.set_yticks(ly)
    ax.tick_params(axis='both', which='major', labelsize=7, pad=2)

    return()
    
def set_xlim_ylim(ax, xlim, ylim):
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=7, pad=2)
    return()

def save_figure(fig, figname, save_path):
    
    fig.savefig(save_path+figname)
    
    return()


cm = 1/2.54

fname = 'Arial'
font = font_manager.FontProperties(family=fname,
                                    weight='normal',
                                    style='normal', size=5)

def plot_correlation(x, y, xlabel, ylabel, nbootstrap):
    
    nbins = 10
    nexcluded = 4
    
    fig15, ax15 = plt.subplots(figsize=(3.4*cm,3.22*cm))
    
    x = np.array(x)
    y = np.array(y)
    
    x_mean, y_mean, y_std, nvalues = bin_plot_negative(x, y, nbins)
    
    ax15.scatter(x, y, s = 0.5, alpha = 0.5)
    ax15.errorbar(x_mean[nvalues > nexcluded], y_mean[nvalues > nexcluded], yerr = y_std[nvalues > nexcluded], color = 'r', linewidth = 0.8)
    
    data_cor = [[p, y[i]] for i,p in enumerate(x)]
    sample_cor, p_value = F.bootstrap_correl(data_cor, nbootstrap)
    print('pvalue = ', p_value)
    sample_cor_mean = np.mean(sample_cor)
    sample_cor_std = np.std(sample_cor)
    xtmp = np.linspace(np.min(x), np.max(x),100)
    print('sample_cor_mean = ', sample_cor_mean)
    line_plot = np.mean(y) + sample_cor_mean*(np.std(y)/np.std(x))*(xtmp-np.mean(x))
    ax15.plot(xtmp, line_plot, '-g', linewidth = 0.7)
    line_plot = np.mean(y) + (1.96*sample_cor_std+sample_cor_mean)*(np.std(y)/np.std(x))*(xtmp-np.mean(x))
    ax15.plot(xtmp, line_plot, '-g', linewidth = 0.7, alpha = 0.2)
    line_plot = np.mean(y) + (-1.96*sample_cor_std+sample_cor_mean)*(np.std(y)/np.std(x))*(xtmp-np.mean(x))
    ax15.plot(xtmp, line_plot, '-g', linewidth = 0.7, alpha = 0.2)
    
    ax15.tick_params(axis='both', which='major', labelsize=7, pad=2)
    ax15.set_ylabel(ylabel, fontsize =7, labelpad = 2, fontname=fname)
    ax15.set_xlabel(xlabel, fontsize =7, labelpad = 3,  fontname=fname)
    
    return(fig15,ax15)

def plot_correlation_more_info(x, y, xlabel, ylabel, nbootstrap):
    
    nbins = 10
    nexcluded = 4
    
    fig15, ax15 = plt.subplots(figsize=(3.4*cm,3.22*cm))
    
    x = np.array(x)
    y = np.array(y)
    
    x_mean, y_mean, y_std, nvalues = bin_plot_negative(x, y, nbins)
    
    ax15.scatter(x, y, s = 0.5, alpha = 0.5)
    ax15.errorbar(x_mean[nvalues > nexcluded], y_mean[nvalues > nexcluded], yerr = y_std[nvalues > nexcluded], color = 'r', linewidth = 0.8)
    
    data_cor = [[p, y[i]] for i,p in enumerate(x)]
    sample_cor, p_value = F.bootstrap_correl(data_cor, nbootstrap)
    print('pvalue = ', p_value)
    sample_cor_mean = np.mean(sample_cor)
    sample_cor_std = np.std(sample_cor)
    xtmp = np.linspace(np.min(x), np.max(x),100)
    print('sample_cor_mean = ', sample_cor_mean)
    line_plot = np.mean(y) + sample_cor_mean*(np.std(y)/np.std(x))*(xtmp-np.mean(x))
    ax15.plot(xtmp, line_plot, '-g', linewidth = 0.7)
    line_plot = np.mean(y) + (1.96*sample_cor_std+sample_cor_mean)*(np.std(y)/np.std(x))*(xtmp-np.mean(x))
    ax15.plot(xtmp, line_plot, '-g', linewidth = 0.7, alpha = 0.2)
    line_plot = np.mean(y) + (-1.96*sample_cor_std+sample_cor_mean)*(np.std(y)/np.std(x))*(xtmp-np.mean(x))
    ax15.plot(xtmp, line_plot, '-g', linewidth = 0.7, alpha = 0.2)
    
    ax15.tick_params(axis='both', which='major', labelsize=7, pad=2)
    ax15.set_ylabel(ylabel, fontsize =7, labelpad = 2, fontname=fname)
    ax15.set_xlabel(xlabel, fontsize =7, labelpad = 3,  fontname=fname)
    
    return(fig15,ax15,p_value)

