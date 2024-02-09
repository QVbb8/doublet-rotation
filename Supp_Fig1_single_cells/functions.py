#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:15:40 2023

@author: vagne
"""
import os
import numpy as np
import useful_functions_ply_files as uf
import useful_functions_interface as uf_int
from scipy.spatial import Delaunay
from numba import njit
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def func_fit(theta, tension_mod, b):
    
    return(1.0 + tension_mod*((b*np.exp(b*np.cos(theta))-np.sinh(b))/(b*np.exp(b)-np.sinh(b))))

def get_tension_mod_fit_phi(Ntheta, avg_phi):
    
    theta = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)
    
    popt, pcov = curve_fit(func_fit, theta, avg_phi, bounds=([0,0.001], [10,100]))
    
    return(popt[0],popt[1])



def generate_paths(name_path,N,scl_fact,use_blur):
    all_paths=[name_path+str(i)+'/' for i in range(1,N+1)]
    if use_blur==True:
        all_img=[str(i)+'_blur.tif' for i in range(1,N+1)]
    else:
        all_img=[str(i)+'.tif' for i in range(1,N+1)]
    name=[str(i) for i in range(1,N+1)]
    scale_factors=[scl_fact for i in range(1,N+1)]
    return all_paths,all_img,name,scale_factors

def make_kymograph(maps,angle):
    #make a plot (kymograph) showing the average signal between phi=-angle and angle
    #as a function of theta (0 - pi) and time
    T=len(maps)
    Ntheta=maps[0].shape[0]
    Nphi=maps[0].shape[1]
    
    #phi index corresponding to the opening angle
    jm=int(0.5*(Nphi*(-angle/180+1)-1))
    jM=int(0.5*(Nphi*(angle/180+1)-1))
    
    kymo=np.zeros((Ntheta,T))
    for t in range(T):
        kymo[:,t]=np.mean(maps[t][:,jm:(jM+1)],axis=1)
    
    return kymo
    
    
    
    

def rotmat(u, th):
    #rotation matrix for a rotation around vector u (normalized) of angle theta
    #works for arrays of vectors u and arrays of th
    nu=np.sqrt(np.sum(u**2,axis=1))
    u=u/nu[:,None]
    
    M=np.zeros((u.shape[0],3,3))
    M[:,0,0]=np.cos(th) +u[:,0]**2*(1-np.cos(th))
    M[:,0,1]=u[:,0]*u[:,1]*(1-np.cos(th)) - u[:,2]*np.sin( th)
    M[:,0,2]=u[:,0]*u[:,2]*(1-np.cos(th)) + u[:,1]*np.sin( th)
    M[:,1,0]=u[:,1]*u[:,0]*(1-np.cos(th)) + u[:,2]*np.sin( th)
    M[:,1,1]=np.cos(th) + u[:,1]**2*(1-np.cos(th))
    M[:,1,2]=u[:,1]*u[:,2]*(1-np.cos(th)) - u[:,0]*np.sin( th)
    M[:,2,0]=u[:,2]*u[:,0]*(1-np.cos(th)) - u[:,1]*np.sin( th)
    M[:,2,1]=u[:,2]*u[:,1]*(1-np.cos(th)) + u[:,0]*np.sin( th)
    M[:,2,2]=np.cos(th) + u[:,2]**2*(1-np.cos(th))
    
    return M


def normalize_vector(vec):
    
    norm_vec = np.copy(vec)
    for i,v in enumerate(vec):
        norm_vec[i] = v/np.linalg.norm(v)
    return(norm_vec)


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


def get_angular_distribution_maxsignal_r12(vertices,triangles, signal_value, r12):
    
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    val_max=np.max(signal_value)
    val_min=np.min(signal_value)
    
    mean_point=np.mean(vertices[signal_value>=(val_min+0.9*(val_max-val_min))],axis=0)
    
    ex=mean_point-com_cell
    ex /= np.linalg.norm(ex)
    
    ey = np.cross(r12,ex)
    ey /= np.linalg.norm(ey)
    
    ez = np.cross(ex,ey)
    ez /= np.linalg.norm(ez)
    
    v = vertices - com_cell
    normv=np.linalg.norm(v,axis=1)
    v = normalize_vector(v)
    
    theta = np.arccos(np.dot(v,ez))
    temp = np.dot(v,ez)[:,None]*ez[None,:]
    vproj = v - temp
    
    x = np.dot(vproj, ex)
    y = np.dot(vproj, ey)
    
    
    phi = np.arctan2(y,x)

    
    return(theta,phi,normv)

def normalize_theta_phi_signal(avg_map):

    
    Ntheta, Nphi = avg_map.shape
    ang = np.linspace(np.pi/(2*Ntheta), np.pi-np.pi/(2*Ntheta), Ntheta)

    phi = np.linspace(-np.pi + 2*np.pi/(2*Nphi), np.pi- 2*np.pi/(2*Nphi), Nphi)
    
    avg_map_norm = []
    
    
    int_phi = np.sum(avg_map, axis = 1)*2*np.pi/Nphi
        
    int_signal = np.sum(np.sin(ang)*int_phi)*np.pi/Ntheta
    print(f'average signal = {int_signal}')    
    if int_signal < 0 :
        print('''C'EST PAS BIEN ! FALLAIT PAS LE FAIRE !''')
    
    avg_map_norm = 4*np.pi*avg_map/int_signal
        
    return(avg_map_norm)

def get_angular_distribution_p(vertices,triangles, signal_value,p,direction):
    
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    val_max=np.max(signal_value)
    val_min=np.min(signal_value)
    
    
    ez=np.copy(p)
    ez /= np.linalg.norm(ez)
    
    ex=np.copy(direction)
    
    ey = np.cross(ez,ex)
    ey /= np.linalg.norm(ey)
    
    ex = np.cross(ey,ez)
    ex /= np.linalg.norm(ex)
    
    v = vertices - com_cell
    v = normalize_vector(v)
    
    theta = np.arccos(np.dot(v,ez))
    temp = np.dot(v,ez)[:,None]*ez[None,:]
    vproj = v - temp
    
    x = np.dot(vproj, ex)
    y = np.dot(vproj, ey)
    
    
    phi = np.arctan2(y,x)

    
    return(theta,phi)

def get_angular_distribution_z(vertices,triangles):
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    v = vertices - com_cell
    v = normalize_vector(v)
    theta=np.arccos(v[:,2])
    phi = np.arctan2(v[:,1],v[:,0])
    
    return (theta,phi)

def get_angular_distribution_input_r12(vertices,triangles, r12, input_ex):
    
    com_cell = uf.calc_centre_of_mass(vertices,triangles)
    
    
    ex=np.copy(input_ex)
    ex /= np.linalg.norm(ex)
    
    ey = np.cross(r12,ex)
    ey /= np.linalg.norm(ey)
    
    ez = np.cross(ex,ey)
    ez /= np.linalg.norm(ez)
    
    v = vertices - com_cell
    v = normalize_vector(v)
    
    theta = np.arccos(np.dot(v,ez))
    temp = np.dot(v,ez)[:,None]*ez[None,:]
    vproj = v - temp
    
    x = np.dot(vproj, ex)
    y = np.dot(vproj, ey)
    
    
    phi = np.arctan2(y,x)

    
    return(theta,phi)


def get_names_ply(path1):
    
    A = np.array(os.listdir(path=path1))
    B = np.array([file.endswith(".ply") for file in A])
    A = A[B]
    return(A)

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

def exponential_calibration_single(img,scale_factor,l):
    #applies an exponential correction exp(z/l) to the raw values in img
    img_resc=np.zeros(img.shape)
    
    # reorder pixel positions to be [z, rows, cols]
    # pix are the  real coordinates of the vertices XYZ in units of ImageJ 
    # but WARNING img = imread(img_name) you give youi img as [z][y][x] 
    
    z=np.arange(0,img.shape[0])*scale_factor
    expcor=np.exp(z/l)
    
    img_resc=img*expcor[:,None,None] #term by term multiplication,first dim is z
    
    return img_resc

@njit
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

@njit
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

@njit
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
        diff = value_signal - avg_signal_cell
        values[k] = diff
        
    mean_before = mean_before / len(pix)
           
    return(values,mean_before,max_before)

@njit
def avg_signal(xyz_inside, im_mrlc):
    
    img_inside_cell = 0.0
    
    for triplet in xyz_inside:
        z = int(triplet[0])
        x = int(triplet[1])
        y = int(triplet[2])
        img_inside_cell+=im_mrlc[z][x][y]
    avg_inside_signal = img_inside_cell/len(xyz_inside)
    return(avg_inside_signal)


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
    
    return(avg_inside_signal)

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

@njit
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
                com_cell1[t-startpoint,:] = com_cell
                
            else :
                com_cell2[t-startpoint,:] = com_cell
                    
    return(np.array(com_cell1),np.array(com_cell2))

def get_series_com_single(cells_path):
    
    startpoint,endpoint = uf_int.find_startpoint_endpoint(cells_path[0])

    com_cell1 = np.zeros((endpoint-startpoint+1,3))
    for path in cells_path:
        filenames = get_names_ply(path)
        for file in filenames:
            t = int(file.split('time')[1].split('_')[0])
            mesh,V,T = uf.get_vertices_triangles(path+'/'+file)
            com_cell = uf.calc_centre_of_mass(V,T)
            
            com_cell1[t-startpoint,:] = com_cell

                    
    return np.array(com_cell1)



def stereographic_projection_polarity(p,com,V,signal,limit_angle,orientation_vec):
    #project vertices V on a plane normal to p, at a distance 1 from the center of mass com
    #also returns the signal of the projected vertices
    #only keep vertices with and angle less than limit_angle
    #use orientation_vec to construct the x and y axis in the plane
    
    pnorm=p/np.linalg.norm(p)
    
    u_vert=V-com
    norm=np.sqrt(np.sum(u_vert**2,axis=1))
    u_vert_norm=u_vert/norm[:,None]
    
    #let's compute the theta angle of every vertex
    theta_vert=np.arccos(np.sum(u_vert_norm*pnorm[None,:],axis=1))
    
    #eliminate those with theta>limit_angle
    filtered_vert=u_vert[theta_vert<limit_angle]
    filtered_sig=signal[theta_vert<limit_angle]
    
    scal=np.sum(pnorm[None,:]*filtered_vert,axis=1)
    filtered_vert=filtered_vert/scal[:,None]-pnorm[None,:]
    
    #we now use orientation_vec to define ex,ey vectors in the projected plane
    ex=orientation_vec-pnorm*np.sum(pnorm*orientation_vec)
    ex=ex/np.linalg.norm(ex)
    ey=np.cross(pnorm,ex)
    
    X=np.sum(filtered_vert*ex[None,:],axis=1)
    Y=np.sum(filtered_vert*ey[None,:],axis=1)
    
    #return the result as angles and radii
    return (X,Y,filtered_sig)

def display_stereographic_projection(X,Y,filtered_sig,Nr,Nphi):
    #make a circular plot from an array of X and Y coordinates, and a signal
    
    #maximum radius
    R=np.sqrt(X**2+Y**2)
    RM=np.max(R)
    #phi angles
    Phi=np.arctan2(Y,X)
    
    #create bin centers
    Phi_b=np.linspace(np.pi/Nphi,2*np.pi-np.pi/Nphi,Nphi)
    R_b=np.linspace(RM/(2*Nr),RM-RM/(2*Nr),Nr)
    XYb=np.zeros((Nr*Nphi,2))
    XYb[:,0]=np.outer(R_b,np.cos(Phi_b)).flatten()
    XYb[:,1]=np.outer(R_b,np.sin(Phi_b)).flatten()
    
    #tree of distances of each bin to each point (X,Y)
    XY=np.stack((X,Y),axis=1)
    tree = cKDTree(XYb)
    distances,indices = tree.query(XY, k = 1)
    
    #put the signal of the vertices into the bins
    indr=np.outer(np.arange(0,Nr),np.ones(np.size(Phi_b))).flatten().astype(np.int64)
    indphi=np.outer(np.ones(np.size(R_b)),np.arange(0,Nphi)).flatten().astype(np.int64)      
    sig_proj=bin_stereo(indr,indphi,RM/Nr,2*np.pi/Nphi,filtered_sig,indices,Nr,Nphi,R,Phi)
    
    #for the plot we need the borders of the bins
    Phi_bb=np.linspace(0,2*np.pi,Nphi+1)
    R_bb=np.linspace(0,RM,Nr+1)
    Az, Rad = np.meshgrid(Phi_bb, R_bb)
    
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
    pc = ax.pcolormesh(Az, Rad, sig_proj, cmap="magma_r")
    fig.colorbar(pc)
    
    
    
@njit
def bin_stereo(indr,indphi,dr,dphi,sig,indices,Nr,Nphi,R,Phi):
    signal_proj=np.zeros((Nphi,Nr))
    for index in range(len(indr)):
        
        #if bin is empty, we use the closest vertex.
        #otherwise we average the signal in the bin
        
        #get i,j the bin indice in r,phi
        #WE WILL FILL signal_proj [j,i] so that we have [phi,r] in the array for the plotting
        i=indr[index]
        j=indphi[index]
        #check number of vertices in bin
        A = np.logical_and(i*dr <= R, R <= (i+1)*dr)
        B = np.logical_and(j*dphi <= Phi, Phi <= (j+1)*dphi)
        ind_in = np.logical_and(A,B).astype(np.int64)
        if np.sum(ind_in)>0:
            #things in bin so we take the mean
            signal_proj[j,i]=np.mean(sig[ind_in])
        else:
            #bin empty, we take the signal of nearest vertex
            signal_proj[j,i]=sig[indices[index]]
    return signal_proj
   
    
    
    