#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:00:46 2023

@author: vagne
"""

#------------------------------------------------------------------------------
#This script generates the plot of Extended Figure 4c, which is the area of the
#interface as a function of time after division
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#Import packages and define functions
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontm
import os
from numba import njit
from pyevtk.hl import unstructuredGridToVTK
os.chdir('../Figure3_Figure4')
import useful_functions_interface as uf_int
import useful_functions_ply_files as uf
import function_final_interface as ff_int
os.chdir('../Figure1_Figure2')

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

def get_series_volumes(cells_path):
    
    startpoint,endpoint = uf_int.find_startpoint_endpoint(cells_path[0])

    vol_cell1 = np.zeros((endpoint-startpoint+1))
    vol_cell2 = np.zeros((endpoint-startpoint+1))
    for path in cells_path:
        filenames = get_names_ply(path)
        for file in filenames:
            t = int(file.split('time')[1].split('_')[0])
            mesh,V,T = uf.get_vertices_triangles(path+'/'+file)
            com_cell = uf.calc_centre_of_mass(V,T)
            
            if 'cell_1' in file: 
                vol_cell1[t-startpoint] = calc_volume(com_cell, V, T)
                
            else :
                vol_cell2[t-startpoint] = calc_volume(com_cell, V, T)
                    
    return(np.array(vol_cell1),np.array(vol_cell2))

def get_names_ply(path):
    A = np.array(os.listdir(path))
    B = np.array([file.endswith(".ply") for file in A])
    A = A[B]
    return(A)

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

@njit
def get_triangles(vert_ind,T):
    #get indices of triangles in T for which all their vertices are in vert_ind
    nt=T.shape[0]
    nv=vert_ind.shape[0]
    ind_T=np.zeros(nt,dtype=np.bool_)
    for i in range(nt):
        found=0
        for j in range(nv):
            if T[i,0]==vert_ind[j]:
                for k in range(nv):
                    if T[i,1]==vert_ind[k]:
                        for l in range(nv):
                            if T[i,2]==vert_ind[l]:
                                found=1
                                break
                    if found==1:
                        break
            if found==1:
                break
        if found==1:
            ind_T[i]=True
    return ind_T
                
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

def calc_surface(vertices, triangles):
    
    # get vertices of all triangles
    tri_pos = np.zeros((np.append(triangles.shape, 3)))
    tri_pos[:, 0, :] = vertices[triangles[:, 0], :]
    tri_pos[:, 1, :] = vertices[triangles[:, 1], :]
    tri_pos[:, 2, :] = vertices[triangles[:, 2], :]
        
    # calculate areas of each triangle
    v1 = tri_pos[:, 1, :] - tri_pos[:, 0, :]
    v2 = tri_pos[:, 2, :] - tri_pos[:, 0, :]
    cross = np.cross(v1, v2)
    areas = 0.5*np.sqrt(cross[:, 0]**2+cross[:, 1]**2+cross[:, 2]**2)
        
    surface_area = areas.sum()
    
    return(surface_area)
    

#------------------------------------------------------------------------------
#Loop over every doublet, extract the interface between the cells and compute
#its area
#------------------------------------------------------------------------------


#plotting settings
cm = 1/2.54
fname = 'Arial'
font = fontm.FontProperties(family=fname,
                                   weight='normal',
                                   style='normal', size=5)
base_path="./data_rotation/"


doub_name=["doublet_"+str(i)+"/" for i in range(1,15)]
nd=len(doub_name)

all_paths=[base_path+k for k in doub_name]

indices_doublets=[]
ang_velocity=[]
area=[]
time_tot=[]
time_tot_ind=[]
area_ind=[]

muperpix=0.206 #this is the conversion factor in micron per pixel size (pixel size in x-y plane)
dt=10 #time in min between frames

for k,path in enumerate(all_paths):
    PATHS = [path + 'Cell_1', path+ 'Cell_2']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    indices_doublets.append(endpoint-startpoint)
    
for k,path in enumerate(all_paths):
    
    print(path)
    

    PATHS = [path + 'Cell_1', path+ 'Cell_2']
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    
    com_cell1, com_cell2 = get_series_com(PATHS)
    vol_cell1, vol_cell2 = get_series_volumes(PATHS)
    com_signal1 = np.zeros(com_cell1.shape)
    com_signal2 = np.zeros(com_cell2.shape)

        

    r_g = (com_cell1 + com_cell2)/2
    
    vec1 = com_cell1-r_g
    vec2 = com_cell2-r_g

    time_int = dt*60 #in sec
    time = np.linspace(0,(len(com_cell1)-2)*dt,len(com_cell1)-1)
    time_tot += time.tolist()
    time_tot_ind.append(time)

    cross_product1 = np.cross(vec1[0:-1],vec1[1:])
    n1 = np.zeros(np.shape(cross_product1))
    for h,e in enumerate(cross_product1) :
        n1[h] = e/np.linalg.norm(e)
    
    
    dtheta1 = np.arccos(np.dot(vec1[0:-1],vec1[1:].T).diagonal(0,0,1)/np.linalg.norm(vec1[1:],axis=1)/np.linalg.norm(vec1[0:-1],axis=1))
    w1 = np.zeros(np.shape(n1))
    for i,e in enumerate(n1) :
        w1[i] = e * dtheta1[i]/time_int
        ang_velocity.append(np.linalg.norm(w1[i])*180/np.pi*60*60/360)
    
    
    area2=[]
    for t in range(startpoint, endpoint): #this loops is consistent with omega, last time point is not done
        
        
        mesh0,V0,T0 = uf.get_vertices_triangles(PATHS[0]+'/time'+str(t)+"_cell_1.ply")
        mesh1,V1,T1 = uf.get_vertices_triangles(PATHS[1]+'/time'+str(t)+"_cell_2.ply")
        
        
        dist0, dist1 = ff_int.compute_distance_between_two_clouds(V0, V1)
        dist_threshold = 10 
        ind0=dist0<dist_threshold #indices of vertices in interface of cell 1
        ind1=dist1<dist_threshold #same for cell 2
        T0_int=get_triangles(np.arange(0,ind0.shape[0])[ind0],T0)
        T1_int=get_triangles(np.arange(0,ind1.shape[0])[ind1],T1)
        #area in microns
        A0=calc_surface(V0,T0[T0_int])*muperpix*muperpix
        A1=calc_surface(V1,T1[T1_int])*muperpix*muperpix
        #average area
        A=0.5*(A0+A1)
        #store it
        area.append(A)
        area2.append(A)
    area_ind.append(area2)
        

area=np.array(area)
ang_velocity=np.array(ang_velocity)
time_tot=np.array(time_tot)


#------------------------------------------------------------------------------
#Generate the figure
#------------------------------------------------------------------------------



fig=plt.figure(figsize=(7.*cm,4*cm))
for i in range(len(area_ind)):
    plt.plot(time_tot_ind[i],area_ind[i],'ko--',alpha=0.2,markersize=1)
plt.xlabel(r"Time after cell division (min)",fontsize =7, labelpad = 3, fontname=fname)
plt.ylabel(r"Interface area ($\mu\mathregular{m^2}$)",fontsize =7, labelpad = 3, fontname=fname)
plt.tick_params(axis='both', which='major', labelsize=7, pad=2)

nbin=20
Xbin=np.linspace(np.min(time_tot),np.max(time_tot),nbin+1)
Ybin=np.zeros(nbin)
Yerr=np.zeros(nbin)
dx=(Xbin[nbin]-Xbin[0])/nbin
Xbinc=np.linspace(Xbin[0]+dx/2,Xbin[nbin]-dx/2,nbin)

for i in range(nbin):
    index=np.logical_and(time_tot >= Xbin[i],time_tot <= Xbin[i+1])
    Ybin[i]=np.mean(area[index])
    Yerr[i]=1.96*np.std(area[index])/np.sqrt(area[index].shape[0])

plt.errorbar(Xbinc,Ybin,yerr=Yerr,fmt='r-',linewidth=1)
        
    
