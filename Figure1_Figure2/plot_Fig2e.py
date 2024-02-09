# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:18:07 2024

@author: Riveline LAB
"""


#------------------------------------------------------------------------------
#
#This script generates the height profiles of the interface between the cells
#that are shown on Fig.2e 
#
#------------------------------------------------------------------------------


import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
from numba import njit
###############################################################################
#####################       FUNCTIONS          ################################
###############################################################################

def find_startpoint_endpoint(cell_path):
    
    
    a = os.listdir(cell_path)
    b = [int(e.split('time')[1].split('_')[0]) for e in a]
    startpoint = min(b)
    endpoint = max(b)
    return(startpoint,endpoint)


def get_vertices_triangles(file): 
    
    mesh = o3d.io.read_triangle_mesh(file) # Read the point cloud

    V = np.asarray(mesh.vertices)

    T = np.asarray(mesh.triangles)
    
    return(mesh,V,T)

def get_names_ply(path1):
    
    A = np.array(os.listdir(path=path1))
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

def get_series_com(cells_path):
    
    startpoint,endpoint = find_startpoint_endpoint(cells_path[0])
    com_cell1 = np.zeros((endpoint-startpoint+1,3))
    com_cell2 = np.zeros((endpoint-startpoint+1,3))
    for path in cells_path:
        filenames = get_names_ply(path)
        for i,file in enumerate(filenames):
            t = int(file.split('time')[1].split('_')[0])
            mesh,V,T = get_vertices_triangles(path+'/'+file)
            com_cell = calc_centre_of_mass(V,T)
            
            if 'cell_1' in file: 
                com_cell1[t-startpoint,:] = com_cell
                
            else :
                com_cell2[t-startpoint,:] = com_cell
                    
    return(np.array(com_cell1),np.array(com_cell2))


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
    
    return(Cint,[ex, ey, Nint], [X,Y,H])

def poly_fit_naive_orthogonal(X,Y,H):
    #the values in X and Y must be representing an interface with a radius of 1.0 so that the fit and the decomposition makes sense
    sp=np.sqrt(np.pi)
    m0=X*0+1
    m10= X
    m11= Y
    S1= X**2
    S2= X*Y
    B= Y**2
    PL1= X**2*Y
    PL2= Y**2*X
    Yy1= X**3
    Yy2= Y**3
    A=np.array([m0,m10,m11,S1,S2,B,PL1,PL2,Yy1,Yy2]).T
    coeff, r, rank,s=np.linalg.lstsq(A,H)

    return coeff

@njit
def compute_distance(A, hull_equations, tolerance = 1e-12):
    

    ind = np.full(len(A),True)
    for i in range(len(A)):
        res = False
        for eq in hull_equations:
            if (np.dot(eq[:-1], A[i]) + eq[-1] >= tolerance) :
                res = True
                break
            
        ind[i] = res
            
    return(ind)

def normalize_vector(vec):
    
    norm_vec = np.copy(vec)
    for i,v in enumerate(vec):
        norm_vec[i] = v/np.linalg.norm(v)
    return(norm_vec)


#------------------------------------------------------------------------------
#Select a doublet and a set of time points
#------------------------------------------------------------------------------
path="./data_rotation/doublet_5/"
timepoints_to_plot = np.array([30,41,68])


#parameters
scale_factor = 1.0/0.206
dt = 10 #in min
time_int = dt*60 #in sec
dist_threshold = int(np.floor(scale_factor) + 1) 
xy_pix = int(np.floor(scale_factor) + 1)
z_pix = 1



PATHS = [path + 'Cell_1', path+ 'Cell_2']
startpoint,endpoint = find_startpoint_endpoint(PATHS[0])

com_cell1, com_cell2 = get_series_com(PATHS)

u1 = normalize_vector(com_cell2-com_cell1)


for i,t in enumerate(np.linspace(startpoint, endpoint, endpoint-startpoint+1)):
    
    
    t = int(t)
    mesh1, V1, T1 = get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')
    mesh2, V2, T2 = get_vertices_triangles(PATHS[1]+'/time'+str(t)+'_cell_2.ply')
    
    dist1, dist2 = compute_distance_between_two_clouds(V1, V2)
    interface_v = np.vstack((V1[dist1<dist_threshold],V2[dist2<dist_threshold]))
    
    Cint, [ex, ey, Nint], [X,Y,H] = naive_fit_plane_oriented(interface_v, u1[i])
        
        
    if t in timepoints_to_plot:
        
        coeff = poly_fit_naive_orthogonal(X,Y,H)
        
        #create the new map
        Npoints = 200
        X_new = np.linspace(np.min(X), np.max(X), Npoints)
        Y_new = np.linspace(np.min(Y), np.max(Y), Npoints)
        xv, yv = np.meshgrid(X_new, Y_new)
        xv = xv.flatten()
        yv = yv.flatten()
        
        XYnew0 = np.vstack((xv,yv)).T
        
        dx = (np.max(X)-np.min(X))/(Npoints-1)
        dy = (np.max(Y)-np.min(Y))/(Npoints-1)
        dist = np.sqrt(dx**2+dy**2)
        
        hull_equation = ConvexHull(np.vstack((X,Y)).T).equations
        ind0 = compute_distance(XYnew0, hull_equation, dist)
        
        XYnew = np.delete(XYnew0,ind0, axis = 0)
        
        Xnew = XYnew[:,0]
        Ynew = XYnew[:,1]
        Znew = coeff[0]+coeff[1]*Xnew + coeff[2]*Ynew + coeff[3]*Xnew**2 + coeff[4]*Xnew*Ynew + coeff[5]*Ynew**2
        Znew += coeff[6]*Xnew**2*Ynew + coeff[7]*Xnew*Ynew**2 + coeff[8]*Xnew**3 +coeff[9]*Ynew**3
        
        X = np.copy(Xnew)
        Y = np.copy(Ynew)
        
        
        fig1, ax1 = plt.subplots(figsize=(19.20,10.80))
        
        ax1.scatter(X, Y, c = Znew*0.206, cmap=cm.viridis, s = 10 , vmin = -4.0, vmax = 4.0)
        ax1.set_aspect((np.max(X)-np.min(X))/(np.max(Y)-np.min(Y)))
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        fig1.patch.set_visible(False)
        ax1.axis('off')