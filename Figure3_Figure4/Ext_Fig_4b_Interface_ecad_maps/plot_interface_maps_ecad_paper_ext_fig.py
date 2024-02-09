# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:18:07 2024

@author: Riveline LAB
"""


#------------------------------------------------------------------------------
#This script generates the plot of Extended Figure 4b. It shows the pattern
#of the cadherin signal on the interface in a reference frame that follows the
#doublet rotation.
#------------------------------------------------------------------------------


import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imread
import open3d as o3d
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial import cKDTree
from numba import njit
import random
###############################################################################
#####################       FUNCTIONS          ################################
###############################################################################

def find_startpoint_endpoint(cell_path):
    
    
    a = os.listdir(cell_path)
    b = [int(e.split('time')[1].split('_')[0]) for e in a if not e=='.DS_Store']
    startpoint = min(b)
    endpoint = max(b)
    return(startpoint,endpoint)

def get_names_ply(path1):
    
    A = np.array(os.listdir(path=path1))
    B = np.array([file.endswith(".ply") for file in A])
    A = A[B]
    return(A)

def get_vertices_triangles(file): 
    
    mesh = o3d.io.read_triangle_mesh(file) # Read the point cloud

    V = np.asarray(mesh.vertices)

    T = np.asarray(mesh.triangles)
    
    return(mesh,V,T)

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

def rotmat(u, th):
    #    ed) of angle theta
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


def convert_pos_to_pix(vertices, scale_factor):
    
    pos = vertices
    pix = np.zeros_like(pos)
    pix[:, 0] = pos[:, 0]
    pix[:, 1] = pos[:, 1]
    pix[:, 2] = pos[:, 2]/scale_factor
    pix = np.round(pix).astype(int)
    
    # reorder pixel positions to be [z, rows, cols]
    pix = pix[:, [2, 1, 0]]
    
    return pix


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
    
    pos = convert_pos_to_pix(vertices,scale_factor)
    
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

        if interface_v[k] == 0:
            diff = value_signal - avg_signal_cell
            values[k] = diff

        else :
            diff = (value_signal - (avg_signal_cell+avg_signal_other_cell)/2)/2
            values[k] = diff
    
    mean_before = mean_before / len(pix)
    return(values,mean_before,max_before)

def find_max_min_signal_value_overtime(img,PATHS, dist_threshold, npix_xy, npix_z,name_signal, startpoint, endpoint, interface = False):
    
    
    if interface :
        print('Computing mean only on interface vertices')
    else : 
        print('Computing mean on all cell vertices')
    
    if name_signal == 'mrlc':
        idx = 0
    elif name_signal == 'ecad':
        idx=1
    min_tot = []
    max_tot = []
    
    # snapshots = np.arange(2,12)
    
    for t in range(startpoint-1,endpoint-1):
    # for t in snapshots:
        im_ecad = img[t,:,1,:,:]
        mesh1, V1, T1 = get_vertices_triangles(PATHS[0]+'/time'+str(t+1)+'_cell_1.ply')
        mesh2, V2, T2 = get_vertices_triangles(PATHS[1]+'/time'+str(t+1)+'_cell_2.ply')
        
        dist1, dist2 = compute_distance_between_two_clouds(V1, V2)
        
        interface_V1 = np.zeros(len(V1))
        interface_V2 = np.zeros(len(V2))
        
        interface_V1[dist1<dist_threshold] = 1
        interface_V2[dist2<dist_threshold] = 1
        interface_V1 = interface_V1.astype(np.uint)
        interface_V2 = interface_V2.astype(np.uint)
        
        avg_inside_cell1 = get_signal_inside_cell(V1, im_ecad, scale_factor)
        avg_inside_cell2 = get_signal_inside_cell(V2,im_ecad, scale_factor)
                
        ecad_values_cell1,mb1,ma1 = get_signal_vertices(V1, im_ecad, xy_pix, z_pix, scale_factor, avg_inside_cell1, avg_inside_cell2, interface_V1)
        ecad_values_cell2,mb2,ma2 = get_signal_vertices(V2, im_ecad, xy_pix, z_pix, scale_factor, avg_inside_cell2, avg_inside_cell1, interface_V2)

        ecad_values = np.array(list(ecad_values_cell1)+list(ecad_values_cell2))
        
        if interface :
            signal_int1 = ecad_values_cell1[dist1<dist_threshold]
            signal_int2 = ecad_values_cell2[dist2<dist_threshold]
            
            ecad_values = np.array(list(signal_int1)+list(signal_int2))
    
        min_value = np.min(ecad_values)
        max_value = np.max(ecad_values)
        
        min_tot.append(min_value)
        max_tot.append(max_value)
        
    return(np.min(min_tot), np.max(max_tot))


def two_random_halves_indices(N):
    
    ind_half_1 = random.sample(range(0,N),N//2)
    ind_half_2 = []
    for i in range(0,N):
        if i not in ind_half_1:
            ind_half_2.append(i)
    ind_half_2 = np.array(ind_half_2)
    return(ind_half_1, ind_half_2)

## to plot the maps of the height of the interface
path = '../myosin_data/b1/Segmentation_2_B1_demo42_s1-2_ecad/'
img_name = '2_ecad_mrlc.tif'
start = 2
finish = 9
timepoints_to_plot = np.arange(2,9)

#parameters
scale_factor = 1.0/0.206
dt = 10 #in min
time_int = dt*60 #in sec
dist_threshold = int(np.floor(scale_factor) + 1) 
xy_pix = int(np.floor(scale_factor) + 1)
z_pix = 1
markermap = cm.Greens

img = imread(path+img_name)


PATHS = [path + 'Cell_1', path+ 'Cell_2']
startpoint,endpoint = find_startpoint_endpoint(PATHS[0])

min_value, max_value = find_max_min_signal_value_overtime(img, PATHS, dist_threshold, xy_pix, z_pix, 'ecad', start, finish, interface = True)
print(min_value, max_value)

com_cell1, com_cell2 = get_series_com(PATHS)

r_g = (com_cell1 + com_cell2)/2
vec1 = com_cell1-r_g

cross_product1 = np.cross(vec1[0:-1],vec1[1:])
n1 = np.zeros(np.shape(cross_product1))

for i,e in enumerate(cross_product1) :
    n1[i] = e/np.linalg.norm(e)

dtheta1 = np.arccos(np.dot(vec1[0:-1],vec1[1:].T).diagonal(0,0,1)/np.linalg.norm(vec1[1:],axis=1)/np.linalg.norm(vec1[0:-1],axis=1))
w1 = np.zeros(np.shape(n1))

M = rotmat(n1, dtheta1) #all rotation matrices between frames

u1 = normalize_vector(com_cell2-com_cell1)

for i,t in enumerate(np.linspace(startpoint, endpoint, endpoint-startpoint+1)):
    
    
    t = int(t)
    mesh1, V1, T1 = get_vertices_triangles(PATHS[0]+'/time'+str(t)+'_cell_1.ply')
    mesh2, V2, T2 = get_vertices_triangles(PATHS[1]+'/time'+str(t)+'_cell_2.ply')
    
    #plt.figure()
    #plt.plot(V1[:,0],V1[:,1],'o')
    #plt.plot(V2[:,0],V2[:,1],'ro')
    #plt.plot(np.array([com_cell1[i,0],com_cell2[i,0]]),np.array([com_cell1[i,1],com_cell2[i,1]]),'k-')
    
    dist1, dist2 = compute_distance_between_two_clouds(V1, V2)
    interface_v = np.vstack((V1[dist1<dist_threshold],V2[dist2<dist_threshold]))
    
    Cint, [ex, ey, Nint], [X,Y,H] = naive_fit_plane_oriented(interface_v, u1[i])
    
    if i == 0:
        ex_rot = ex
    else :
        ex_rot = ex_rot@M[i-1].T
        
        
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
        
        X3D = X*ex[0] + Y*ey[0] + Znew*Nint[0] + Cint[0]
        Y3D = X*ex[1] + Y*ey[1] + Znew*Nint[1] + Cint[1]
        Z3D = X*ex[2] + Y*ey[2] + Znew*Nint[2] + Cint[2]
        
        V = np.vstack((np.vstack((X3D,Y3D)),Z3D)).T
        
        ex_rot_proj = ex_rot - np.dot(ex_rot,Nint)*Nint #vecteur ex_rot dans le plan de l'interface définie par [ex, ey, Nint]
        ex_rot_proj /= np.linalg.norm(ex_rot_proj)
        ex_rot_x_proj = np.dot(ex_rot_proj,ex) #coordonnées de ex_rot_proj dans le plan [ex,ey]
        ex_rot_y_proj = np.dot(ex_rot_proj, ey)
        
        ey_rot_x_proj = - ex_rot_y_proj
        ey_rot_y_proj = ex_rot_x_proj
        
        X_rot = X*ex_rot_x_proj+Y*ex_rot_y_proj
        Y_rot = X*ey_rot_x_proj+Y*ey_rot_y_proj
        
        
        ### get the signal of the interface
        im_ecad = img[t-1,:,1,:,:]
        
        avg_inside_cell1 = get_signal_inside_cell(V1, im_ecad, scale_factor)
        avg_inside_cell2 = get_signal_inside_cell(V2,im_ecad, scale_factor)
        
        ecad_values,a,b = get_signal_vertices(V, im_ecad, xy_pix, z_pix, scale_factor, avg_inside_cell2, avg_inside_cell1, np.ones(len(V)))
        normalized_ecad_values_interface = (ecad_values-min_value)/(max_value-min_value)

        
        fig1, ax1 = plt.subplots(figsize=(19.20,10.80))
        
        ax1.scatter(X_rot/scale_factor, Y_rot/scale_factor, c = normalized_ecad_values_interface, cmap=markermap, s = 10 , vmin = 0, vmax = 1)
        ax1.set_aspect((np.max(X_rot)-np.min(X_rot))/(np.max(Y_rot)-np.min(Y_rot)))
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        fig1.patch.set_visible(False)
        ax1.axis('off')
        fig1.savefig(f'time_{t}.png')
        
        
        