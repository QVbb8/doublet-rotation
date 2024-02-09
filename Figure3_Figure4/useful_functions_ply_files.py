# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 16:55:55 2022

@author: Tristan Guyomar
"""
import os
import numpy as np

import read_ply

import numpy as np

import pandas as pd
import open3d as o3d

from skimage.io import imread, imsave
from tifffile import imwrite



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

def get_all_volumes(ply_path):
    
    os.chdir(ply_path)
    Volumes = []
    for file in os.listdir():
        if file.endswith(".ply"):
            mesh,V,T = get_vertices_triangles(file)
            Volumes.append(calc_volume(calc_centre_of_mass(V,T), V, T))
            
    return(Volumes)

def calc_com_tissue(ply_path):
    
    os.chdir(ply_path)
    A = np.array(os.listdir())
    B = np.array([file.endswith(".ply") for file in A])
    A = A[B]
    COM = np.zeros((len(A),3))
    for i,file in enumerate(A):
        mesh,V,T = get_vertices_triangles(file)
        COM[i,:] = calc_centre_of_mass(V,T)   
    
    return(np.mean(COM,axis=0))

def get_all_com(ply_path):
    
    os.chdir(ply_path)
    com = []
    for file in os.listdir():
        if file.endswith(".ply"):
            mesh,V,T = get_vertices_triangles(file)
            com.append(calc_centre_of_mass(V,T))
            
    return(np.array(com))


def convert_pos_to_pix(vertices, scale_factor):
    
    pos = vertices
    pix = np.zeros_like(pos)
    pix[:, 0] = pos[:, 0]
    pix[:, 1] = pos[:, 1]
    pix[:, 2] = pos[:, 2]/scale_factor
    pix = np.round(pix).astype(np.int)
    
    # reorder pixel positions to be [z, rows, cols]
    pix = pix[:, [2, 1, 0]] 
    
    return pix
