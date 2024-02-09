# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:36:42 2022

@author: Riveline LAB
"""


#------------------------------------------------------------------------------
#This must be executed before 'plots_Fig1_Fig2.py'. It loads the meshes of the
#cells (after cell division, .ply files in Cell_1, Cell_2 folders) and extract 
#information in the shape of numpy arrays that are then stored as .npy files:
#    - doublet_i_time.npy (time after cell division)
#    - doublet_i_velocity.npy (rotation rate of doublet)
#    - doublet_i_cell_j_com.npy (center of mass of cell j)
#    - interface_cloud_i_t_j.npy (vertices of the cell-cell interface at time j)
#    - outter_cells_cloud_i_t_j.npy (vertices that are not on the cell-cell interface)
#    - x_y_h_interface_i_t_j.npy (Interface points in local coordinates relative to the interface plane)
#    - data_plane_vectors_i_t_j.npy (Normal vector to the interface plane)
#------------------------------------------------------------------------------



import os
import numpy as np

os.chdir('../Figure3_Figure4')
import read_ply
import basal_maps as bm
import useful_functions_ply_files as uf
import useful_functions_interface as uf_int
import function_final_interface as ff_int
os.chdir('../Figure1_Figure2')

def get_names_ply(path1):
    
    A = np.array(os.listdir(path=path1))
    B = np.array([file.endswith(".ply") for file in A])
    A = A[B]
    return(A)

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


def compute_cell_rotational_velocity(com_cell1, com_cell2, time_int):
    ## from the cell center of mass as a function of time and the time_int in sec
    ## returns the magnitude of rotational velocity in revolution per hour

    r_g = (com_cell1 + com_cell2)/2
    vec1 = com_cell1-r_g
    vec2 = com_cell2-r_g
    
    cross_product1 = np.cross(vec1[0:-1],vec1[1:])
    n1 = np.zeros(np.shape(cross_product1))
    for i,e in enumerate(cross_product1) :
        n1[i] = e/np.linalg.norm(e)
    
    
    dtheta1 = np.arccos(np.dot(vec1[0:-1],vec1[1:].T).diagonal(0,0,1)/np.linalg.norm(vec1[1:],axis=1)/np.linalg.norm(vec1[0:-1],axis=1))
    w1 = np.zeros(np.shape(cross_product1))
    for i,e in enumerate(n1) :
        w1[i] = e * dtheta1[i]/time_int
    
    scale_factor = 180/np.pi*60*60/360
    cell_velocity = np.linalg.norm(w1,axis=1)*scale_factor
    
    return(cell_velocity)



save_path = './data_rotation/'


for i in range(1,15):
    
    save_path_doublet = save_path+'doublet_'+str(i)+'/'

    print(f'doublet {i}')
    
    
    PATHS = [save_path_doublet + 'Cell_1', save_path_doublet + 'Cell_2']
    
    time_int = 10*60 #time interval in s
    
    com_cell1, com_cell2 = get_series_com(PATHS)
    
    cell_velocity = compute_cell_rotational_velocity(com_cell1, com_cell2, time_int)
    t = np.linspace(0,len(cell_velocity)-1,len(cell_velocity))*time_int/60


    filename = 'doublet_'+str(i)+'_time'
    np.save(save_path_doublet+filename+'.npy', t)
    np.save(save_path_doublet+'doublet_'+str(i)+'_velocity.npy', cell_velocity)
    np.save(save_path_doublet+'doublet_'+str(i)+'_cell_1_com', com_cell1)
    np.save(save_path_doublet+'doublet_'+str(i)+'_cell_2_com', com_cell2)
    
    startpoint,endpoint = uf_int.find_startpoint_endpoint(PATHS[0])
    timepoints = np.linspace(startpoint, endpoint, endpoint-startpoint+1,dtype=int)
    
    for t in timepoints:
        
        cell1, cell2 = bm.get_cells_cloud_time(t, PATHS)
        dist1, dist2 = ff_int.compute_distance_between_two_clouds(cell1, cell2)
        
        interface_cloud = np.vstack((cell1[dist1<10],cell2[dist2<10]))
        rest_cells = np.vstack((cell1[dist1>=10],cell2[dist2>=10]))
        np.save(save_path_doublet+'interface_cloud_'+str(i)+'_t_'+str(t), interface_cloud)
        np.save(save_path_doublet+'outter_cells_cloud_'+str(i)+'_t_'+str(t), rest_cells)
        
        filename = 'x_y_h_interface_'+str(i)+'_t_'+str(t)
        # print(filename)
        data = ff_int.save_x_y_h_quantities(interface_cloud, save_path_doublet, filename)
        
        filename = 'data_plane_vectors_'+str(i)+'_t_'+str(t)
        data = ff_int.save_plane_vectors(interface_cloud, save_path_doublet, filename)
