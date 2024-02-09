#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:07:10 2023

@author: vagne
"""

#------------------------------------------------------------------------------
#This script generates the experimental plots of Figure 5g and 5h, analysing
#the reaction of cell doublets to local optogenetic activation of myosin
#------------------------------------------------------------------------------
        

#------------------------------------------------------------------------------
#Import packages and define necessary functions
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt 
from random import randint
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as font_manager
import matplotlib as mpl
from random import choices

cm = 1/2.54

fname = 'Arial'
font = font_manager.FontProperties(family=fname,
                                    weight='normal',
                                    style='normal', size=5)

def rotmat_2d(vec,alpha):
    
    x_vec = vec[:,0]
    y_vec = vec[:,1]
    
    xnew_vec = x_vec * np.cos(alpha) - y_vec*np.sin(alpha)
    ynew_vec = x_vec * np.sin(alpha) + y_vec*np.cos(alpha)
    
    new_vec = np.array([xnew_vec, ynew_vec]).T
    
    return(new_vec)

def rotmat(vec,alpha):
    
    x_vec = vec[0]
    y_vec = vec[1]
    
    xnew_vec = x_vec * np.cos(alpha) - y_vec*np.sin(alpha)
    ynew_vec = x_vec * np.sin(alpha) + y_vec*np.cos(alpha)
    
    new_vec = np.array([xnew_vec, ynew_vec]).T
    
    return(new_vec)


def extract_time_info(data):
    
    dt = data[:,0] #time between frames
    nframe = data[:,1].astype(int) #number of frames
    on_off = data[:,2].astype(int) # activation on : 1 , activation off : 0
    recording_on_off = data[:,3].astype(int) # recording on : 1; recording off : 0
    #print('dts :', dt)
    #print('number of frames :', nframe)
    #print('activation_on_off :', on_off)
    #print('recording_on_off :', recording_on_off)
    
    start = 0
    nframe_activation = 0
    time = []
    for i,delta_t in enumerate(dt) :
        
        if nframe[i] > 1 :
            
            time_int = np.linspace(start, start + (nframe[i]-1)*delta_t, nframe[i])
            time += list(time_int)
            start = time[-1] + 30

        else : 
            if recording_on_off[i] == 1:
                time_int = np.array([start])
                time += list(time_int)
                start = time[-1] + 30
            else :
                start = time[-1] + delta_t + 30
    
    a = np.cumsum(nframe)
    b = np.argwhere(on_off==1)[0,0]-1
    index_activation = a[b]
    tim = np.array(time)
    time = tim - tim[index_activation-1]
    
    return(time, index_activation)


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

def bootstrap(data1,nsample):
    #give p-value for mean of data1 being larger than zero
    sample_diff=[]
    for i in range(nsample):
        y1=np.mean(choices(data1,k=len(data1)))
        sample_diff.append(y1)
    
    
    return (sample_diff,np.sum(np.array(sample_diff)<0)/len(sample_diff))


#------------------------------------------------------------------------------
#Main loop of analysis, plotting the individual curves for each doublet of 
#figure 5.g and 5.h
#------------------------------------------------------------------------------

path = './data/'
label_numbers = [4,5,6,7,8,9,12,13,14]
x_range =[]
y_range = []
slopes_before = []
slopes_act = []
Delta_slope=[]
dx_bef=[]
dx_aft=[]
stock_num_of_rev = []
min_range_time = []
max_range_time = []
time_avg = []
viridis = plt.cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
cmp = ListedColormap(newcolors)

fig1, ax1 = plt.subplots(figsize = (3.3*cm, 3.4*cm))
fig0, ax0 = plt.subplots(figsize = (3.3*cm, 3.4*cm))

for k,n in enumerate(label_numbers) :
    
    doublet_name = 'doublet_'+str(n)

    analysis_name = path + doublet_name + '/' + doublet_name+'_time.csv'
    
    data = np.genfromtxt(analysis_name, delimiter=',')
    
    time, idx_fiji_on =  extract_time_info(data)
    time = time / 3600
    #print('\n'+'For the '+doublet_name+' :')
    #print('Activation at the frame :', idx_fiji_on)
    n_act = 10


    analysis_name_cell_1 = path + doublet_name + '/' + doublet_name+'_fiducial_marker_cell1_xy.csv'
    
    data_cell1 = np.genfromtxt(analysis_name_cell_1, delimiter=',')
    
    analysis_name_cell_2 = path + doublet_name + '/' + doublet_name+'_fiducial_marker_cell2_xy.csv'
    
    data_cell2 = np.genfromtxt(analysis_name_cell_2, delimiter=',')
    
    coord_fid_1 = data_cell1[1:,1:3]
    coord_fid_2 = data_cell2[1:,1:3]
    
    analysis_name = path+doublet_name+'/'+doublet_name+'_hand_analysis.csv'
        
    data = np.genfromtxt(analysis_name, delimiter=',')
    
    major = data[1:,5]
    minor = data[1:,6]
    R = np.mean(np.sqrt(major*minor)/2)
    x = data[1:,2]
    y = data[1:,3]
    
    coord_com = data[1:, 2:4]
    
    vec_1_com = coord_fid_1 - coord_com
    norm = np.linalg.norm(vec_1_com, axis = 1)
    vec_1_com = vec_1_com/norm[:,None]
    vec_2_com = coord_fid_2 - coord_com
    norm = np.linalg.norm(vec_2_com, axis = 1)
    vec_2_com = vec_2_com/norm[:,None]
    vec_1_2 = coord_fid_1 - coord_fid_2
    norm = np.linalg.norm(vec_1_2, axis = 1)
    vec_1_2 = vec_1_2/norm[:,None]
    
    cp_1_com = np.cross(vec_1_com[0:-1, :], vec_1_com[1:, :])
    dtheta_1_com = np.arcsin(cp_1_com)
    cp_2_com = np.cross(vec_2_com[0:-1, :], vec_2_com[1:, :])
    dtheta_2_com = np.arcsin(cp_2_com)
    cp_1_2 = np.cross(vec_1_2[0:-1, :], vec_1_2[1:, :])
    dtheta_1_2 = np.arcsin(cp_1_2)
    
    dtheta_mean = (dtheta_1_com+dtheta_2_com + dtheta_1_2)/3 #angle avec signe
    
    dtheta = dtheta_mean * np.sign(np.cumsum(dtheta_mean)[-1])
    
    num_of_rev = (np.array([0] + list(np.cumsum(dtheta))))/2/np.pi
    
    slope1 = num_of_rev[idx_fiji_on-1]/(-time[0])
    
    slope, orig = np.polyfit(time[0:idx_fiji_on], num_of_rev[0:idx_fiji_on],1)
    
    slope_act, orig_act = np.polyfit(time[idx_fiji_on:idx_fiji_on+10], num_of_rev[idx_fiji_on:idx_fiji_on+10],1)
    
    slopes_before.append(slope)
    slopes_act.append(slope_act)
    Delta_slope.append(slope_act-slope)
    
    
    analysis_name = path+doublet_name+'/'+doublet_name+'_roi_xy.csv'
    data = np.genfromtxt(analysis_name, delimiter=',')
    
    x_roi = data[1:,1]
    y_roi = data[1:,2]
    
    x_com_act = [np.mean(x_roi[4*i:4*(i+1)]) for i in range(len(x_roi)//4)]
    y_com_act = [np.mean(y_roi[4*i:4*(i+1)]) for i in range(len(x_roi)//4)]
    
    vec_opto_act_com = coord_com[idx_fiji_on-1] - np.array([x_com_act[0], y_com_act[0]]) #vecteur entre le centre de la ROI d'activation et le COM du doublet au temps initial d'activation
    
    ex = np.zeros((len(time),2))
    
    ex[idx_fiji_on-1] = np.copy(vec_opto_act_com/np.linalg.norm(vec_opto_act_com))
    
    for i in range(idx_fiji_on, len(time)):
        
        ex[i] = rotmat(ex[i-1], dtheta_mean[i-1])
    
    for i in range(idx_fiji_on-2, -1, -1):
        
        ex[i] = rotmat(ex[i+1], -dtheta_mean[i])
        
    ey = np.zeros((len(time),2))
    
    ey[:,0] = -np.copy(ex[:,1])
    ey[:,1] = np.copy(ex[:,0])
    
    coord_rot_com = np.zeros((len(time),2))
    coord_rot_com[:,0] = np.sum((coord_com-coord_com[idx_fiji_on-1])*ex, axis = 1)/R
    coord_rot_com[:,1] = np.sum((coord_com-coord_com[idx_fiji_on-1])*ey, axis = 1)/R
    
    ax0.plot(coord_rot_com[idx_fiji_on-1:idx_fiji_on-1+10,0], coord_rot_com[idx_fiji_on-1:idx_fiji_on-1+10,1], '--k', alpha = 0.2, lw = 0.5)
    ax0.scatter(coord_rot_com[idx_fiji_on-1:idx_fiji_on-1+11,0], coord_rot_com[idx_fiji_on-1:idx_fiji_on-1+11,1], s = 5, c = time[idx_fiji_on-1:idx_fiji_on-1+11], cmap = cmp, vmin = np.min(time[idx_fiji_on-1:idx_fiji_on-1+11]), vmax = np.max(time[idx_fiji_on-1:idx_fiji_on-1+11]))
    ax0.plot(coord_rot_com[0:idx_fiji_on-2,0], coord_rot_com[0:idx_fiji_on-2,1], '--k', alpha = 0.2, lw = 0.5)
    ax0.scatter(coord_rot_com[0:idx_fiji_on-2,0], coord_rot_com[0:idx_fiji_on-2,1], s = 5, alpha = 0.2, c = 'k', cmap = cmp, vmin = np.min(time), vmax = np.max(time))
    
    
    dx_bef.append(coord_rot_com[idx_fiji_on-1-11,0])
    dx_aft.append(coord_rot_com[idx_fiji_on-1+11,0])
    
    ax1.plot(time*60, num_of_rev-orig, 'o-', ms = 0.4, lw = 1, color = 'k', alpha = 0.1)
    
    
    stock_num_of_rev+=list(num_of_rev-orig)
    max_range_time.append(np.max(time*60))
    min_range_time.append(np.min(time*60))
    time_avg += list(time*60)
    

#------------------------------------------------------------------------------
#Finish formatting figure 5.g 
#------------------------------------------------------------------------------
ax0.set_aspect('equal')
borne = 0.1
ax0.set_xlim([-borne, borne])
ax0.set_ylim([-borne, borne])
ly = [-0.1, -0.05, 0.0, 0.05, 0.1]
y = [str(e) for e in ly] 
ax0.set_yticks(ly)
ax0.set_yticklabels(y)
lx = [-0.1, -0.05, 0.0, 0.05, 0.1]
x = [str(e) for e in lx] 
ax0.set_xticks(lx)
ax0.set_xticklabels(x)
ax0.axvline(0, linestyle = '--', color = 'black', lw = 0.5)
ax0.axhline(0, linestyle = '--', color = 'black', lw = 0.5)
ax0.set_aspect('equal', adjustable='box')
ax0.tick_params(axis='both', which='major', labelsize=7, pad=2)
ax0.set_xlabel('X/R', fontsize =7, labelpad = 2, fontname=fname)
ax0.set_ylabel('Y/R', fontsize =7, labelpad = 2, fontname=fname)


#------------------------------------------------------------------------------
#Finishing figure 5.h by adding the average curve and the error bars
#------------------------------------------------------------------------------


stock_num_of_rev = np.array(stock_num_of_rev)
time_avg = np.array(time_avg)
avg_num_of_rev = []
std_num_of_rev = []
avg_time = []
for t in np.unique(time_avg):
    n = len(stock_num_of_rev[time_avg == t])
    if n>1:
        avg_num_of_rev.append(np.mean(stock_num_of_rev[time_avg == t]))
        std_num_of_rev.append(1.96*np.std(stock_num_of_rev[time_avg == t])/n)
        avg_time.append(t)

t_mean, num_of_rev_mean, y_std, nvalues = bin_plot_negative(time_avg, stock_num_of_rev, 30)

ax1.axvline(0*60,color ='red', lw = 0.8, alpha = 1, ls = '--', label = 'Optogenetic Activation')
ax1.axvline(0.1111*60,color ='cyan', lw = 0.8, alpha = 1, ls = '--')

nexcluded = 0
ax1.errorbar(t_mean[nvalues > nexcluded], num_of_rev_mean[nvalues > nexcluded], yerr = y_std[nvalues > nexcluded], color = 'r', linewidth = 0.8)


lx = [-10,-5,0,5,10,15,20]
x = [str(e) for e in lx] 
ax1.set_xticks(lx)
ax1.set_xticklabels(x)
ax1.set_ylabel(r'Number of revolutions', fontsize =7, labelpad = 2, fontname=fname)
ax1.set_xlabel('Time (min)', fontsize =7, labelpad = 3,  fontname=fname)
ax1.tick_params(axis='both', which='major', labelsize=7, pad=2)




#------------------------------------------------------------------------------
#Statistical test of figure 5.g, displacement of the doublet before vs after
#activation
#------------------------------------------------------------------------------
dx_bef=np.array(dx_bef)
dx_aft=np.array(dx_aft)
dx_diff=np.abs(dx_aft)-np.abs(dx_bef)

a,b=bootstrap(dx_diff,100000)
print(b)

#------------------------------------------------------------------------------
#Statistical test of figure 5.h, average change of slope during the activation
#period compared to before the activation
#------------------------------------------------------------------------------

a,b=bootstrap(Delta_slope,100000)
print(1-b)



