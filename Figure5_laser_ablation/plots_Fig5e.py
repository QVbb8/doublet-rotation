#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:07:10 2023

@author: vagne
"""

#------------------------------------------------------------------------------
#This script generates the plot of Figure 5e which shows the number of revolution
#of doublets as a function of time, before and after laser ablation of the
#myosins spots.  
#------------------------------------------------------------------------------
        


#------------------------------------------------------------------------------
#Importing packages and defining functions 
#------------------------------------------------------------------------------        
import numpy as np
import matplotlib.pyplot as plt 
from random import randint
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as font_manager
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
    print('dts :', dt)
    print('number of frames :', nframe)
    print('activation_on_off :', on_off)
    print('recording_on_off :', recording_on_off)
    
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



def bootstrap_double(data1,data2,nsample):
    #give p-value for mean of data1 being larger than mean of data2
    sample_diff=[]
    for i in range(nsample):
        y1=np.mean(choices(data1,k=len(data1)))
        y2=np.mean(choices(data2,k=len(data2)))
        sample_diff.append(y1-y2)
    
    
    return (sample_diff,np.sum(np.array(sample_diff)<0)/len(sample_diff))


#------------------------------------------------------------------------------
#Loop on doublets, generating the individual curve for each doublet
#------------------------------------------------------------------------------    
fig1, ax1 = plt.subplots(figsize = (2.1*cm, 2.1*cm))
label_numbers = [1,2,3,4,5,7,8,9,10]
ablation_frame = [5,6,5,6,5,5,5,5,5]

path = './data/'
slopes_before = []
slopes_act = []
stock_num_of_rev = []
min_range_time = []
max_range_time = []
time_avg = []

for k,n in enumerate(label_numbers) :
    
    doublet_name = 'doublet_'+str(n)


    analysis_name_cell_1 = path + doublet_name + '/' + doublet_name+'_fiducial_marker_cell1_xy.csv'
    
    data_cell1 = np.genfromtxt(analysis_name_cell_1, delimiter=',')
    
    analysis_name_cell_2 = path + doublet_name + '/' + doublet_name+'_fiducial_marker_cell2_xy.csv'
    
    data_cell2 = np.genfromtxt(analysis_name_cell_2, delimiter=',')
    
    coord_fid_1 = data_cell1[1:,1:3]
    coord_fid_2 = data_cell2[1:,1:3]
    time = np.arange(0,2*len(coord_fid_1),2) 
    time -= (time[ablation_frame[k]-1])

    analysis_name = path+doublet_name+'/'+doublet_name+'_hand_analysis.csv'
        
    data = np.genfromtxt(analysis_name, delimiter=',')
    
    major = data[1:,5]
    minor = data[1:,6]
    R = np.mean(np.sqrt(major*minor/2))
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
    
    slope1 = num_of_rev[ablation_frame[k]-1]/(-time[0])
    
    slope, orig = np.polyfit(time[0:ablation_frame[k]], num_of_rev[0:ablation_frame[k]],1)
    
    slope_act, orig_act = np.polyfit(time[ablation_frame[k]:ablation_frame[k]+5], num_of_rev[ablation_frame[k]:ablation_frame[k]+5],1)
    
    slopes_before.append(slope)
    slopes_act.append(slope_act)
    
    ax1.plot(time, num_of_rev-orig,  'o-', ms = 0.4, lw = 1, color = 'k', alpha = 0.1)

    stock_num_of_rev+=list(num_of_rev-orig)
    time_avg += list(time)
    
#------------------------------------------------------------------------------
#Plot the average curve on all doublets, with error bars
#------------------------------------------------------------------------------  

stock_num_of_rev = np.array(stock_num_of_rev)
time_avg = np.array(time_avg)

avg_num_of_rev = []
std_num_of_rev = []
avg_time = []
for t in np.unique(time_avg):
    n = len(stock_num_of_rev[time_avg == t])
    #print(n)
    if n>2:
        avg_num_of_rev.append(np.mean(stock_num_of_rev[time_avg == t]))
        std_num_of_rev.append(1.96*np.std(stock_num_of_rev[time_avg == t])/np.sqrt(n))
        avg_time.append(t)
        
t_mean, num_of_rev_mean, y_std, nvalues = bin_plot_negative(time_avg, stock_num_of_rev, 31)

ax1.axvline(1,color ='red', lw = 0.8, ls = '--', alpha = 1, label = 'Laser Ablation')
ax1.errorbar(avg_time, avg_num_of_rev, yerr = std_num_of_rev, color = 'r', linewidth = 0.8)

lx = [-10, 0, 10, 20]
x = [str(e) for e in lx] 
lx = [e+1 for e in lx]
ax1.set_xticks(lx)
ax1.set_xticklabels(x)
ly = [-0.2,0.0, 0.2]
y = [str(e) for e in ly] 
ax1.set_yticks(ly)
ax1.set_yticklabels(y)
ax1.set_ylabel(r'Number of revolutions', fontsize =5, labelpad = 2, fontname=fname)
ax1.set_xlabel('Time after ablation (min)', fontsize =5, labelpad = 3,  fontname=fname)
ax1.tick_params(axis='both', which='major', labelsize=5, pad=2)


#------------------------------------------------------------------------------
#Statistical test comparing the average slope before ablation to the average 
#slope after ablation
#------------------------------------------------------------------------------   

a,b=bootstrap_double(slopes_before,slopes_act,500000)
print(b)


