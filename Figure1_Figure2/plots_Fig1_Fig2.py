#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 13:09:07 2023

@author: vagne
"""
#------------------------------------------------------------------------------
#This script generates the plots of Figure 1, Figure 2 and Extended Figure 3
# Figure 1 -
# f - omega vs time for one doublet 
# h - omega vector on sphere one doublet (Ext Fig 3 too.) 
# g - omega vs time all doublets 
# i - omega correl vs time all doublets 
#
# Fig 2
# b - elongation plot vs time 
# c - correlation elongation omega 
# e - map of height of interface -> Tristan?
# f - deflection vs time 
# g - relative mode amplitude 
# h - correl mode rotation
#
#Before running this script, the data in './data_rotation' must be prepared for
#analyisis by running 'prepare_data_for_analysis.py', which creates the .npy
#files necessary for this script.
# 
#------------------------------------------------------------------------------


import functions_doublet as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontm


#plotting settings
cm = 1/2.54

fname = 'Arial'
font = fontm.FontProperties(family=fname,
                                   weight='normal',
                                   style='normal', size=5)


#path of base folders containing all doublets
base_paths="./data_rotation/"
nd=14 #number of doublets
Start_ind=np.array([0,14,22,30,15,9,13,5,5,4,13,4,5,2,8]) #starting indices for each doublet (0.1) is 
#there so that we start accumulating variables from index 0 for first doublet
End_ind=np.array([1,80,87,87,77,73,82,34,29,53,44,74,46,79,84])
#compute how many rotation vectors can be computed
Np_rot=End_ind-Start_ind
Np_rot[0]=0
Np_rot=np.cumsum(Np_rot)
#omega is computed between two time points so we store n-1 omegas if we have
#n time points for a doublet (forget about index 0 of arrays End_ind and Start_ind and add a zero to indicate that for the 
#first doublet we start accumulating from beginning of array).

#compute how many points can be used if we decide to analyse AFTER t=something.
t_offset=15
Start_ind_offset=Start_ind+t_offset
Np_rot_offset=End_ind-Start_ind_offset
Np_rot_offset[0]=0
Np_rot_offset=np.cumsum(Np_rot_offset)


#interface reference frame
Plane_vec=np.zeros((3,3))

#loop on doublets for analysis
#accumulation of scalars defined at every time points
Wp=[]
Wn=[]
Wmid=[]
Wtime=[]
S=[]
Eta=[]
#accumulation of scalars to compare with omega, and accumulation of omega (amplitude)
Omamp=np.zeros(Np_rot[nd])
Hsq=np.zeros(Np_rot[nd]) #sqrt(<h^2>) measured by the coeff of the modes
Hsq2=np.zeros(Np_rot[nd]) #sqrt(h^2>) measured by the discrete experimental points
MS=np.zeros(Np_rot[nd]) #saddle-node amplitude
MB=np.zeros(Np_rot[nd]) #bowl amplitude
MPL=np.zeros(Np_rot[nd]) #3fold sym amplitude
MY=np.zeros(Np_rot[nd]) #yin-yang amplitude
M0=np.zeros(Np_rot[nd]) #mode zeros
M1=np.zeros(Np_rot[nd]) #mode 1
Mrest=np.zeros(Np_rot[nd]) # rest of the modes
Rint=np.zeros(Np_rot[nd]) #radius of interface
Lamb=np.zeros(Np_rot[nd]) #aspect ratio

#accumulation of scalar with time offset
Omamp_offset=np.zeros(Np_rot_offset[nd])
Hsq_offset=np.zeros(Np_rot_offset[nd])
MS_offset=np.zeros(Np_rot_offset[nd]) #saddle-node amplitude
MB_offset=np.zeros(Np_rot_offset[nd]) #bowl amplitude
MPL_offset=np.zeros(Np_rot_offset[nd]) #3fold sym amplitude
MY_offset=np.zeros(Np_rot_offset[nd]) #yin-yang amplitude

#accumulation of Omega and of other vectors to compare to Omega (or not)
Omegas=np.zeros((Np_rot[nd],3)) #direction of rotation vector
QP=np.zeros((Np_rot[nd],3))#prolate direction of Q tensor
QN=np.zeros((Np_rot[nd],3))#oblate direction of Q tensor
QM=np.zeros((Np_rot[nd],3))#middle direction of Q tensor
R12=np.zeros((Np_rot[nd],3))#com1 to com2 vector
Npl=np.zeros((Np_rot[nd],3))#plane vector of interface
Sdir=np.zeros((Np_rot[nd],3))#direction of saddle-node mode
Ydir=np.zeros((Np_rot[nd],3))#direction of Yin-Yang mode
TFdir=np.zeros((Np_rot[nd],3))#direction of Three-fold mode
PLSdir=np.zeros((Np_rot[nd],3))#direction of the combination of Saddle-node and three-fold symmetry
Bo=np.zeros((Np_rot[nd],3)) #bowl mode vector
Com_mov=np.zeros((Np_rot[nd],3)) #center of mass movement
N12xOm=np.zeros((Np_rot[nd],3)) #N12 cross Omega (vector used to correlate with the modes)
Wdir=np.zeros((Np_rot[nd],3)) #new wave mode defined by PL1+Y1 and PL2+Y2
RMdir=np.zeros((Np_rot[nd],3)) #ray manta mode PL1 - Y1 and PL2 - Y2
OmQ=np.zeros((Np_rot[nd],3)) # QP(t) x Qp(t+Dt)
#accumulation of bowl autocorrelations
Bowl_ac=[]
#list of omega vectors with the corresponding times
Nrot_list=[]
#autocorrelation of omega for each doublet
Om_correl=[]
for d in range(1,nd+1):
    print(f'doublet {d}')
    folder_path='doublet_'+str(d)+"/"
    #Time=np.load(base_paths+folder_path+'doublet_'+str(d)+'_time.npy')
    #no need for time array, we can make it ourselves
    Com1=np.load(base_paths+folder_path+"doublet_"+str(d)+"_cell_1_com.npy")
    Time=np.array([i*10 for i in range(len(Com1))])
    Com2=np.load(base_paths+folder_path+"doublet_"+str(d)+"_cell_2_com.npy")
    Comd=0.5*(Com1+Com2) #center of mass of doublet
    Vel=np.load(base_paths+folder_path+"doublet_"+str(d)+"_velocity.npy") #same as Omega in the end
    
    
    #rotation measurement
    npoints=len(Com1)
    u1=Com1-Comd
    u2=Com2-Comd
    Nrot=np.cross(u1[:-1],u1[1:])
    Dth=np.zeros(npoints-1)
    Nrot_no=np.sqrt(np.sum(Nrot**2,axis=1))
    Dthno=np.sqrt(np.sum(u1[1:]**2,axis=1)*np.sum(u1[:-1]**2,axis=1))
    Nrot=Nrot/Nrot_no[:,None]
    Dth=np.arccos(np.sum(u1[1:]*u1[:-1],axis=1)/Dthno)
    Omega=Dth/(Time[1:]-Time[:-1]) #rotation speed in radian per minute
    Omega=Omega/(2*np.pi)*60 #turn per hour
    
    Omega_smooth=F.smoothing_gaussian(Omega,5)
    
    
    #store Omega direction and amplitude
    Omegas[Np_rot[d-1]:Np_rot[d],:]=Nrot
    Omamp[Np_rot[d-1]:Np_rot[d]]=Omega
    
    #make the autocorrelation of the rotation vector and store it
    Om_correl.append(F.autocorrel(Nrot))

    #store also omega with time offset
    Omamp_offset[Np_rot_offset[d-1]:Np_rot_offset[d]]=Omega[t_offset:]
    
    Omegavector=Nrot*Omega[:,None]
    #F.show_omega(Omegavector,Time[:-1]/60)
    #F.show_omega_norm(Omegavector)
    Nrot_list+=[[Omegavector,Time[:-1]/60]]
    
    
    
    #Omegas[Np_rot[d-1]:Np_rot[d],:]=F.interp_rot(Nrot,False)#not nematic vector
    
    #R12 remove last point
    R12[Np_rot[d-1]:Np_rot[d],:]=Com2[:-1]-Com1[:-1]
    #center of mass movement
    Com_mov[Np_rot[d-1]:Np_rot[d],:]=(Comd[1:,:]-Comd[:-1,:])*0.206 #scale factpr!
    
    
    #loading the files defined at each time points
    #accumulate max eigenvalues of Q tensor
    QPtmp=np.zeros((npoints,3)) #accumiulate prolate eigenvector
    QNtmp=np.zeros((npoints,3)) #accumiulate oblate eigenvector
    QMtmp=np.zeros((npoints,3)) #accumiulate oblate eigenvector
    Npltmp=np.zeros((npoints,3)) #plane of interface
    Sdirtmp=np.zeros((npoints,3)) #direction of saddle-node
    Ydirtmp=np.zeros((npoints,3)) #direction of Yin-Yang
    TFdirtmp=np.zeros((npoints,3)) #direction of three-fold mode
    PLSdirtmp=np.zeros((npoints,3)) #direction of saddle-node+three fold
    Wdirtmp=np.zeros((npoints,3)) #new wave mode defined by PL1+Y1 and PL2+Y2
    RMdirtmp=np.zeros((npoints,3)) #ray manta mode PL1 - Y1 and PL2 - Y2
    
    #Local array for the signed bowl amplitude as a function of time
    Wb_signed=np.zeros(npoints) #signed bowl amplitude
    for t in range(Start_ind[d],End_ind[d]+1):
        #print(f'd:{d} t:{t}')
        #XYH=np.load(base_paths+folder_path+'x_y_h_interface_'+str(d)+'_t_'+str(t)+'.npy')
        #Plane_vec=np.load(base_paths+folder_path+'data_plane_vectors_'+str(d)+'_t_'+str(t)+'.npy')
        Outer_shell=np.load(base_paths+folder_path+'outter_cells_cloud_'+str(d)+'_t_'+str(t)+'.npy')
        Interface=np.load(base_paths+folder_path+'interface_cloud_'+str(d)+'_t_'+str(t)+'.npy')
        
        #now we redo the fit of the interface with a proper method using r12 vector to define center of interface
        Cint=np.mean(Interface,axis=0)
        #define r12
        r12=Com2[t-Start_ind[d],:]-Com1[t-Start_ind[d],:]
        r1C=Cint-Com1[t-Start_ind[d],:]
        dr=r12*(np.dot(r12,r1C))/(np.linalg.norm(r12)**2)-r1C
        Cint=Cint+dr
        
        svd=np.linalg.svd((Interface-Cint).T)
        #the last vector is the normal vector of the plane
        Plane_vec[:,2]=svd[0][:,-1]
        Plane_vec[:,0]=svd[0][:,0] #ex
        Plane_vec[:,1]=svd[0][:,1] #ey
        
        #Plane_vec[2] is the normal vector to the interface.
        #we have to make it such that it is always going from cell 1 to cell 2
        #make basis direct ! That might be important
        if np.sum(Plane_vec[:,2]*r12)<0:
            Plane_vec[:,2]=-Plane_vec[:,2]
            Plane_vec[:,1]=-Plane_vec[:,1] #this to keep basis direct...
        Npltmp[t-Start_ind[d],:]=Plane_vec[:,2]
        

        
        #project points to define X,Y,H
        XYH=np.zeros(Interface.shape)
        
        XYH[:,2]=np.dot(Interface-Cint,Plane_vec[:,2])
        XYH[:,2]=XYH[:,2]-np.mean(XYH[:,2])
        XYH[:,0]=np.dot(Interface-Cint,Plane_vec[:,0])
        XYH[:,1]=np.dot(Interface-Cint,Plane_vec[:,1])
    
        
        #do the mode decomposition
        hsq,coeff,R,lamb,err=F.extract_modes(XYH)
        #amplitude of modes
        W0=coeff[0]**2
        W1=coeff[1]**2+coeff[2]**2
        WS=coeff[3]**2+coeff[4]**2
        WB=coeff[5]**2
        WPL=coeff[6]**2+coeff[7]**2
        WY=coeff[8]**2+coeff[9]**2
        #signed bowl to accumulate
        Wb_signed[t-Start_ind[d]]=coeff[5]
        #print(WS,WB,WPL,WY)
        #we extract the direction of the saddle-node using the coeff 3 and 4 related to S1 and S2 modes
        alpha=0.5*np.arctan2(coeff[4],coeff[3])+np.pi
        Sdirtmp[t-Start_ind[d],:]=Plane_vec[:,0]*np.cos(alpha)+Plane_vec[:,1]*np.sin(alpha)
        #coeff 8 and 9 correspond to Y1 and Y2 for the directioni of the Yin-Yang
        Ydirtmp[t-Start_ind[d],:]=(Plane_vec[:,0]*coeff[8]+Plane_vec[:,1]*coeff[9])/np.sqrt(coeff[8]**2+coeff[9]**2)
        #coeff 6 and 7 correspond to PL1 and PL2, for the direction of the three-fold symetric mode
        alpha=np.arctan2(coeff[7],-coeff[6])/3.0
        TFdirtmp[t-Start_ind[d],:]=Plane_vec[:,0]*np.cos(alpha)+Plane_vec[:,1]*np.sin(alpha)
        #new modes defined a bit differently this one is sum PLi + Yi
        cx=coeff[6]+coeff[8]
        cy=coeff[7]+coeff[9]
        Wdirtmp[t-Start_ind[d],:]=(Plane_vec[:,0]*cx+Plane_vec[:,1]*cy)/np.sqrt(cx**2+cy**2)
        cx=coeff[6]-coeff[8]
        cy=coeff[7]-coeff[9]
        RMdirtmp[t-Start_ind[d],:]=(Plane_vec[:,0]*cx+Plane_vec[:,1]*cy)/np.sqrt(cx**2+cy**2)
        
        #coeff 6 and 7 correspond to PL1 and PL2, to be combined with 3 and 4 corresponding to S1, S2
        a1=coeff[3]
        a2=coeff[4]
        b1=coeff[6] 
        b2=coeff[7]
        vx=(a1*b1-a2*b2)
        vy=(-a2*b1-a1*b2)
        PLSdirtmp[t-Start_ind[d],:]=(Plane_vec[:,0]*vx+Plane_vec[:,1]*vy)/np.sqrt(vx**2+vy**2)
        #accumulation here must be compared to omega so we don't do the last point
        if t!=End_ind[d]:
            #it must start at Np_rot[d-1] when t is equal to Start_ind[d]
            Hsq[Np_rot[d-1]+t-Start_ind[d]]=R*np.sqrt(W0+W1+WS+WB+WPL+WY)/np.sqrt(np.pi)
            Hsq2[Np_rot[d-1]+t-Start_ind[d]]=hsq
            MS[Np_rot[d-1]+t-Start_ind[d]]=WS/(WS+WB+WPL+WY)
            MB[Np_rot[d-1]+t-Start_ind[d]]=WB/(WS+WB+WPL+WY)
            MPL[Np_rot[d-1]+t-Start_ind[d]]=WPL/(WS+WB+WPL+WY)
            MY[Np_rot[d-1]+t-Start_ind[d]]=WY/(WS+WB+WPL+WY)
            M0[Np_rot[d-1]+t-Start_ind[d]]=W0/(WS+WB+WPL+WY+W0+W1)
            M1[Np_rot[d-1]+t-Start_ind[d]]=W1/(WS+WB+WPL+WY+W0+W1)
            Mrest[Np_rot[d-1]+t-Start_ind[d]]=(WS+WB+WPL+WY)/(WS+WB+WPL+WY+W0+W1)
            Rint[Np_rot[d-1]+t-Start_ind[d]]=R
            Lamb[Np_rot[d-1]+t-Start_ind[d]]=lamb
            Bo[Np_rot[d-1]+t-Start_ind[d]]=WB*Plane_vec[:,2] #we multiply the plane vector by the bowl mode amplitude
            #to get the vector associated to the bowl.
            #additional test on t in the case of time offset
            if t>=t_offset:
                #it must start at Np_rot_offset[d-1] when t is equal to Start_ind_offset[d]
                Hsq_offset[Np_rot_offset[d-1]+t-Start_ind_offset[d]]=R*np.sqrt(W0+W1+WS+WB+WPL+WY)/np.sqrt(np.pi)
                MS_offset[Np_rot_offset[d-1]+t-Start_ind_offset[d]]=WS/(WS+WB+WPL+WY)
                MB_offset[Np_rot_offset[d-1]+t-Start_ind_offset[d]]=WB/(WS+WB+WPL+WY)
                MPL_offset[Np_rot_offset[d-1]+t-Start_ind_offset[d]]=WPL/(WS+WB+WPL+WY)
                MY_offset[Np_rot_offset[d-1]+t-Start_ind_offset[d]]=WY/(WS+WB+WPL+WY)
        
        #Q tensor 3d of doublet minus interface
        w,v=F.Q_tensor3d_elong(Outer_shell[:,0],Outer_shell[:,1],Outer_shell[:,2])
        indp=np.argmax(w)
        indn=np.argmin(w)
        indm=np.array([0,1,2])
        indm=indm[(indm!=indp) & (indm!=indn)][0]
        wp=w[indp]
        wm=np.delete(w,[indp,indn])
        wn=w[indn]
        #we accumulate the "relative radiuses" instead of the eigenvalues
        #it's sqrt[lambda/lambda_mid] which corresponds to (a/b,1,c/b) for an ellipsoid (a>b>c)
        Wmid.append(1.0)
        Wp.append(np.sqrt(wp/wm))
        Wn.append(np.sqrt(wn/wm))
        Wtime.append(Time[t-Start_ind[d]])
        #for prolate direction of Q tensor, align it with the direction of N12
        if np.sum(Npltmp[t-Start_ind[d],:]*v[:,indp])>0:
            QPtmp[t-Start_ind[d],:]=v[:,indp]
        else:
            QPtmp[t-Start_ind[d],:]=-v[:,indp]
        QNtmp[t-Start_ind[d],:]=v[:,indn]
        QMtmp[t-Start_ind[d],:]=v[:,indm]
        
        inds=np.argmax(np.abs(w))
        S.append(w[inds])
        wrest=np.delete(w,[inds])
        Eta.append(0.5*np.abs(wrest[0]-wrest[1]))
        
        #F.show_doublet(Outer_shell,Interface,Plane_vec[:,2],Com1[t-Start_ind[d],:],Com2[t-Start_ind[d],:])
    #here autocorrel of wb
    Bowl_ac.append(F.autocorrel(Wb_signed))
    
    #Cross product of Qp(t) and Qp(t+Dt)
    OmQ[Np_rot[d-1]:Np_rot[d],:] = np.cross(QPtmp[:-1],QPtmp[1:])
    
    #accumulate arrays by removing last points to compare with rotation vector
    QP[Np_rot[d-1]:Np_rot[d],:]=QPtmp[:-1]
    QN[Np_rot[d-1]:Np_rot[d],:]=QNtmp[:-1]
    QM[Np_rot[d-1]:Np_rot[d],:]=QMtmp[:-1]
    #Same thing for the plane vector (its sign is random and does not matter)
    Npl[Np_rot[d-1]:Np_rot[d],:]=Npltmp[:-1]
    Sdir[Np_rot[d-1]:Np_rot[d],:]=Sdirtmp[:-1]
    Ydir[Np_rot[d-1]:Np_rot[d],:]=Ydirtmp[:-1]
    TFdir[Np_rot[d-1]:Np_rot[d],:]=TFdirtmp[:-1]
    PLSdir[Np_rot[d-1]:Np_rot[d],:]=PLSdirtmp[:-1]
    Wdir[Np_rot[d-1]:Np_rot[d],:]=Wdirtmp[:-1]
    RMdir[Np_rot[d-1]:Np_rot[d],:]=RMdirtmp[:-1]
    #also creates a rotational velocity vector in the plane of the interface (N12xOmega)
    N12xOm[Np_rot[d-1]:Np_rot[d],:]=np.cross(Npltmp[:-1],Nrot)
    
#normalize all N12xOm vectors
norm=np.sqrt(np.sum(N12xOm**2,axis=1))
N12xOm=N12xOm/norm[:,None]
#normalize OmQ
norm=np.sqrt(np.sum(OmQ**2,axis=1))
OmQ=OmQ/norm[:,None]


%matplotlib qt 

#------------------------------------------------------------------------------
#FIGURE 1, panel f, example with doublet number 6
#------------------------------------------------------------------------------

d=6
#smooth the data with a gaussian filter
omsmooth=F.smoothing_gaussian(Omamp[Np_rot[d-1]:Np_rot[d]],5)

Tarray=np.array([i*10 for i in range(0,Np_rot[d]-Np_rot[d-1])])
plt.figure()
plt.plot(Tarray,Omamp[Np_rot[d-1]:Np_rot[d]],'mo-')
plt.plot(Tarray,omsmooth,'-')
plt.ylim([0,1.1])
plt.ylabel('$\omega$ (revolution.h$^{-1}$)')
plt.xlabel('Time after cell division (min)')


#------------------------------------------------------------------------------
#FIGURE 1, panel g, rotation rate as a function of time.
#------------------------------------------------------------------------------
tM=0
for d in range(1,nd+1):
    #last time point in min
    if 10*(Np_rot[d]-Np_rot[d-1]-1) > tM:
        tM=10*(Np_rot[d]-Np_rot[d-1]-1)


nbins=20
data=[]
x=[]
for d in range(1,nd+1):
    data+=list(Omamp[Np_rot[d-1]:Np_rot[d]])
    x+=[t*10 for t in range(0,Np_rot[d]-Np_rot[d-1])]
x_bins,y_mean,y_std,nvalues = F.bin_plot(x, data, tM)
    
plt.figure()
for d in range(1,nd+1):
    plt.plot(np.array([t*10 for t in range(0,Np_rot[d]-Np_rot[d-1])]),Omamp[Np_rot[d-1]:Np_rot[d]],'k--',alpha=0.2)
plt.errorbar(x_bins[nvalues>1],y_mean[nvalues>1],yerr=y_std[nvalues>1],fmt='r-')
plt.ylabel('$\omega$ (revolution.h$^{-1}$)')
plt.xlabel('Time after cell division (min)')

#------------------------------------------------------------------------------
#FIGURE 1, panel h (also EXTENDED FIGURE 3)  
#3d visualisation of the rotation vector as a function of time
#------------------------------------------------------------------------------
d=6
F.show_omega(Nrot_list[d-1][0],Nrot_list[d-1][1])

#------------------------------------------------------------------------------
#FIGURE 1, panel i, autocorrelation of the normalised rotation vector
#------------------------------------------------------------------------------
tM=0
for d in range(0,nd):
    if 10*(Om_correl[d].shape[0]-1) > tM:
        tM=10*(Om_correl[d].shape[0]-1)

nbins=20
data=[]
x=[]
for d in range(0,nd):
    data+=list(Om_correl[d])
    x+=[t*10 for t in range(0,Om_correl[d].shape[0])]
x_bins,y_mean,y_std,nvalues = F.bin_plot(x, data, tM)

plt.figure()
for d in range(0,nd):
    plt.plot(np.array([t*10 for t in range(0,Om_correl[d].shape[0])]),Om_correl[d],'k--',alpha=0.2)
plt.errorbar(x_bins[nvalues>1],y_mean[nvalues>1],yerr=y_std[nvalues>1],fmt='r-')
plt.ylabel(r'$\langle \omega_n(t)\cdot \omega_n(t+\Delta t)\rangle $')
plt.xlabel(r'$\Delta t$ in (min)')
plt.ylim([-1,1])


#------------------------------------------------------------------------------
#FIGURE 2, panel b, doublet elongation as a function of time
#------------------------------------------------------------------------------
  
Wp=np.array(Wp)
Wn=np.array(Wn)
Wmid=np.array(Wmid)
Wdiff=np.array(Wp)
S=np.array(S)
Eta=np.array(Eta)


nbins=20
tM=np.max(Wtime)
tm=np.min(Wtime)
dt=(tM-tm)/nbins
Wpb=np.zeros(nbins)
Wnb=np.zeros(nbins)
Wmidb=np.zeros(nbins)
Wdiffb=np.zeros(nbins)
Sb=np.zeros(nbins)
Etab=np.zeros(nbins)
Ep=np.zeros(nbins)
En=np.zeros(nbins)
Emid=np.zeros(nbins)
Ediff=np.zeros(nbins)
ES=np.zeros(nbins)
EEta=np.zeros(nbins)
for i in range(nbins):
    ind=np.logical_and(Wtime>=i*dt,Wtime<=(i+1)*dt)
    Wpb[i]=np.mean(Wp[ind])
    Wmidb[i]=np.mean(Wmid[ind])
    Wnb[i]=np.mean(Wn[ind])
    Wdiffb[i]=np.mean(Wp[ind]+Wn[ind]-2)
    Sb[i]=np.mean(S[ind])
    Etab[i]=np.mean(Eta[ind])
    
    Ep[i]=1.96*np.std(Wp[ind])/np.sqrt(len(Wp[ind]))
    Emid[i]=1.96*np.std(Wmid[ind])/np.sqrt(len(Wp[ind]))
    En[i]=1.96*np.std(Wn[ind])/np.sqrt(len(Wp[ind]))
    Ediff[i]=1.96*np.std(Wp[ind]+Wn[ind]-2)/np.sqrt(len(Wp[ind]))
    ES[i]=1.96*np.std(S[ind])/np.sqrt(len(S[ind]))
    EEta[i]=1.96*np.std(Eta[ind])/np.sqrt(len(Eta[ind]))
    
                                    
    
Timeb=np.array([0.5*dt + i*dt for i in range(nbins)])


fig=plt.figure(figsize=(7*cm,5.9*cm))
plt.plot(Wtime,Wp,'r+',alpha=0.1,markersize=5)
plt.plot(Wtime,Wn,'b+',alpha=0.1,markersize=5)
plt.plot(Wtime,Wmid,'g-',alpha=0.1,markersize=5)
plt.errorbar(Timeb, Wpb,yerr=Ep,fmt='r-',linewidth=1)
plt.errorbar(Timeb, Wmidb,yerr=Emid,fmt='g-',linewidth=1)
plt.errorbar(Timeb, Wnb,yerr=En,fmt='b-',linewidth=1)
plt.xlabel('Time after division (min)',fontsize =7, labelpad = 3, fontname=fname)
plt.ylabel(r'Relative elongations ($a/b$,1,$c/b$)',fontsize =7, labelpad = 3, fontname=fname)
plt.tick_params(axis='both', which='major', labelsize=7, pad=2)

fig=plt.figure(figsize=(3.75*cm,1.5*cm))
plt.plot(Wtime,np.array(Wp)+np.array(Wn)-2,'k+',alpha=0.1,markersize=3)
plt.errorbar(Timeb, Wdiffb,yerr=Ediff,fmt='k-',linewidth=1)
#plt.xlabel('Time after division (min)',fontsize=5, labelpad = 3, fontname=fname)
#plt.ylabel('Elongation difference',fontsize =5, labelpad = 3, fontname=fname)
plt.tick_params(axis='both', which='major', labelsize=5, pad=2)
plt.ylabel('(a+c-2)/b',fontsize =7, labelpad = 3, fontname=fname)
plt.ylim([0,0.5])

#------------------------------------------------------------------------------
#FIGURE 2, panel c, correlation between elongation and rotation
#------------------------------------------------------------------------------

#Rotation vector vs Prolate/oblate/middle direction of Q tensor
OQP=1.5*(np.sum(Omegas*QP,axis=1)**2)-0.5
OQN=1.5*(np.sum(Omegas*QN,axis=1)**2)-0.5
OQM=1.5*(np.sum(Omegas*QM,axis=1)**2)-0.5
#control random data
#vec=np.random.normal(0,1,Omegas.shape)
#norm=np.sqrt(np.sum(vec**2,axis=1))
#vec=vec/norm[:,None]
#Octrl=1.5*(np.sum(Omegas*vec,axis=1)**2)-0.5
data=[OQP,OQM,OQN]

#qa,qb,qc
fig=plt.figure(figsize=(6.37*cm,4.26*cm))
plt.violinplot(data,showmeans=True)
plt.xticks([1,2,3],[r'$q_a$',r'$q_b$',r'$q_c$'],fontsize =7, fontname=fname)
plt.ylabel(r'Correlation with $||\omega||$',fontsize =7, labelpad = 3, fontname=fname)
plt.tick_params(axis='both', which='major', labelsize=7, pad=2)
plt.yticks([-0.5,0,0.5,1])
plt.axhline(0,c='k')

#Or also we compute QP.R12 (realign QP with respect to R12 instead of N12)
norm=np.sqrt(np.sum(R12**2,axis=1))
R12n=R12/norm[:,None]
data=[np.abs(np.sum(QP*R12n,axis=1))]
fig=plt.figure(figsize=(0.5*cm,4.26*cm))
#bp=ax.boxplot(data,whis=[5,95],showmeans=True)
plt.violinplot(data,showmeans=True)
plt.xticks([1],[r'$q_a$'],fontsize =7, fontname=fname)
plt.ylabel(r'$q_a \cdot r_{12}$',fontsize =7, labelpad = 3, fontname=fname)
plt.tick_params(axis='both', which='major', labelsize=7, pad=2)

#statistical tests
pval=F.bootstrap(OQP,False,50000)
print(pval)
pval=F.bootstrap(OQM,True,50000)
print(pval)
pval=F.bootstrap(OQN,True,50000)
print(pval)

pval=F.bootstrap(np.abs(np.sum(QP*R12n,axis=1)),True,50000)
print(pval)


#------------------------------------------------------------------------------
#Binning of variables as a function of omega
#------------------------------------------------------------------------------

#hsq, hsq2 as function of omega
nbins=17
nbins_offset=17
OM=1.2 #np.max(Omamp)
om=np.min(Omamp)
do=(OM-om)/nbins
do_offset=(OM-om)/nbins_offset
Hsqb=np.zeros(nbins)
EHsq=np.zeros(nbins)

Hsq_offsetb=np.zeros(nbins_offset)
EHsq_offset=np.zeros(nbins_offset)
MS_offsetb=np.zeros(nbins_offset)
MB_offsetb=np.zeros(nbins_offset)
MPL_offsetb=np.zeros(nbins_offset)
MY_offsetb=np.zeros(nbins_offset)
ES_offset=np.zeros(nbins_offset)
EB_offset=np.zeros(nbins_offset)
EPL_offset=np.zeros(nbins_offset)
EY_offset=np.zeros(nbins_offset)

Hsq2b=np.zeros(nbins)
EHsq2=np.zeros(nbins)
Hsqadim=np.zeros(nbins)
Ehsqadim=np.zeros(nbins)
MSb=np.zeros(nbins)
MBb=np.zeros(nbins)
MPLb=np.zeros(nbins)
MYb=np.zeros(nbins)
ES=np.zeros(nbins)
EB=np.zeros(nbins)
EPL=np.zeros(nbins)
EY=np.zeros(nbins)

Rintb=np.zeros(nbins)
Lambb=np.zeros(nbins)
ERint=np.zeros(nbins)
ELamb=np.zeros(nbins)

for i in range(nbins):
    ind=np.logical_and(Omamp>=om+i*do,Omamp<=om+(i+1)*do)
    #REMOVE OUTLIERS WHERE HSQ*0.206 >= 4
    #ind=np.logical_and(ind,Hsq*0.206<4)
    Hsqb[i]=np.mean(Hsq[ind])
    Hsq2b[i]=np.mean(Hsq2[ind])
    Hsqadim[i]=np.mean(Hsq[ind]/Rint[ind])
    
    EHsq[i]=1.96*np.std(Hsq[ind])/np.sqrt(len(Hsq[ind]))
    EHsq2[i]=1.96*np.std(Hsq2[ind])/np.sqrt(len(Hsq2[ind]))
    Ehsqadim[i]=1.96*np.std(Hsq[ind]/Rint[ind])/np.sqrt(len(Hsq[ind]))
    
    MSb[i]=np.mean(MS[ind])
    MBb[i]=np.mean(MB[ind])
    MPLb[i]=np.mean(MPL[ind])
    MYb[i]=np.mean(MY[ind])
    
    ES[i]=1.96*np.std(MS[ind])/np.sqrt(len(MS[ind]))
    EB[i]=1.96*np.std(MB[ind])/np.sqrt(len(MB[ind]))
    EPL[i]=1.96*np.std(MPL[ind])/np.sqrt(len(MPL[ind]))
    EY[i]=1.96*np.std(MY[ind])/np.sqrt(len(MY[ind]))
    
    Rintb[i]=np.mean(Rint[ind])
    Lambb[i]=np.mean(Lamb[ind])
    
    ERint[i]=1.96*np.std(Rint[ind])/np.sqrt(len(Rint[ind]))
    ELamb[i]=1.96*np.std(Lamb[ind])/np.sqrt(len(Lamb[ind]))
for i in range(nbins_offset):
    #offset part
    ind=np.logical_and(Omamp_offset>=om+i*do_offset,Omamp_offset<=om+(i+1)*do_offset)
    Hsq_offsetb[i]=np.mean(Hsq_offset[ind])
    EHsq_offset[i]=1.96*np.std(Hsq_offset[ind])/np.sqrt(len(Hsq_offset[ind]))
    
    MS_offsetb[i]=np.mean(MS_offset[ind])
    MB_offsetb[i]=np.mean(MB_offset[ind])
    MPL_offsetb[i]=np.mean(MPL_offset[ind])
    MY_offsetb[i]=np.mean(MY_offset[ind])
    
    ES_offset[i]=1.96*np.std(MS_offset[ind])/np.sqrt(len(MS_offset[ind]))
    EB_offset[i]=1.96*np.std(MB_offset[ind])/np.sqrt(len(MB_offset[ind]))
    EPL_offset[i]=1.96*np.std(MPL_offset[ind])/np.sqrt(len(MPL_offset[ind]))
    EY_offset[i]=1.96*np.std(MY_offset[ind])/np.sqrt(len(MY_offset[ind]))
    
    

                                    
Omb=np.array([om+0.5*do + i*do for i in range(nbins)])
Omb_offset=np.array([om+0.5*do_offset + i*do_offset for i in range(nbins_offset)])


#------------------------------------------------------------------------------
#FIGURE 2, panel f, interface deflection as a function of omega
#------------------------------------------------------------------------------

fig=plt.figure(figsize=(8.2*cm,2.8*cm))
#correct method from analytical integral of modes
#plt.plot(Omamp,Hsq*0.206,'r+',alpha=0.2)
plt.errorbar(Omb, Hsqb*0.206,yerr=EHsq*0.206,fmt='r-',linewidth=1)
#plt.errorbar(Omb_offset,Hsq_offsetb*0.206,yerr=EHsq_offset*0.206,fmt='-g',linewidth=1)
plt.plot(Omb,Hsqb*0.206,'ko',markersize=2)
plt.plot(np.array([0.1,0.7,1.0]),np.array([0.45,1.2,1.6]),'ob',markersize=2) #cherry picked data for figure
plt.xlim([0,1.2])
plt.ylim([0.4,2.2])
plt.xlabel(r'$||\omega}|$ (revolution.$h^{-1}$)',fontsize =7, labelpad = 3, fontname=fname)
plt.ylabel(r'Interface deflection ($\mu m$)',fontsize =7, labelpad = 3, fontname=fname)
plt.tick_params(axis='both', which='major', labelsize=7, pad=2)



#------------------------------------------------------------------------------
#FIGURE 2, panel g, Relative amplitude of modes
#------------------------------------------------------------------------------

Es=1.96*np.std(MS)/np.sqrt(len(MS))
Eb=1.96*np.std(MB)/np.sqrt(len(MB))
Epl=1.96*np.std(MPL)/np.sqrt(len(MPL))
Ey=1.96*np.std(MY)/np.sqrt(len(MY))
error=[Es,Eb,Epl,Ey]

fig=plt.figure(figsize=(6.03*cm,3.2*cm))
plt.bar([1,2,3,4], [np.mean(MS),np.mean(MB),np.mean(MPL),np.mean(MY)], yerr=error, width=0.9,capsize=2)
plt.xticks([1,2,3,4],['Saddle-node','Bowl','Three-fold','Yin-yang'],fontsize =7, fontname=fname)
plt.ylabel('Mode amplitudes (relative)',fontsize =7, labelpad = 3, fontname=fname)
plt.tick_params(axis='both', which='major', labelsize=7, pad=2)


#------------------------------------------------------------------------------
#FIGURE 2, panel h, correlation between rotation and modes orientation
#------------------------------------------------------------------------------
#for the saddle-node correlation we use cos(4*theta) with cos(theta)=Sdir.N12xOm
# 4 because the saddle-node, with respect to the rotation, actually has a 4-attic symmetry
ScorrOm2=np.cos(2*np.arccos(np.sum(Sdir*N12xOm,axis=1)))
ScorrOm4=np.cos(4*np.arccos(np.sum(Sdir*N12xOm,axis=1)))
ScorrOm8=np.cos(8*np.arccos(np.sum(Sdir*N12xOm,axis=1)))
#for the three-fold mode we use cos(3*theta)
TFcorrOm=np.cos(3*np.arccos(np.sum(TFdir*N12xOm,axis=1)))
TFcorrOm6=np.cos(6*np.arccos(np.sum(TFdir*N12xOm,axis=1)))
#yin-yang
Ycorr=np.sum(N12xOm*Ydir,axis=1)

fig=plt.figure(figsize=(6.7*cm,3.2*cm))
data=[ScorrOm4,TFcorrOm,Ycorr]
plt.violinplot(data,showmeans=True)
plt.ylabel(r'Correlation with $\omega$',fontsize =7, labelpad = 3, fontname=fname)
plt.xticks([1,2,3],['Saddle-node', 'Three-fold','Yin-yang'],fontsize =7, fontname=fname)
plt.tick_params(axis='both', which='major', labelsize=7, pad=2)
plt.axhline(0,c='k')

#statistical test
pval=F.bootstrap(Ycorr,True,50000)
print(pval)