#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:33:11 2022

@author: vagne
"""

import numpy as np
from numpy import linalg as LA
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.cm as cm
from random import choices

import pyvista

def read_vtu(name):
    #load vtu file and extract velocities and vertices
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(name)
    reader.Update()  # Needed because of GetScalarRange
    vtkmesh = reader.GetOutput()

    #to polydata
    geometry_filter = vtk.vtkDataSetSurfaceFilter()
    geometry_filter.SetInputData(vtkmesh)
    geometry_filter.Update()
    vtkmesh = geometry_filter.GetOutput()

    # subdiv=vtk.vtkLoopSubdivisionFilter()
    # subdiv.SetInputData(vtkmesh)
    # subdiv.SetNumberOfSubdivisions(0)
    # subdiv.Update()
    # vtkmesh=subdiv.GetOutput()

    pointdata=vtkmesh.GetPointData()
    vx=vtk_to_numpy(pointdata.GetAbstractArray("vx"))
    vy=vtk_to_numpy(pointdata.GetAbstractArray("vy"))
    vz=vtk_to_numpy(pointdata.GetAbstractArray("vz"))

    vel=np.transpose(np.array([vx,vy,vz]))


    #get coordinates of vertices
    points=vtkmesh.GetPoints()
    vert=vtk_to_numpy(points.GetData())

    return vert,vel

def calc_omegas(Dvert,vel):
    #inertia tensor
    I=np.zeros((3,3))

    delta=np.diag([1,1,1])

    sympart=np.sum(Dvert[:,:]*Dvert[:,:],axis=1)

    for a in range(3):
        for b in range(3):
            I[a,b]=np.sum(sympart*delta[a,b]-Dvert[:,a]*Dvert[:,b])

    #angular momentum
    L=np.sum(np.cross(Dvert,vel),axis=0)

    #Rotation vector, return it
    return np.linalg.solve(I,L)

def poly_fit_interface(X,Y,H):
    A=np.array([X*0+1,X**2,X*Y,Y**2,X**3-3*X*Y**2,Y**3-3*X**2*Y,X,Y,3*X**3+X*Y**2,3*Y**3+X**2*Y]).T
    coeff, r, rank,s=np.linalg.lstsq(A,H)

    H2=coeff[0]*(X*0+1)+coeff[1]*X**2+coeff[2]*X*Y+coeff[3]*Y**2 #mode 2 (includes constant term)
    H3f=coeff[4]*(X**3-3*X*Y**2)+coeff[5]*(Y**3-3*X**2*Y) #mode 3 (2 pi/ 3 rotation symmetry)
    H3y=coeff[6]*X+coeff[7]*Y+coeff[8]*(3*X**3+X*Y**2)+coeff[9]*(3*Y**3+X**2*Y) #mode containing Yin-Yang
    Hfit=H2+H3f+H3y

    #extracting the part of H3y which has a line of zeroes y=lambda*x or x=lambda*y
    a=coeff[6]
    b=coeff[7]
    p=coeff[8]
    k=coeff[9]
    #solve for lambda in two different ways
    if abs(k)<abs(p):
        root=fsolve(lambda x: 3*k+p*x+k*x**2+3*p*x**3,0) #x=lambda*y
        b=-a*root#this is the necesarry condition on the mode 1 so that it also has the same line of zeroes
    else:
        root=fsolve(lambda x: 3*p+k*x+p*x**2+3*k*x**3,0) #y=lambda*x
        a=-b*root
    H3ys=a*X+b*Y+p*(3*X**3+X*Y**2)+k*(3*Y**3+X**2*Y) #this is the purely symmetric part of the Yin-Yang
    H1r=(coeff[6]-a)*X+(coeff[7]-b)*Y #this is the remaining mode 1 that is not aligned with the Yin-Yang

    #relative weight of every mode
    #w1r=np.sqrt(np.mean(H1r*H1r))
    #w2=np.sqrt(np.mean(H2*H2))
    #w3f=np.sqrt(np.mean(H3f*H3f))
    #w3ys=np.sqrt(np.mean(H3ys*H3ys))
    #print(f'Mode weights: 0:{w0} 1r:{w1r} 2:{w2} 3f:{w3f} 3ys:{w3ys}')
    #print(f'Total weitght: {wt}')
    return (H1r,H2,H3f,H3ys,Hfit)

def Q_tensor(X,Y):
    #take the points in X,Y, compute their Q tensor and returns basis vectors, effective radius and aspect ratio
    Q=np.zeros((2,2))
    Q[0,0]=np.mean(X*X)
    Q[0,1]=np.mean(X*Y)
    Q[1,0]=Q[0,1]
    Q[1,1]=np.mean(Y*Y)

    w, v = LA.eig(Q)
    if w[1]>=w[0]:
        k=w[0]
        w[0]=w[1]
        w[1]=k
        vk=np.copy(v[:,0])
        v[:,0]=np.copy(v[:,1])
        v[:,1]=np.copy(vk)
    R=2*(w[0]*w[1])**(1/4)
    lamb=(w[0]/w[1])**(1/4)
    return (v,R,lamb)

def Q_tensor3d(X,Y,Z):

    #center the X,Y,Z and normalize them using average distance to center
    Xc=X-np.mean(X)
    Yc=Y-np.mean(Y)
    Zc=Z-np.mean(Z)

    davg=np.sqrt(np.mean(Xc**2+Yc**2+Zc**2))
    Xc=Xc/davg
    Yc=Yc/davg
    Zc=Zc/davg
    Q=np.zeros((3,3))
    Q[0,0]=np.mean(Xc*Xc)
    Q[0,1]=np.mean(Xc*Yc)
    Q[0,2]=np.mean(Xc*Zc)
    Q[1,1]=np.mean(Yc*Yc)
    Q[1,2]=np.mean(Yc*Zc)
    Q[2,2]=np.mean(Zc*Zc)
    tr=Q[0,0]+Q[1,1]+Q[2,2]
    Q[0,0]=Q[0,0]-tr/3.0
    Q[1,1]=Q[1,1]-tr/3.0
    Q[2,2]=Q[2,2]-tr/3.0

    Q[1,0]=Q[0,1]
    Q[2,0]=Q[0,2]
    Q[2,1]=Q[1,2]

    w,v=LA.eig(Q)
    return (w,v)

def Q_tensor3d_elong(X,Y,Z):

    #This returns the eigenvalues of the tensor without normalization and without removing the trace
    #center the X,Y,Z and don't normalize
    Xc=X-np.mean(X)
    Yc=Y-np.mean(Y)
    Zc=Z-np.mean(Z)

    Q=np.zeros((3,3))
    Q[0,0]=np.mean(Xc*Xc)
    Q[0,1]=np.mean(Xc*Yc)
    Q[0,2]=np.mean(Xc*Zc)
    Q[1,1]=np.mean(Yc*Yc)
    Q[1,2]=np.mean(Yc*Zc)
    Q[2,2]=np.mean(Zc*Zc)

    Q[1,0]=Q[0,1]
    Q[2,0]=Q[0,2]
    Q[2,1]=Q[1,2]

    w,v=LA.eig(Q)
    return (w,v)


def bootstrap(data,pos,nsample):
    #give p-value for mean of data being positive or negative
    sample_mean=[]
    for i in range(nsample):
        y=choices(data,k=len(data))
        sample_mean.append(np.mean(y))

    if pos==True:
        return (sample_mean, np.sum(np.array(sample_mean)<0)/len(sample_mean))
    else:
        return (sample_mean, np.sum(np.array(sample_mean)>0)/len(sample_mean))
    

def bootstrap_double(data1,data2,nsample):
    #give p-value for mean of data1 being larger than mean of data2
    sample_diff=[]
    for i in range(nsample):
        y1=np.mean(choices(data1,k=len(data1)))
        y2=np.mean(choices(data2,k=len(data2)))
        sample_diff.append(y1-y2)


    return (sample_diff,np.sum(np.array(sample_diff)<0)/len(sample_diff))


def bootstrap_correl(data,nsample):
    #data=[..., [d0i,d1i], ...]
    #give p-value that correlation of d0 and d1 is positive
    sample_corr=[]
    for i in range(nsample):
        y=choices(data,k=len(data))
        y0=np.array([x[0] for x in y])
        y1=np.array([x[1] for x in y])
        m0=np.mean(y0)
        m1=np.mean(y1)
        sample_corr.append(np.mean((y0-m0)*(y1-m1))/np.std(y0)/np.std(y1))

    return (sample_corr,np.sum(np.array(sample_corr)<0)/len(sample_corr))


def bootstrap_correl_guillaume(x,y,nsample):
    #data=[..., [d0i,d1i], ...]
    #give p-value that correlation of d0 and d1 is positive
    
    corr_ref = np.mean((x-np.mean(x))*(y-np.mean(y)))/np.std(x)/np.std(y)
    
    sample_corr=[]
    for i in range(nsample):
        ys=choices(y,k=len(y))
        xs=choices(x,k=len(x))
        
        mxs=np.mean(xs)
        mys=np.mean(ys)
        
        sample_corr.append(np.mean((xs-mxs)*(ys-mys))/np.std(ys)/np.std(xs))

    return (sample_corr, np.sum(np.array(sample_corr)>corr_ref)/len(sample_corr))

def bootstrap_beta(beta,beta_ref,pos,nsample):
    #give p-value for the average "angle" of beta being larger (True) or
    #lower (False) than beta_ref
    sample_beta_angle=[]
    for i in range(nsample):
        y=choices(beta,k=len(beta))
        vx=np.mean(np.cos(y))
        vy=np.mean(np.sin(y))
        beta_angle=np.arctan2(vy,vx)%(2*np.pi)
        
        sample_beta_angle.append(beta_angle)

    if pos==True:
        return (sample_beta_angle, np.sum(np.array(sample_beta_angle)<beta_ref)/len(sample_beta_angle))
    else:
        return (sample_beta_angle, np.sum(np.array(sample_beta_angle)>beta_ref)/len(sample_beta_angle))



def make_unit_disk(X,Y,v,R,lamb):
    #take points X,Y with effective radius R and aspect ratio lamb
    #v[:,0] is the first eigenvector
    #and bring them back on a unit disk
    #align coordinates with aspect ratio
    Xp=X*v[0,0]+Y*v[1,0]  #xi*e1x+yi*e1y
    Yp=X*v[0,1]+Y*v[1,1]  #xi*e2x+yi*e2y
    #Then bring back to unit disk and back to the original coordinate system
    Xt=Xp/(R*lamb)*v[0,0]+Yp*lamb/R*v[0,1]
    Yt=Xp/(R*lamb)*v[1,0]+Yp*lamb/R*v[1,1]
    return (Xt,Yt)

def poly_fit_orthogonal(X,Y,H):
    #the values in X and Y must be representing an interface with a radius of 1.0 so that the fit and the decomposition makes sense
    sp=np.sqrt(np.pi)
    m0=X*0+1/sp
    m10=2*X/sp
    m11=2*Y/sp
    S1=np.sqrt(6)/sp*(X**2-Y**2)
    S2=2*np.sqrt(6)/sp*X*Y
    B=2*np.sqrt(3)/sp*(X**2+Y**2-0.5)
    PL1=2*np.sqrt(2)/sp*X*(X**2-3*Y**2)
    PL2=2*np.sqrt(2)/sp*Y*(Y**2-3*X**2)
    Yy1=6*np.sqrt(2)/sp*X*(X**2+Y**2-2.0/3.0)
    Yy2=6*np.sqrt(2)/sp*Y*(Y**2+X**2-2.0/3.0)
    A=np.array([m0,m10,m11,S1,S2,B,PL1,PL2,Yy1,Yy2]).T
    coeff, r, rank,s=np.linalg.lstsq(A,H)

    return coeff

def construct_modes(X,Y,coeff):
    sp=np.sqrt(np.pi)
    m0=X*0+1/sp
    m10=2*X/sp
    m11=2*Y/sp
    S1=np.sqrt(6)/sp*(X**2-Y**2)
    S2=2*np.sqrt(6)/sp*X*Y
    B=2*np.sqrt(3)/sp*(X**2+Y**2-0.5)
    PL1=2*np.sqrt(2)/sp*X*(X**2-3*Y**2)
    PL2=2*np.sqrt(2)/sp*Y*(Y**2-3*X**2)
    Yy1=6*np.sqrt(2)/sp*X*(X**2+Y**2-2.0/3.0)
    Yy2=6*np.sqrt(2)/sp*Y*(Y**2+X**2-2.0/3.0)

    H0=coeff[0]*m0
    H1=coeff[1]*m10+coeff[2]*m11
    HS=coeff[3]*S1+coeff[4]*S2
    HB=coeff[5]*B
    HPL=coeff[6]*PL1+coeff[7]*PL2
    HY=coeff[8]*Yy1+coeff[9]*Yy2
    return (H0,H1,HS,HB,HPL,HY)


def show_interface_vid(X,Y,Hfit):
    #create first a plane
    Lx=np.max(X)-np.min(X)
    Ly=np.max(Y)-np.min(Y)
    Mh=np.max(Hfit)
    planemesh=pyvista.Plane(center=(0,0,0),direction=(0,0,1),i_size=Lx,j_size=Ly)


    pts=np.array([X,Y,Hfit]).T
    ptc=pyvista.PolyData(pts)
    pl=pyvista.Plotter() #plotter creation and initialize a movie
    pl.open_movie('test.mp4')
    actor=pl.add_mesh(ptc,color='blue',point_size=10.0,render_points_as_spheres=True)
    actor=pl.add_mesh(planemesh,opacity=0.5)

    pl.camera.position = (-2*Lx,0,Mh)
    pl.camera.focal_point=(0,0,0)

    #pl.show()
    pl.write_frame()#first frame written
    nframes=200 #number of frames to rotate from theta=(1/np)*2*pi to (np-1)/np*2*pi
    for i in range(1,nframes):
        theta=2*np.pi*i/nframes
        pl.camera.position=(-2*Lx*np.cos(theta),-2*Lx*np.sin(theta),Mh)
        pl.write_frame()

    pl.close()

def show_omega(Omegavector,time):

    pyvista.global_theme.font.color = 'black'
    ptc=pyvista.PolyData(Omegavector)
    pl=pyvista.Plotter(window_size=[1600,1600]) #plotter creation and initialize a movie
    #pl.open_movie('test.mp4')
    #actor=pl.add_mesh(ptc,point_size=10.0,render_points_as_spheres=True,scalars=np.linspace(0,1,len(Omegavector)))
    #pl.add_bounding_box()
    #norm=np.max(np.sqrt(np.sum(Omegavector**2,axis=1)))
    #arx=pyvista.Arrow(start=(0,0,0),direction=(1,0,0),scale=norm/10)
    #ary=pyvista.Arrow(start=(0,0,0),direction=(0,1,0),scale=norm/10)
    #arz=pyvista.Arrow(start=(0,0,0),direction=(0,0,1),scale=norm/10)

    #line going from point to point
    npoints=len(Omegavector)

    # Display the points
    pl.add_mesh(ptc,point_size=10.0,render_points_as_spheres=True,scalars=time,show_scalar_bar=False)
    #display lines
    for i in range(npoints-1):
        a0=[Omegavector[i,0],Omegavector[i,1],Omegavector[i,2]]
        a1=[Omegavector[i+1,0],Omegavector[i+1,1],Omegavector[i+1,2]]
        pl.add_lines(np.array([a0,a1]),color=cm.viridis(i/(npoints-1)))
    #display a sphere
    Mnorm=np.max(np.sqrt(np.sum(Omegavector**2,axis=1)))
    sph=pyvista.Sphere(radius=Mnorm,center=(0,0,0))

    pl.add_mesh(sph,opacity=0.2)

    pl.set_background('white')
    pl.show_bounds(location='outer',font_size=10000000,grid=True)

    pl.show()

def show_omega_norm(Omegavector):
    norm=np.sqrt(np.sum(Omegavector**2,axis=1))
    Npts=Omegavector/norm[:,None]
    ptc=pyvista.PolyData(Npts)
    pl=pyvista.Plotter() #plotter creation and initialize a movie
    #pl.open_movie('test.mp4')
    #actor=pl.add_mesh(ptc,point_size=10.0,render_points_as_spheres=True,scalars=np.linspace(0,1,len(Omegavector)))
    #pl.add_bounding_box()
    #norm=np.max(np.sqrt(np.sum(Omegavector**2,axis=1)))
    #arx=pyvista.Arrow(start=(0,0,0),direction=(1,0,0),scale=norm/10)
    #ary=pyvista.Arrow(start=(0,0,0),direction=(0,1,0),scale=norm/10)
    #arz=pyvista.Arrow(start=(0,0,0),direction=(0,0,1),scale=norm/10)

    #line going from point to point
    npoints=len(Omegavector)

    # Display the points
    pl.add_mesh(ptc,point_size=10.0,render_points_as_spheres=True,scalars=np.linspace(0,1,npoints))
    #display lines
    for i in range(npoints-1):
        a0=[Npts[i,0],Npts[i,1],Npts[i,2]]
        a1=[Npts[i+1,0],Npts[i+1,1],Npts[i+1,2]]
        pl.add_lines(np.array([a0,a1]),color=cm.viridis(i/(npoints-1)))
    #display a sphere
    sph=pyvista.Sphere(radius=1.0,center=(0,0,0))

    pl.add_mesh(sph,opacity=0.2)

    pl.show_grid()
    pl.show()

def show_interface(X,Y,Hfit):
    #create first a plane
    Lx=np.max(X)-np.min(X)
    Ly=np.max(Y)-np.min(Y)
    Mh=np.max(Hfit)
    planemesh=pyvista.Plane(center=(0,0,0),direction=(0,0,1),i_size=Lx,j_size=Ly)


    pts=np.array([X,Y,Hfit]).T
    ptc=pyvista.PolyData(pts)
    pl=pyvista.Plotter() #plotter creation and initialize a movie
    pl.open_movie('test.mp4')
    actor=pl.add_mesh(ptc,color='blue',point_size=10.0,render_points_as_spheres=True)
    actor=pl.add_mesh(planemesh,opacity=0.5)

    pl.show()

def show_doublet(Outer_shell,Interface,N,Nrot):
    ptc=pyvista.PolyData(Outer_shell)
    ptc2=pyvista.PolyData(Interface)

    #centroid of interface
    C=np.mean(Interface,axis=0)
    #average size of interface
    S=np.mean(np.sqrt(np.sum((Interface-C)**2,axis=1)))
    planemesh=pyvista.Plane(center=(C[0],C[1],C[2]),direction=(N[0],N[1],N[2]),i_size=2*S,j_size=2*S)
    #arrow for the rotation vector
    #maax distance from centroid of interface
    Md=np.max(np.sqrt(np.sum((Outer_shell-C[None,:])**2,axis=1)))
    ar=pyvista.Arrow(start=(C[0],C[1],C[2]),direction=(Nrot[0],Nrot[1],Nrot[2]),scale=Md)
    #ar2=pyvista.Arrow(start=(C[0],C[1],C[2]),direction=(-Nrot[0],-Nrot[1],-Nrot[2]),scale=Md)

    pl=pyvista.Plotter(window_size=[1000,1500])
    actor=pl.add_mesh(ptc,color='blue')
    actor=pl.add_mesh(ptc2,color='red')
    actor=pl.add_mesh(planemesh,opacity=0.5)
    actor=pl.add_mesh(ar)
    #actor=pl.add_mesh(ar2)
    pl.camera.position = (C[0], C[1]+2*63.436, C[2])
    pl.camera.focal_point = (C[0], C[1], C[2])
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.zoom(0.5)


    pl.show()
    
def show_doublet_polarities(V1,V2,p1,p2,r12,rg,w,r1,r2,val1,val2):
    #create coordinate system based on rg a origin, then r12=ez and w=ex
    ex=w/np.linalg.norm(w)
    ez=r12/np.linalg.norm(r12)
    ey=np.cross(ez,ex)
    
    #new coordinates of points in new reference frame
    V1n=np.zeros(V1.shape)
    V2n=np.zeros(V2.shape)
    V1n[:,0] = np.sum((V1-rg[None,:])*ex,axis=1)
    V1n[:,1] = np.sum((V1-rg[None,:])*ey,axis=1)
    V1n[:,2] = np.sum((V1-rg[None,:])*ez,axis=1)
    
    V2n[:,0] = np.sum((V2-rg[None,:])*ex,axis=1)
    V2n[:,1] = np.sum((V2-rg[None,:])*ey,axis=1)
    V2n[:,2] = np.sum((V2-rg[None,:])*ez,axis=1)
    
    #rescale max distance 1 to origin
    dmax=np.max(np.array([np.max(np.sqrt(np.sum(V1n**2,axis=1))),np.max(np.sqrt(np.sum(V2n**2,axis=1)))]))
    V1n=V1n/dmax
    V2n=V2n/dmax
    
    ptc1=pyvista.PolyData(V1n)
    ptc2=pyvista.PolyData(V2n)
    
    #starting points of arrows in correct reference frame and rescaled
    r1n=[np.sum((r1-rg)*ex)/dmax,np.sum((r1-rg)*ey)/dmax,np.sum((r1-rg)*ez)/dmax]
    r2n=[np.sum((r2-rg)*ex)/dmax,np.sum((r2-rg)*ey)/dmax,np.sum((r2-rg)*ez)/dmax]
    
    #also arrows (polarities) must be rotated into the new reference frane (but not rescaled)
    p1n=[np.sum(p1*ex),np.sum(p1*ey),np.sum(p1*ez)]
    p2n=[np.sum(p2*ex),np.sum(p2*ey),np.sum(p2*ez)]
    
    p1_arrow=pyvista.Arrow(start=(r1n[0],r1n[1],r1n[2]),direction=(p1n[0],p1n[1],p1n[2]),scale=np.linalg.norm(p1n))
    p2_arrow=pyvista.Arrow(start=(r2n[0],r2n[1],r2n[2]),direction=(p2n[0],p2n[1],p2n[2]),scale=np.linalg.norm(p2n))
    
    pl=pyvista.Plotter()
    actor=pl.add_mesh(ptc1,scalars=val1)
    actor=pl.add_mesh(ptc2,scalars=val2)
    actor=pl.add_mesh(p1_arrow)
    actor=pl.add_mesh(p2_arrow)
    
    pl.camera.position = (2, 0, 0)
    pl.camera.focal_point = (0, 0, 0)
    pl.camera.up = (0.0, 0.0, 1.0)
    pl.camera.zoom(0.5)
    
    pl.show()
    
def show_doublet_signal(V1,V2,val1,val2):
    
    
    ptc1=pyvista.PolyData(V1)
    ptc2=pyvista.PolyData(V2)
    
    pl=pyvista.Plotter()
    actor=pl.add_mesh(ptc1,scalars=val1)
    actor=pl.add_mesh(ptc2,scalars=val2)
    pl.add_axes()
    
    pl.show()
    
def show_doublet_outer(V,val):
    ptc1=pyvista.PolyData(V)
    
    pl=pyvista.Plotter()
    actor=pl.add_mesh(ptc1,scalars=val)
    pl.add_axes()
    pl.show()

def read_vtm_info(filepath):
    file=open(filepath,"r")
    for i in range(9):
        line=file.readline()
    time=float(line)
    for i in range(9):
        line=file.readline()
    A1=float(line)
    for i in range(3):
        line=file.readline()
    X1=float(line)
    for i in range(3):
        line=file.readline()
    Y1=float(line)
    for i in range(3):
        line=file.readline()
    Z1=float(line)
    for i in range(3):
        line=file.readline()
    A2=float(line)
    for i in range(3):
        line=file.readline()
    X2=float(line)
    for i in range(3):
        line=file.readline()
    Y2=float(line)
    for i in range(3):
        line=file.readline()
    Z2=float(line)
    C0=np.array([X1/A1,Y1/A1,Z1/A1])
    C1=np.array([X2/A2,Y2/A2,Z2/A2])
    return (time,C0,C1)

def read_vtm_info_r12(filepath):
    #special function for when r12 is also in the file
    file=open(filepath,"r")
    for i in range(18):
        line=file.readline()
    time=float(line)
    for i in range(9):
        line=file.readline()
    A1=float(line)
    for i in range(3):
        line=file.readline()
    X1=float(line)
    for i in range(3):
        line=file.readline()
    Y1=float(line)
    for i in range(3):
        line=file.readline()
    Z1=float(line)
    for i in range(3):
        line=file.readline()
    A2=float(line)
    for i in range(3):
        line=file.readline()
    X2=float(line)
    for i in range(3):
        line=file.readline()
    Y2=float(line)
    for i in range(3):
        line=file.readline()
    Z2=float(line)
    C0=np.array([X1/A1,Y1/A1,Z1/A1])
    C1=np.array([X2/A2,Y2/A2,Z2/A2])
    return (time,C0,C1)

def extract_modes_movie(tstart,tend,filepath):
    W0=np.zeros(tend-tstart+1)
    W1=np.zeros(tend-tstart+1)
    WS=np.zeros(tend-tstart+1)
    WB=np.zeros(tend-tstart+1)
    WPL=np.zeros(tend-tstart+1)
    WY=np.zeros(tend-tstart+1)
    Rtot=np.zeros(tend-tstart+1)
    L=np.zeros(tend-tstart+1)
    hsq=np.zeros(tend-tstart+1)
    for t in range(tend-tstart+1):
        time=t+tstart
        XYH=np.load(filepath+str(time)+'.npy')
        X=XYH[:,0]
        Y=XYH[:,1]
        H=XYH[:,2]
        hsq[t]=np.sqrt(np.mean(H**2))
        v,R,lamb=Q_tensor(X,Y)
        Xt,Yt=make_unit_disk(X,Y,v,R,lamb)
        Rtot[t]=R
        L[t]=lamb
        coeff=poly_fit_orthogonal(Xt,Yt,H/R) #normalization of H so that the size of the interface does not influence the modes
        H0,H1,HS,HB,HPL,HY=construct_modes(Xt,Yt,coeff)
        W0[t]=coeff[0]**2
        W1[t]=coeff[1]**2+coeff[2]**2
        WS[t]=coeff[3]**2+coeff[4]**2
        WB[t]=coeff[5]**2
        WPL[t]=coeff[6]**2+coeff[7]**2
        WY[t]=coeff[8]**2+coeff[9]**2
    return hsq,W0,W1,WS,WB,WPL,WY,Rtot,L

def give_unit_disk_coo(XYH):
    X=XYH[:,0]
    Y=XYH[:,1]
    H=XYH[:,2]
    #hsq=np.sqrt(np.mean(H**2))
    v,R,lamb=Q_tensor(X,Y)
    Xt,Yt=make_unit_disk(X,Y,v,R,lamb)
    return Xt,Yt,H/R

def extract_modes(XYH):
    X=XYH[:,0]
    Y=XYH[:,1]
    H=XYH[:,2]
    #hsq=np.sqrt(np.mean(H**2))
    v,R,lamb=Q_tensor(X,Y)
    Xt,Yt=make_unit_disk(X,Y,v,R,lamb)
    coeff=poly_fit_orthogonal(Xt,Yt,H/R) #normalization of H so that the size of the interface does not influence the modes
    H0,H1,HS,HB,HPL,HY=construct_modes(Xt,Yt,coeff)
    Hfit=H0+H1+HS+HB+HPL+HY
    err=np.sqrt(np.mean((H/R-Hfit)**2))
    hsq=np.sqrt(np.mean((Hfit*R)**2))
    return hsq,coeff,R,lamb,err

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

def rotvec(u,v):
    #find the rotation axis and the angle theta between two vectors (or arrays of vectors)

    nu=np.sqrt(np.sum(u**2,axis=1))
    un=u/nu[:,None]
    nv=np.sqrt(np.sum(v**2,axis=1))
    vn=v/nv[:,None]

    ax=np.cross(u,v)
    norm=np.sqrt(np.sum(ax**2,axis=1))
    ax=ax/norm[:,None]

    th=np.arccos(np.sum(un*vn,axis=1))

    return ax,th




def interp_rot(A,nematic):
    #take vector A of size n and returns vector of size n-1 where element i
    #is an interpolation of the vector i and i+1 of original array
    #ASSUMING THAT the vector is rotating between the frames

    #if nematic is true, we assume that the sign of the vector A is indicating of a nematic object
    #so we can flip the vector always between two time points so that the scalar product is positive
    #if nematic is false we don't flip vectors

    n=A.shape[0]

    Aint=np.zeros((n-1,3))

    sp=np.sign(np.sum(A[:-1,:]*A[1:,:],axis=1)) #scalar product array (sign of it is important)
    sp[sp==0]=1 #correct case where there is perfect perpendicularity

    if nematic==True:
        ax,th=rotvec(A[:-1],A[1:]*sp[:,None]) #if scalar product was negative we flip the second vector
    else:
        ax,th=rotvec(A[:-1],A[1:])

    M=rotmat(ax,th/2) #rotation matrix to rotate of half the angle between two time points

    Aint=np.einsum(M,[0,1,2],A[:-1],[0,2],[0,1]) #now rotate all the inital vectors of half the angle (sum on last axis of M and
    #last axis of As)

    #then these vectors amplitudes must be rescaled
    nA=np.sqrt(np.sum(A[:-1,:]*A[:-1,:],axis=1))
    nA1=np.sqrt(np.sum(A[1:,:]*A[1:,:],axis=1))
    Aint=0.5*Aint*(1+nA1[:,None]/nA[:,None])


    return Aint