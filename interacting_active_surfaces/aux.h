//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#ifndef aux_TissSim_h
#define aux_TissSim_h

#include <random>

#include "ias_Tissue.h"
#include "ias_TissueGen.h"
#include "ias_Integration.h"
#include "ConfigFile.h"

void internal(Teuchos::RCP<ias::SingleIntegralStr> fill);
void interaction(Teuchos::RCP<ias::DoubleIntegralStr> fill);

inline double SmoothStep(double rc1, double rc2, double r)
{
    if(r<rc1)
    {
        return 1.0;
    }
    else if (r<rc2)
    {
        double x =(r - rc2)/(rc2 - rc1);
        double x3 = x*x*x;
        double x4 = x3*x;
        double x5=x4*x;
        return -6.0*x5 - 15.0*x4 - 10.0 * x3;
    }
    else
    {
        return 0.0;
    }
}

inline double dSmoothStep(double rc1, double rc2, double r)
{
    if(r<rc1 or r>rc2)
    {
        return 0.0;
    }
    else
    {
        double x =(r - rc2)/(rc2 - rc1);
        double x2 = x*x;
        double x3 = x2*x;
        double x4 = x3*x;
        return (-30.0*x4 - 60.0*x3 - 30.0 * x2)/(rc2 - rc1);
    }
}

inline double ddSmoothStep(double rc1, double rc2, double r)
{
    if(r<rc1 or r>rc2)
    {
        return 0.0;
    }
    else
    {
        double x =(r - rc2)/(rc2 - rc1);
        double x2 = x*x;
        double x3 = x2*x;
        return (-120.0*x3 - 180.0*x2 - 60.0 * x)/(rc2 - rc1)/(rc2 - rc1);
    }
}

inline double MorsePotential(double D, double r0, double w, double r)
{
    return D * ((exp(-2.0*(r-r0)/w)-2.0*exp(-(r-r0)/w)));
}

inline double dMorsePotential(double D, double r0, double w, double r)
{
    return D * 2.0/w * (-exp(-2.0*(r-r0)/w)+exp(-(r-r0)/w));
}

inline double ddMorsePotential(double D, double r0, double w, double r)
{
    return D * 2.0/(w*w) * (2.0 * exp(-2.0*(r-r0)/w)-exp(-(r-r0)/w));
}

inline double ModMorsePotential(double D, double r0, double w, double rc1, double rc2, double r)
{
    return MorsePotential(D, r0, w, r) * SmoothStep(rc1, rc2, r);
}
inline double dModMorsePotential(double D, double r0, double w, double rc1, double rc2, double r)
{
    return dMorsePotential(D, r0, w, r) * SmoothStep(rc1, rc2, r) + MorsePotential(D, r0, w, r) * dSmoothStep(rc1, rc2, r);
}
inline double ddModMorsePotential(double D, double r0, double w, double rc1, double rc2, double r)
{
    return ddMorsePotential(D, r0, w, r) * SmoothStep(rc1, rc2, r) + 2.0 * dMorsePotential(D, r0, w, r) * dSmoothStep(rc1, rc2, r) + MorsePotential(D, r0, w, r) * ddSmoothStep(rc1, rc2, r);
}

inline double ConfinementPotential(double k, Tensor::tensor<double,1>& centre, Tensor::tensor<double,1>& direction, double h, Tensor::tensor<double,1>& x)
{
    double z = (x-centre)*direction;

    double pot{};
    if(z > 0.5 * h)
    {
        double aux = (z-0.5*h);
        pot = k/3.0 * aux * aux * aux;
    }
    else if(z < -0.5 * h)
    {
        double aux = z+0.5*h;
        pot = -k/3.0 * aux * aux * aux;
    }

    return pot;
}

inline Tensor::tensor<double,1> dConfinementPotential(double k, Tensor::tensor<double,1>& centre, Tensor::tensor<double,1>& direction, double h, Tensor::tensor<double,1>& x)
{
    double z = (x-centre)*direction;

    Tensor::tensor<double,1> dpot = {0.0, 0.0, 0.0};

    if(z > 0.5 * h)
    {
        double aux = (z-0.5*h);
        dpot = k * aux * aux * direction;
    }
    else if(z < -0.5 * h)
    {
        double aux = z+0.5*h;
        dpot = -k * aux * aux * direction;
    }

    return dpot;
}

inline Tensor::tensor<double,2> ddConfinementPotential(double k, Tensor::tensor<double,1>& centre, Tensor::tensor<double,1>& direction, double h, Tensor::tensor<double,1>& x)
{
    double z = (x-centre)*direction;

    Tensor::tensor<double,2> ddpot(3,3);

    ddpot = 0.0;
    if(z > 0.5 * h)
    {
        double aux = (z-0.5*h);
        ddpot = 2.0 * k * aux * Tensor::outer(direction, direction);
    }
    else if(z < -0.5 * h)
    {
        double aux = z+0.5*h;
        ddpot = -2.0 * k * aux * Tensor::outer(direction, direction);
    }

    return ddpot;
}

//building the vector joining the two cell centers
inline Tensor::tensor<double,1> getr12(double X1,double Y1,double Z1,double A1,double X2,double Y2,double Z2,double A2)
{
    Tensor::tensor<double,1> r12={0.0,0.0,0.0};
    r12(0)=(X2/A2-X1/A1);
    r12(1)=(Y2/A2-Y1/A1);
    r12(2)=(Z2/A2-Z1/A1);

    return r12;
}

inline Tensor::tensor<double,1> cross(Tensor::tensor<double,1>& a, Tensor::tensor<double,1>& b)
{
    //cross product between two 3d vectors
    Tensor::tensor<double,1> cp={0.0,0.0,0.0};
    cp(0)=a(1)*b(2)-a(2)*b(1);
    cp(1)=a(2)*b(0)-a(0)*b(2);
    cp(2)=a(0)*b(1)-a(1)*b(0);
    return cp;
}

//building polarity vectors in spherical coordinates in the reference frame (r12,ez x r12, ez)
//theta is the angle between r12 and the projected polarity in (r12,ezxr12) 
//phi is the angle between ez and the polarity
//this builds p1 with theta1 and p2 with theta2, p2's definition of phi is the same, but theta 2 is from -r12 to the projected polarity in (r12,ezxr12)
inline void getpol(Tensor::tensor<double,1>& r12,double theta1, double theta2,double phi,Tensor::tensor<double,1>& p1,Tensor::tensor<double,1>& p2)
{
    //normalized r12
    Tensor::tensor<double,1> r12n=r12/sqrt(r12*r12);
    //build vector ey perpendicular  to ez and r12 using cross product ez x r12
    Tensor::tensor<double,1> ez={0.0,0.0,1.0};
    Tensor::tensor<double,1> ey=cross(ez,r12n);
    ey=ey/sqrt(ey*ey);

    p1=sin(phi)*cos(theta1)*r12n+sin(phi)*sin(theta1)*ey+cos(phi)*ez;
    p2=-sin(phi)*cos(theta2)*r12n-sin(phi)*sin(theta2)*ey+cos(phi)*ez; // pay attention to minus sign, p1=-p2 in plane r12n,ey if theta1=theta2
}


#endif //aux_mzg_h
