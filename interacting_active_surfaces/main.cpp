//    **************************************************************************************************
//    ias - Interacting Active Surfaces
//    Project homepage: https://github.com/torressancheza/ias
//    Copyright (c) 2020 Alejandro Torres-Sanchez, Max Kerr Winter and Guillaume Salbreux
//    **************************************************************************************************
//    ias is licenced under the MIT licence:
//    https://github.com/torressancheza/ias/blob/master/licence.txt
//    **************************************************************************************************

#include <iostream>
#include <fstream>
#include <random>

#include <ias_Belos.h>

#include "aux.h"
#include "ias_ParametrisationUpdate.h"


int main(int argc, char **argv)
{
    using namespace Tensor;
    using namespace std;
    using namespace ias;
    using Teuchos::RCP;
    using Teuchos::rcp;
        
    MPI_Init(&argc, &argv);
    
    //---------------------------------------------------------------------------
    // [0] Input parameters
    //---------------------------------------------------------------------------
    double d{2.1};
    double R{1.0};
    int nSubdiv{3};
    
    bool restart{false};
    string resLocation;
    string resFileName;

    double     intEL = 1.E-1;
    double     intCL = 5.E-2;
    double     intSt = 1.0;
    double   tension0 = 1.E0; //base tension cell 0
    double   tension1 = 1.0; //base tension cell 1
    double  dtension0 = 0.0; //modulation around the base
    double  dtension1 = 0.0; //modulation around the base
    double narrowness0 = 1.0; //how narrow is the myosin spot (b larger means more narrow)
    double narrowness1= 1.0;
    double dtension_nem_0=0.0; //nematic profile of tension
    double dtension_nem_1=0.0;
    double laser_ab_t=10.0; //time at each both cell's tension gradients are modified
    double dtension0_new=0.0; //new value for the tension gradient after t=laser_ab_t
    double dtension1_new=0.0;
    double additional_spot_t=1e6; //time at which additional spot simulating optogenetic activation is created on cell 0
    double additional_spot_end_t=1e6; // time at which the spot is destroyed.
    double dtension_spot=0.0; //tension gradient to put on the spot
    double     kappa = 1.E-2;
    double viscosity = 1.E0;
    double frictiont = 1.E-3;
    double frictionn = 1.E-3;
    double dfriction = 0; //modulation of friction along polarities
    bool anis_friction_r12{false}; //if true, it will be modulated along the r12 axis instead
    double theta1=45*M_PI/180; //angle of polarity with r12 (cell 1)
    double theta2=45*M_PI/180; //same for cell 2
    double phi=M_PI/2.0; //tilt of polarities along z vector

    int ablation_done=0; //ablation not performed at t=0, this is not an input parameter
    bool new_spot_started=false; //no additional spot at t=0
    
    double totTime{1.E3};
    double deltat{1.E-2};
    double stepFacsuccess{0.9};
    double stepFacfail{0.3};
    double maxDeltat{1.0};

    int    nr_maxite{5};
    double nr_restol{1.E-8};
    double nr_soltol{1.E-8};

    bool remesh{false};

    //parameters for polarity and tension pattern
    Tensor::tensor<double,1> r12;
    Tensor::tensor<double,1> p1(3),p2(3);
    Tensor::tensor<double,1> dx(3);
    Tensor::tensor<double,1> x(3);
    Tensor::tensor<double,1> x_com(3);
    Tensor::tensor<double,1> p1_op(3);
    Tensor::tensor<double,1> p2_op(3);
    double bspot=4.0;
    double ct,b,tension_base,tension_mod,tension_mod2,tension,frn,tension_nem;
    tensor<double,1> y_vec(3);
    y_vec(0)=0.0;
    y_vec(1)=1.0;
    y_vec(2)=0.0;

    string  fEnerName{"energies.txt"};
    ofstream fEner;
    
    if(argc == 2)
    {
        const char *config_filename = argv[1];
        ConfigFile config(config_filename);

        config.readInto(       d, "d");

        config.readInto(        R, "R");

        config.readInto(  nSubdiv, "nSubdiv");

        config.readInto(    intEL, "intEL");
        config.readInto(    intCL, "intCL");
        config.readInto(    intSt, "intSt");
        config.readInto( tension0 , "tension0");
        config.readInto( tension1 , "tension1");
        config.readInto( dtension0 , "dtension0");
        config.readInto( dtension1 , "dtension1");
        config.readInto( dtension_nem_0 , "dtension_nem_0");
        config.readInto( dtension_nem_1 , "dtension_nem_1");
        config.readInto(narrowness0,"narrowness0");
        config.readInto(narrowness1,"narrowness1");
        config.readInto(laser_ab_t, "laser_ab_t");
        config.readInto( dtension0_new , "dtension0_new");
        config.readInto( dtension1_new , "dtension1_new");
        config.readInto(    kappa, "kappa");
        config.readInto(viscosity, "viscosity");
        config.readInto(frictiont, "frictiont");
        config.readInto(frictionn, "frictionn");
        config.readInto(dfriction,"dfriction");
        config.readInto(anis_friction_r12,"anis_friction_r12");

        config.readInto(additional_spot_t,"additional_spot_t");
        config.readInto(additional_spot_end_t,"additional_spot_end_t");
        config.readInto(dtension_spot,"dtension_spot");

        config.readInto(theta1 , "theta1");
        config.readInto(theta2 , "theta2");
        config.readInto(phi , "phi");
        
        config.readInto(  totTime,   "totTime");
        config.readInto(   deltat,   "deltat");
        config.readInto(  stepFacsuccess,   "stepFacsuccess");
        config.readInto(  stepFacfail,   "stepFacfail");
        config.readInto(  maxDeltat, "maxDeltat");

        config.readInto(nr_maxite, "nr_maxite");
        config.readInto(nr_restol, "nr_restol");
        config.readInto(nr_soltol, "nr_soltol");

        config.readInto(fEnerName, "fEnerName");
        
        config.readInto(restart, "restart");
        config.readInto(resLocation, "resLocation");
        config.readInto(resFileName, "resFileName");

        config.readInto(remesh, "remesh");

    }
    //---------------------------------------------------------------------------


    RCP<Tissue> tissue;
    if(!restart)
    {
        RCP<TissueGen> tissueGen = rcp( new TissueGen);
        tissueGen->setBasisFunctionType(BasisFunctionType::LoopSubdivision);
        
        tissueGen->addNodeFields({"vx","vy","vz"});
        tissueGen->addNodeFields({"x0","y0","z0"});
        tissueGen->addNodeFields({"vx0","vy0","vz0"});
        tissueGen->addNodeField("tension"); //field to record the active tension profile on a cell
        tissueGen->addNodeField("friction"); //field to record the friction profile on a cell

        tissueGen->addCellFields({"P", "P0"});
        tissueGen->addCellFields({"intEL","intCL","intSt","tension","dtension","dtension_nem","narrowness","kappa","viscosity","frictiont","frictionn","dfriction","anis_friction_r12"});
        tissueGen->addCellFields({"V0"});

        //additional cell field just used by cell 0 to know whether to put an additional spot or not
        //activated_spot is a double but it will be understood as a boolean (0.0 or 1.0)
        tissueGen->addCellFields({"dtension_spot","activated_spot"});

        //cell polarities can be cell fields because one cell does not need to access another cell's polarity
        tissueGen->addCellFields({"px","py","pz"});

        //we put r12 into a tissue field because both cells might need it for anisotropic friction
        tissueGen->addTissFields({"R12X","R12Y","R12Z"});
        
        tissueGen->addTissFields({"time", "deltat"});
        tissueGen->addTissField("Ei");
        //center of masses and areas are tissue fields because we need to be able to build r_12
        //in any local partition (we also need values at 'previous time step' of these so that 
        //we can use a constant tension on vertices during newton-raphson algorithm)
        tissueGen->addTissFields({"A1","X1","Y1","Z1"});
        tissueGen->addTissFields({"A2","X2","Y2","Z2"});
        tissueGen->addTissFields({"A10","X10","Y10","Z10"});
        tissueGen->addTissFields({"A20","X20","Y20","Z20"});

        //default should be icosahedron meshes
        tissue = tissueGen->genRegularGridSpheres(2, 1, 1, d, 0, 0, R, nSubdiv);
        
        
        if(remesh)
        {
            for(int i = 0; i < tissue->getLocalNumberOfCells(); i++)
            {
                auto cell = tissue->GetCell(i,idxType::local);
                int nElem = cell->getNumberOfElements();
                cell->remesh(4.0*M_PI/nElem);
            }
        }
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();
        tissue->balanceDistribution();
                
        tissue->getTissField("time") = 0.0;
        tissue->getTissField("deltat") = deltat;
        tissue->getTissField("Ei") = 0.0;

        //calculate initial polarities and r12 knowing that:
        //(the system starts with spheres of radius 1 with centers disposed in (0,0,0) and (2.1,0,0)
        r12=getr12(0,0,0,1,2.1,0,0,1);
        getpol(r12,theta1,theta2,phi,p1,p2);

        tissue->getTissField("R12X") = r12(0);
        tissue->getTissField("R12Y") = r12(1);
        tissue->getTissField("R12Z") = r12(2);

        tissue->getTissField("A10") = 4.0*M_PI*R*R;
        tissue->getTissField("A20") = 4.0*M_PI*R*R;
        tissue->getTissField("X10") = 0.0;
        tissue->getTissField("Y10") = 0.0;
        tissue->getTissField("Z10") = 0.0;
        tissue->getTissField("X20") = 2.1;
        tissue->getTissField("Y20") = 0.0;
        tissue->getTissField("Z20") = 0.0;

        //these are parameters to put into cells only when we start from scratch
        for(auto cell: tissue->getLocalCells())
        { 
            cell->getCellField("activated_spot")=0.0;
            if(cell->getCellField("cellId")==0)
            {
                cell->getCellField("px")=p1(0);
                cell->getCellField("py")=p1(1);
                cell->getCellField("pz")=p1(2);
            }
            else
            {
                cell->getCellField("px")=p2(0);
                cell->getCellField("py")=p2(1);
                cell->getCellField("pz")=p2(2);
            }
            cell->getCellField("V0") = 4.0*M_PI/3.0;
        }
    }
    else
    {
        tissue = rcp(new Tissue);
        tissue->loadVTK(resLocation, resFileName, BasisFunctionType::LoopSubdivision);
        tissue->Update();
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->balanceDistribution();
        tissue->updateGhosts();
        
        deltat = tissue->getTissField("deltat");
        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("vx") *= deltat;
            cell->getNodeField("vy") *= deltat;
            cell->getNodeField("vz") *= deltat;
        }
    }

    //here parameters to put into cells in any case (restart or not), we can update some parameters if we want.
    for(auto cell: tissue->getLocalCells())
    {
        cell->getCellField("intEL") = intEL;
        cell->getCellField("intCL") = intCL;
        cell->getCellField("intSt") = intSt;
        cell->getCellField("kappa") = kappa;
        cell->getCellField("dfriction")=dfriction;
        cell->getCellField("anis_friction_r12")=anis_friction_r12;
        cell->getCellField("dtension_spot")=dtension_spot;
        if(cell->getCellField("cellId")==0)
        {
            cell->getCellField("tension")=tension0;
            cell->getCellField("dtension")=dtension0;
            cell->getCellField("dtension_nem")=dtension_nem_0;
            cell->getCellField("narrowness")=narrowness0;
        }
        else
        {
            cell->getCellField("tension")=tension1;
            cell->getCellField("dtension")=dtension1;
            cell->getCellField("dtension_nem")=dtension_nem_1;
            cell->getCellField("narrowness")=narrowness1;
        }
        cell->getCellField("viscosity") = viscosity;
        cell->getCellField("frictiont") = frictiont;
        cell->getCellField("frictionn") = frictionn;
    }

    tissue->saveVTK("Cell","_t"+to_string(0));

    RCP<ParametrisationUpdate> paramUpdate = rcp(new ParametrisationUpdate);
    paramUpdate->setTissue(tissue);
    paramUpdate->setMethod(ParametrisationUpdate::Method::ALE);
    paramUpdate->setRemoveRigidBodyTranslation(false);
    paramUpdate->setRemoveRigidBodyRotation(false);
    paramUpdate->setDisplacementFieldNames({"vx","vy","vz"});
    paramUpdate->Update();

    RCP<Integration> physicsIntegration = rcp(new Integration);
    physicsIntegration->setTissue(tissue);
    physicsIntegration->setNodeDOFs({"vx","vy","vz"});
    physicsIntegration->setCellDOFs({"P"});
    physicsIntegration->setSingleIntegrand(internal);
    physicsIntegration->setDoubleIntegrand(interaction);
    physicsIntegration->setNumberOfIntegrationPointsSingleIntegral(3);
    physicsIntegration->setNumberOfIntegrationPointsDoubleIntegral(3);
    physicsIntegration->setTissIntegralFields({"Ei","A1","X1","Y1","Z1","A2","X2","Y2","Z2","A10","X10","Y10","Z10","A20","X20","Y20","Z20"});
    physicsIntegration->setCellDOFsInInteractions(false);
    physicsIntegration->setDisplacementFieldNames("vx","vy","vz");
    physicsIntegration->setCutoffLength(intEL+3.0*intCL);
    physicsIntegration->Update();


    RCP<solvers::TrilinosBelos> physicsLinearSolver = rcp(new solvers::TrilinosBelos);
    physicsLinearSolver->setIntegration(physicsIntegration);
    physicsLinearSolver->setSolverType("GMRES");
    physicsLinearSolver->setMaximumNumberOfIterations(5000);
    physicsLinearSolver->setResidueTolerance(1.E-8);
    physicsLinearSolver->Update();
    
    RCP<solvers::NewtonRaphson> physicsNewtonRaphson = rcp(new solvers::NewtonRaphson);
    physicsNewtonRaphson->setLinearSolver(physicsLinearSolver);
    physicsNewtonRaphson->setSolutionTolerance(nr_soltol);
    physicsNewtonRaphson->setResidueTolerance(nr_restol);
    physicsNewtonRaphson->setMaximumNumberOfIterations(nr_maxite);
    physicsNewtonRaphson->setVerbosity(true);
    physicsNewtonRaphson->setUpdateInteractingGaussPointsPerIteration(true);
    physicsNewtonRaphson->Update();

    int step{};
    double time = tissue->getTissField("time");

    fEner.open (fEnerName);
    fEner.close();

    int conv{};
    bool rec_str{};
    while(time < totTime)
    {
        for(auto cell: tissue->getLocalCells())
        {
            cell->getNodeField("x0")  = cell->getNodeField("x");
            cell->getNodeField("y0")  = cell->getNodeField("y");
            cell->getNodeField("z0")  = cell->getNodeField("z");
            
            cell->getNodeField("vx0") = cell->getNodeField("vx");
            cell->getNodeField("vy0") = cell->getNodeField("vy");
            cell->getNodeField("vz0") = cell->getNodeField("vz");
            
            cell->getCellField("P0")    = cell->getCellField("P");
        }
        tissue->updateGhosts();
        
        if(conv)
            rec_str = max(rec_str, physicsIntegration->getRecalculateMatrixStructure());
        else if(rec_str)
        {
            physicsIntegration->recalculateMatrixStructure();
            physicsLinearSolver->recalculatePreconditioner();
            rec_str = false;
        }
        
        if(tissue->getMyPart()==0)
            cout << "Step " << step << ", time=" << time << ", deltat=" << deltat << endl;

        
        if(tissue->getMyPart()==0)
            cout << "Solving for velocities" << endl;
        
        physicsNewtonRaphson->solve();
        conv = physicsNewtonRaphson->getConvergence();
        
        if ( conv )
        {
            int nIter = physicsNewtonRaphson->getNumberOfIterations();
            
            conv = paramUpdate->UpdateParametrisation();

            if (conv)
            {
                if(tissue->getMyPart()==0)
                    cout << "Solved!"  << endl;
                //update polarity orientation and r12 based on the new cell center positions
                r12=getr12(tissue->getTissField("X1"),tissue->getTissField("Y1"),tissue->getTissField("Z1"),tissue->getTissField("A1"),tissue->getTissField("X2"),tissue->getTissField("Y2"),tissue->getTissField("Z2"),tissue->getTissField("A2"));
                getpol(r12,theta1,theta2,phi,p1,p2);
                if(tissue->getMyPart()==0)
                {
                    tissue->getTissField("R12X")=r12(0);
                    tissue->getTissField("R12Y")=r12(1);
                    tissue->getTissField("R12Z")=r12(2);
                }
                for(auto cell:tissue->getLocalCells())
                {
                    if(cell->getCellField("cellId")==0)
                    {
                        cell->getCellField("px")=p1(0);
                        cell->getCellField("py")=p1(1);
                        cell->getCellField("pz")=p1(2);
                    }
                    else
                    {
                        cell->getCellField("px")=p2(0);
                        cell->getCellField("py")=p2(1);
                        cell->getCellField("pz")=p2(2);
                    }
                }

                //put the active tension profile into the nodefields so that it is saved in the vtk files
                //also the fricion profile
                //normalize r12
                r12=r12/sqrt(r12*r12);
                for(auto cell:tissue->getLocalCells())
                {
                    tension_base   = cell->getCellField("tension");
                    tension_mod = cell->getCellField("dtension");
                    tension_mod2=cell->getCellField("dtension_spot");
                    tension_nem=cell->getCellField("dtension_nem");
                    b=cell->getCellField("narrowness");
                    if((int) cell->getCellField(0)== 0) //if cell 1 we access the center of mass of cell 1
                    {
                        x_com(0) = tissue->getTissField("X1")/tissue->getTissField("A1");
                        x_com(1) = tissue->getTissField("Y1")/tissue->getTissField("A1");
                        x_com(2) = tissue->getTissField("Z1")/tissue->getTissField("A1");
                    }
                    else
                    {
                        x_com(0) = tissue->getTissField("X2")/tissue->getTissField("A2");
                        x_com(1) = tissue->getTissField("Y2")/tissue->getTissField("A2");
                        x_com(2) = tissue->getTissField("Z2")/tissue->getTissField("A2");
                    }
                    //friction modulation
                    double f_base=cell->getCellField("frictionn");
                    double f_mod=cell->getCellField("dfriction");
                    //optogen
                    getpol(r12,(-90-10)*M_PI/180.0, 0,M_PI/2.0,p1_op,p2_op);
                    //loop on vertices
                    for(int n=0;n<cell->getNumberOfPoints();n++)
                    {
                        x(0)=cell->getNodeField("x")(n);
                        x(1)=cell->getNodeField("y")(n);
                        x(2)=cell->getNodeField("z")(n);
                        dx=x-x_com;
                        dx=dx/sqrt(dx*dx);
                        if((int) cell->getCellField(0)==0) //use correctpolarity
                        {
                            ct=dx*p1;
                        }
                        else
                        {
                            ct=dx*p2;
                        }
    
                        tension = tension_base + tension_mod*((b*exp(b*ct)-sinh(b))/(b*exp(b)-sinh(b)))+tension_nem*(ct*ct-1.0/3.0);
                        //little numerical safety in case b is small we just put the traditional linear profile
                        if(b<1e-3)
                        {
                            tension = tension_base + tension_mod*(ct)+tension_nem*(ct*ct-1.0/3.0);
                        }

                        //add additional spot if it is there, on cell 0 
                        if(cell->getCellField("activated_spot")>=0.5) //activated_spot =0.0 or 1.0 it's a double used as a boolean
                        {
                            ct=dx*p1_op;
                            if(bspot<1e-3)
                            {
                                tension += tension_mod2*(ct);
                            }
                            else
                            {
                                tension += tension_mod2*((bspot*exp(bspot*ct)-sinh(bspot))/(bspot*exp(bspot)-sinh(bspot)));
                            }
                        }
                        cell->getNodeField("tension")(n)=tension;
                        //friction
                        if(cell->getCellField("anis_friction_r12")==true)
                        {
                            if((int) cell->getCellField(0)==0)//cell 0 uses r12
                            {
                                ct=dx*r12;
                            }
                            else //cell 1 uses -r12
                            {
                                ct=-dx*r12;
                            }
                        }
                        frn=f_base+f_mod*ct;
                        cell->getNodeField("friction")(n)=frn;
                    }
                }
                
                time += deltat;
                tissue->getTissField("time") = time;
                //if time is after the laser ablation time, remove the tension gradient in cell 0 and cell 1
                if(time >= laser_ab_t && ablation_done==0)
                {
                    ablation_done=1;
                    if(tissue->getMyPart()==0)
                        cout << "Tension modification performed now." << endl;
                    for(auto cell:tissue->getLocalCells())
                    {
                        if(cell->getCellField("cellId")==0)
                        {
                            cell->getCellField("dtension")=dtension0_new;
                        }
                        else if(cell->getCellField("cellId")==1)
                        {
                            cell->getCellField("dtension")=dtension1_new;
                        }
                    }
                }
                //also create additional spot in cell 0 if it's time
                if(time >= additional_spot_t && new_spot_started==false)
                {
                    new_spot_started=true;
                    if(tissue->getMyPart()==0)
                        cout << "Additional spot created now." << endl;
                    for(auto cell:tissue->getLocalCells())
                    {
                        if(cell->getCellField("cellId")==0)
                        {
                            cell->getCellField("activated_spot")=1.0;
                        }
                    }
                }

                //deletes the spot after a given time
                if(time >= additional_spot_end_t && new_spot_started==true)
                {
                    if(tissue->getMyPart()==0)
                        cout << "Additional spot destroyed now." << endl;
                    for(auto cell:tissue->getLocalCells())
                    {
                        if(cell->getCellField("cellId")==0)
                        {
                            cell->getCellField("activated_spot")=0.0;
                        }
                    }
                }

                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("vx") /= deltat;
                    cell->getNodeField("vy") /= deltat;
                    cell->getNodeField("vz") /= deltat;
                }
                tissue->saveVTK("Cell","_t"+to_string(step+1));
                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("vx") *= deltat;
                    cell->getNodeField("vy") *= deltat;
                    cell->getNodeField("vz") *= deltat;
                }

                if(nIter < nr_maxite)
                {
                    deltat /= stepFacsuccess;
                    for(auto cell: tissue->getLocalCells())
                    {
                        cell->getNodeField("vx") /= stepFacsuccess;
                        cell->getNodeField("vy") /= stepFacsuccess;
                        cell->getNodeField("vz") /= stepFacsuccess;
                    }
                }
                if(deltat > maxDeltat)
                    deltat = maxDeltat;
                step++;


            }
            else
            {
                cout << "failed!" << endl;
                deltat *= stepFacfail;
                for(auto cell: tissue->getLocalCells())
                {
                    cell->getNodeField("x") = cell->getNodeField("x0");
                    cell->getNodeField("y") = cell->getNodeField("y0");
                    cell->getNodeField("z") = cell->getNodeField("z0");
                }
                tissue->updateGhosts();
            }
        }
        else
        {
            deltat *= stepFacfail;
            
            for(auto cell: tissue->getLocalCells())
            {
                cell->getNodeField("vx") = cell->getNodeField("vx0") * stepFacfail;
                cell->getNodeField("vy") = cell->getNodeField("vy0") * stepFacfail;
                cell->getNodeField("vz") = cell->getNodeField("vz0") * stepFacfail;
                cell->getCellField("P")  = cell->getCellField("P0");
            }
            tissue->updateGhosts();
        }        
        tissue->getTissField("deltat") = deltat;
        
        tissue->calculateCellCellAdjacency(3.0*intCL+intEL);
        tissue->updateGhosts();        
    }
        
    MPI_Finalize();

    return 0;
}
