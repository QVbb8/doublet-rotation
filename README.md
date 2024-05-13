# doublet-rotation
This repository provides all the python and c++ codes used in the following article: 'Polarity-driven three-dimensional spontaneous rotation of a cell doublet', doi:https://doi.org/10.1038/s41567-024-02460-w.

The code can be used to perform the analysis of raw data (3d microscopy images and segmented meshes), and generate the plots shown in the article.

## Input data
Raw data should be introduced in the following folders:

### 2min_time_resolution_analysis/data_2min/
One folder per doublet, containing: tiff files of the rotating doublet (blurred and/or not blurred), two folders named 'Cell_1', 'Cell_2' that contains timeX_cell_Y.ply files containing the segmented meshes of individual cells at the given time.

### Figure1_Figure2/data_rotation/
One folder 'doublet_X' per doublet, containing : A folder 'Cell_0' containing the meshes 'timeX_cell_0.ply' of the mother cell before division, and two folders 'Cell_1', 'Cell_2' containing meshes 'timeX_cell_Y.ply' of the two daughter cells after division.

### Figure3_Figure4/myosin_data/
One folder per doublet, containing : a tiff file of the microscopy image (in 3d and in time) of the myosin intensity, two folders 'Cell_1', 'Cell_2' with the segmented meshes of each cell.

### Figure5_blebbistatin/blebbistatin_data/
One folder per doublet, containing : Folders 'Cell_1' and 'Cell_2' containing the meshes of the two cells.

### Figure5_laser_ablation/data/
One folder 'doublet_X' per doublet. Each folder must contain two files 'doublet_X_fiducial_marker_cellY_xy.csv' (with Y=1 or 2) of the following format:

```
,X,Y,Ch,Frame
1,19.722,18.250,2,1
2,20.337,16.484,2,2
etc.
```
The relevant information is contained in the X,Y columns tracking the positions of fiducial markers rotating with the doublet. The folder must also contain one file 'doublet_X_hand_analysis.csv' of the following format:

```
,Area,X,Y,Major,Minor,Angle
1,306.785,13.941,14.217,20.300,19.242,179.851
2,321.401,13.460,13.982,20.917,19.564,22.945
etc.
```
This is the information that can be generated from FIJI, where an ellipse is fitted to the doublet at each time point. Only the (x,y) positions of the doublet center of mass (center of the ellipse) are being used.

### Figure5_optogenetic/data/
One folder 'doublet_X' per doublet. Each folder must contain a file 'doublet_X_time.csv' of the following format:
```
33.86,20,0,1
30   ,1 ,0,0
39.96,10,1,1
33.85,20,0,1
```
This format is linked to the specific procedure used for recording images, in the case of the optogenetic experiments. The first column shows the time interval between frames, the second columns shows how many frames have this time interval, the third column is 0 when there is no optogenetic activation during these frames, or 1 if there is optogenetic activation. The last column is related to waiting times introduced when switching from a recording mode to a non-recording mode. Ultimately, this file is read by the function 'extract_time_info' in the script 'plots_Fig5gh.py', to construct an array of time points for the different frames, where the time zero is defined when optogenetic activation starts. Alternatively, one can simply provide such a time array, instead of using 'doublet_X_time.csv' files.

Each folder must also contain two files 'doublet_X_fiducial_marker_cellY_xy.csv' and 'doublet_X_hand_analysis.csv' which track two fiducial markers and the doublet center of mass like in "Figure5_laser_ablation/data/".

### Supp_Fig1_single_cells/data_single_cells/
One folder 'Segmentation_X' per doublet. Each folder contains a tiff file 'X.tif' of 'X_blur.tif' with the myosin signal (in 3D and in time) of the single cell. Each folder also contains a folder 'Cell_1' with the segmented meshes 'timeX_cell_1.ply' of the cell.

## How to run the codes

### Python Part
The python scripts must be run in a particular order, since some of the analysis is sequential. Here is a way to do it:
#### Figure1_Figure2/
1. prepare_data_for_analysis.py :  It loads the meshes of the cells (after cell division, .ply files in Cell_1, Cell_2 folders) and extract information in the shape of numpy arrays that are then stored as .npy files.
2. plots_Fig1_Fig2.py, plot_Fig2e.py, Ext_Fig_4c.py : These scripts generate the plots of Figure 1, Figure 2, Extended Figure 3, Extended Figure 4c.
#### Figure3_Figure4/
1. calibration_pz.py : This computes the correction to apply to the myosin signal. It also generates the plots of Extended Figure 5.d,e.
2. plots_Fig3_Fig4.py : This script then generates the plots of Figure 3,4, and Extended Figure 7.
3. plot_interface_maps_ecad_paper_ext_fig.py : This generates the plot of Extended Figure 4b.
#### Figure 5
1. Figure5_blebbistatin/plots_Fig5bc.py, Figure5_laser_ablation/plots_Fig5e.py, Figure5_optogenetic/plots_Fig5gh.py : In no particular order, these scripts generate the plots of Figure 5b,c,e,g,h.
#### Supp_Fig1_single_cells/
1. calibration_z_2min_single_cell.py : First one must compute the correction to apply to the myosin signal.
2. Supp_Fi1cd.py : This script then generates the plots of Supplementary Figure 1c,d.

### C++ part, interacting active surfaces
The simulation introduced in the paper is based on the 'interacting active surfaces' (IAS) framework that can be found here: https://github.com/torressancheza/ias. One must first compile the IAS code separately, which requires the following libraries: OpenMPI, OpenMP, Trilinos and VTK. Then, one must compile the C++ code of this specific project (in interacting_active_surfaces/) which imports the IAS framework using cmake. In 'aux.h' and 'aux.cpp', functions are defined that compute all the forces internal to the membranes (including the active tension modulation) as well as the adhesion forces between cells. In 'main.cpp', the simulation parameters are introduced (they can also be provided in a text file read by the program), the two cells are initialised as spheres (or loaded directly from previous simulation results) and the simulation is run until a given time is reached.
