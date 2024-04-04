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
30,1,0,0
39.96,10,1,1
33.85,20,0,1
```
Each folder must also contain two files 'doublet_X_fiducial_marker_cellY_xy.csv' and 'doublet_X_hand_analysis.csv' which track two fiducial markers and the doublet center of mass like in "Figure5_laser_ablation/data/".

### Supp_Fig1_single_cells/data_single_cells/
One folder 'Segmentation_X' per doublet. Each folder contains a tiff file 'X.tif' of 'X_blur.tif' with the myosin signal (in 3D and in time) of the single cell. Each folder also contains a folder 'Cell_1' with the segmented meshes 'timeX_cell_1.ply'
 of the cell.
