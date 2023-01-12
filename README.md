# Open-Full-Jaw Dataset Repository
> An open-access dataset and nearly-automated pipeline for generating finite element models of the human jaw ([ link to the article](https://www.sciencedirect.com/science/article/pii/S0169260722003911) )

### Table of contents
* [Motivation](#motivation)
* [Content of the repository](#content-of-the-repository)
* [How to cite us](#how-to-cite-us)
* [How to install](#how-to-install)
* [How to use](#how-to-use)
* [Features](#features)
* [Funding](#funding)
* [Acknowledgements](#acknowledgements)

## Motivation
Developing accurate patient-specific computational models of the human jaw acquired from cone-beam computed tomography (CBCT) scans is labor-intensive and non-trivial. In addition, it involves time-consuming unreproducible manual procedures. Hence, we share an open-access dataset of 17 patient-specific computational models of human jaws and the utilized pipeline for generating them.

## Content of the repository
This repository comprises **17 patient-specific computational models of human jaws (including the mandible, maxilla, their associated teeth, and PDL meshes, as well as teeth principal axes)** and the utilized **code** for generating them.


### 1. Dataset

![dataset_gallery_named](https://user-images.githubusercontent.com/30265621/172655083-a12f4842-aaa8-4a69-be95-2cfde6063008.png)

:warning: Please make sure to install [git lfs](https://git-lfs.github.com/) on your system before cloning the repository.

The [dataset](https://github.com/diku-dk/Open-Full-Jaw/tree/main/dataset) includes:
1. [Clinically validated segmented geometries](https://github.com/diku-dk/Open-Full-Jaw/tree/main/images) in form of unprocessed dense surface meshes that can be imported to different meshing tools
2. The quality and adaptive volumetric meshes to be used directly in FE simulations
3. The reduced surface meshes with no undesired gaps/penetrations
4. The principal axes of every patient's teeth that provides great information for setting up different scenarios automatically
5. The automatically generated FEM files for tipping scenario and a biting scenario
  

### 2. Code for reproducibility of dataset

6. The python code utilized for generating the volumetric meshes and producing the simulation files is shared in this repository. A summary of the pipeline is illustrated as follows

![](https://user-images.githubusercontent.com/30265621/168096619-a86fc97d-3887-49eb-a05e-590ac90d8822.png)
High resolution PDF format: [graphical abstract](./images/graphical_abstract_final.pdf)

## How to cite us
If you use the content of this repository please consider to cite us as below,

```bibtex
@article{gholamalizadeh2022open,
  title     = {Open-Full-Jaw: An open-access dataset and pipeline for finite element models of human jaw},
  author    = {Gholamalizadeh, Torkan and Moshfeghifar, Faezeh and Ferguson, Zachary and Schneider, Teseo and Panozzo, Daniele and Darkner, Sune and Makaremi, Masrour and Chan, Fran{\c{c}}ois and S{\o}ndergaard, Peter Lampel and Erleben, Kenny},
  journal   = {Computer Methods and Programs in Biomedicine},
  volume    = {224},
  pages     = {107009},
  year      = {2022},
  publisher = {Elsevier}
}

```


## How to install

#### Setting up the conda environment and the required packages

The first step is to set up your conda environment for running the jupyter notebook as below: 
```python
conda create -n fulljaw
conda activate fulljaw
conda config --add channels conda-forge
```
Next, install the following packages in the fulljaw environment:
```python
conda install igl
conda install jupyter
conda install meshio
conda install meshplot
conda install -c anaconda scikit-image
```
#### Other Dependencies
:bangbang: **You need to install [PyMesh](https://github.com/PyMesh/PyMesh) by cloning the repository and building it on your machine.**

:bangbang: **Note that our pipeline uses fTetWild as a backbone for volumetric mesh generation. Therefore, you need to install the [fTetWild](https://wildmeshing.github.io/ftetwild/) by cloning the repository and building it on your machine in a desired directory.**

## How to use
First, activate the environment, then run jupyter notebook: 
```python
conda activate fulljaw
jupyter notebook
```
Now you can open and run the [pipeline.ipynb](https://github.com/diku-dk/Open-Full-Jaw/blob/main/src/Pipeline.ipynb), which is an interactive python code, shared in the "src" folder, for reproducing the models of this dataset or generating new models for your own input data.

:warning: Make sure to update the "path_ftetwild" variable in [pipeline.ipynb](https://github.com/diku-dk/Open-Full-Jaw/blob/main/src/Pipeline.ipynb) with the path of fTetwild's binary file, e.g.,

path_ftetwild = '/your_desired_directory/fTetWild/build/FloatTetwild_bin'


## Features

#### Params of step1 (Preprocessing and Smoothing)
* ***preprocess_bone_eps:*** The epsilon value for fTetWild used in preprocessing of the bone mesh.
* ***preprocess_bone_l:*** The edge_length value for fTetWild used in preprocessing of the bone mesh.
* ***preprocess_teeth_eps:*** The epsilon value for fTetWild used in preprocessing of the teeth mesh.
* ***preprocess_teeth_l:*** The edge_length value for fTetWild used in preprocessing of the teeth mesh.
* ***preprocess_smooth_bone_iter:*** The number of iterations for smoothing the bone.
* ***preprocess_smooth_bone_stepsize:*** The step size used for smoothing the bone.

#### Params of step2 (Gap Generation)
* ***gap_thickness:*** The gap thickness which indicates the average width of the PDL layer.

#### Params for step3 (PDL Rim Generation)
* ***rim_distance_rest:*** The gap distance used in for selecting the base on all teeth sockets except for molars.
* ***rim_distance_molars:*** The gap distance used in for selecting the base on molars' sockets.
* ***rim_thickness_factor:*** The constant which will be multiplied by the distance between two surfaces.
* ***rim_trim_iter:*** The number of times the trimming step should be performed on the detected base.
* ***rim_smooth_iter_base:*** The number of times the smoothing step should be performed on the boundary of the detected base.
* ***rim_smooth_iter_extruded_base:*** The number of times the smoothing step should be performed on the boundary of the extruded base.

#### Params for step4 (Multi-Domain Volumetric Meshing)
* ***volume_mesh_eps:*** The epsilon value for fTetWild in multi-material meshing step.
* ***volume_mesh_l:*** The edge_length value for fTetWild in multi-material meshing step.
* ***volume_mesh_json_path:*** The directory where the json file for CSG operation will be stored.

#### Params for step5 (Tetrahedra Filtering) 
* ***tet_filtering_dist:*** The distance used for filtering the tetrahedra to get the intersection of teeth and bone hollows. 

#### Params for step6 (Simulation Setup)


#### Summery of params and their default values:
:warning:  Note that we have used identical values for most of the abovementioned parameters across all patients to prepare the Open-Full-Jaw dataset. Only a few of the parameters marked below (with  🚧 ) were used to adapt the pipeline's result to the patient-specific geometries.

```python
# params for step1( preprocessing and smoothing step)
preprocess_bone_eps  =  1e-4  # fTetWild's default value for the epsilon: 1e-3
preprocess_bone_l    =  0.02  # fTetWild's default value for the ideal_edge_length: 0.02
preprocess_teeth_eps =  1e-4
preprocess_teeth_l   =  0.01
preprocess_smooth_bone_iter  =  10
preprocess_smooth_bone_stepsize  =  0.0005

# params for step2( gap generation step)
gap_thickness        =  0.2
gap_tooth_bone_ratio =  0.5

# params for step3( PDL rim generation step)
rim_distance_rest    =  0.26   🚧 
rim_distance_molars  =  0.28   🚧 
rim_thickness_factor =  1
rim_trim_iter        =  2
rim_smooth_iter_base =  16   🚧
rim_smooth_iter_extruded_base  = 10   🚧

# params for step4( multi-domain volumetric mesh generation step)
volume_mesh_json_path       =  mid_o_dir + 'logical_operation.json'
volume_mesh_output_fname    =  'unified'
volume_mesh_operation_type  =  'union'
volume_mesh_eps             =  2e-4
volume_mesh_l               =  0.05       

# params for step5( tetrahedra filtering step) 
tet_filtering_dist  =  0.3   🚧

# general params for fTetWild
self.verbose_level  =  3
self.max_iter       =  20
self.ftet_path      =  path_ftetwild
self.output_dir     =  mid_o_dir 
```
## Funding
- This project has received funding from the European Union's Horizon 2020 research and innovation program under the Marie Sklodowska-Curie grant agreement No. 764644. This paper only contains the authors' views, and the Research Executive Agency and the Commission are not responsible for any use that may be made of the information it contains. 
- This work was also partially supported by the NSF CAREER award under Grant No. 1652515, the NSF grants OAC-1835712, OIA-1937043, CHS-1908767, CHS-1901091, NSERC DGECR-2021-00461 and RGPIN-2021-03707, a Sloan Fellowship, a gift from Adobe Research and a gift from Advanced Micro Devices, Inc.
- The funders had no role in study design, data collection/analysis, publication decision, or manuscript preparation.

## Acknowledgements
- We thank 3Shape A/S for providing this study's CBCT scans
- We especially thank 3Shape A/S Dental CAD AI team for their support in the CBCT segmentation and teeth principal axes computations. 
- We also thank NYU IT High-Performance Computing for resources, services, and staff expertise. 


