# CarGen - A direct geometry processing cartilage generation method

## About
CarGen (short for Cartilage Generation) is a python code library to generate subject-specific cartilages. Given bone geometry, our approach is agnostic to image modality, creates conforming interfaces, and is well suited for finite element analysis.

![teaser](https://user-images.githubusercontent.com/45920627/135057084-0b40ceb5-3145-46fa-b604-250802aef935.png)

## Installation
First, setup your conda environment for jupyter notebooks as below: 
```python
conda create -n cargen
conda activate cargen
conda config --add channels conda-forge
```
Next, install the following packages in the cargen enviroment:
```python
conda install igl
conda install meshplot 
conda install ipympl
conda install jupyter
```
Remember to activate your environment upon running the juyter notebook: 
```python
conda activate cargen
jupyter notebook
```
## Tutorials and examples
### Input/Output
This method is based only on bone surface mesh and distance measure. Bone geometry samples are in the [models](https://github.com/diku-dk/CarGen/tree/main/models) folder. The output is the surface mesh of the generated tiisue.
### Hip joint example
Below you can find the steps to generate femoral hip joint cartilage. We assume the Femur as the primary and the Pelvis as the secondary bone. The pipeline has four main steps. 

![bone-cross80%](https://user-images.githubusercontent.com/45920627/135130157-4e5dd3de-42d7-4b0d-9ec2-9071c75f6a54.png)

#### 1. Distance Filtering
In this step we select faces on the primary bone which will serve as an initial guess of the bone-cartilage interface. The initial selection is based on the distance between the face barycenters of the primary bone and the secondary bone vertices (yellow). To provide additional robustness to the initial guess, we trim the outer boundary by removing layers from its outer rim and discarding faces with two boundary edges (red). In the codes we refer this red surface as the *Initial surface*.

![distance-filtering35%](https://user-images.githubusercontent.com/45920627/135129869-290442cc-44b4-452e-aaf1-eac81df6b5b8.gif)

#### 2. Curvature-Based Region Filling
In these two steps we apply our curvature-based region filling approach to ensure the bone-cartilage interface fully covers the femoral head. In the codes we refer this red surface as the *Base surface*.

![region-filling35%](https://user-images.githubusercontent.com/45920627/135134968-c81eeb51-1c54-4d78-92f6-eb91f246a1cc.gif)

#### 3 & 4. Extrusion & Harmonic Boundary Blending
In these steps we assign a thickness profile to the Base surface by first extruding the vertices of the initial surface towards the secondary bone. The extrusion height is half the distance of each vertex towards the other side. Then, we interpolate between the boundary of the extruded and the Base surface to create a soft blend in the cartilage edges. In the codes we refer the orange mesh as *Cartilage surface*..
 
![femoral-cart35%](https://user-images.githubusercontent.com/45920627/135140574-61ade660-5f89-45ba-8d53-610ee3ddf438.png)

With a similar approach, we can generate the pelvic hip joint cartilage ensuring conforming cartilage-cartilage interface.

![endresult80%](https://user-images.githubusercontent.com/45920627/135150788-6004667d-0248-4ecb-84b5-83595d26c62e.png)

### Features
You can always change the free parameters of our algorithm to calibrate this code for your subject-specific application. The parameters and their default values are located in the [utils.py](https://github.com/diku-dk/CarGen/blob/main/cargen/utils.py) file.
#### Params related to the Distance Filtering: 
* **Gap_distance:** The maximum distance between the primary bone and the secondary bone.
* **Trimming_iteration:** Number of times the trimming step should be performed.
* **Smoothing_factor:** The size of the smoothing step (this is similar to the 'h' parameter in mean curvature flow)
* **Smoothing_iteration:** The number of times the smoothing step should be performed.
#### Params related to the Curvature-Based Region Filling: 
* **Curvature_type:** choose between "gaussian", "mean", "minimum", "maximum" curvature type. Refers to the gaussian, mean, the maximum and the minimum of the principal curvatures, respectively.
* **Neighbourhood_size:** How far away (in terms of edges) a vertex can be, while still considered a neighbour. Also referred to as 'k-ring' in some literature.
* **Min_curvature_threshold:** The minimum curvature where we will consider the surface to be part of the cartilage.
* **Max_curvature_threshold:** The maximum curvature where we will consider the surface to be part of the cartilage.
#### Params related to the Extrusion & Harmonic Boundary Blending:
* **Thickness_factor:** a constant which will be multiplied by the distance between two surfaces. This allows you to 
control the thickness value.
* **Blending_order:** Order of the harmonic weight computation during cartilage generation.
#### Input/ Output params
* **Input_dimension:** The scale of the input surface mesh ("mm" = millimeters, "m" = meters).
* **Output_dimension:** The scale of the output surface mesh ("mm" = millimeters, "m" = meters)

## Documentation 
Please refer to our published paper at [Computational Biomechanics for Medicine XVI](https://cbm.mech.uwa.edu.au/CBM2021/index.html)
## Citation
```python
soon to be added
```
## Acknowledgement 
❤️ We would like to thank the [Libigl](https://libigl.github.io) team, as the core of CarGen algorithm is based on the various functions available in the Libigl library.

❤️ This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No. 764644.
This repository only contains the [RAINBOW](https://rainbow.ku.dk) consortium’s views and the Research Executive Agency and the Commission are not responsible for any use that may be made of the information it contains.

![Webp net-resizeimage](https://user-images.githubusercontent.com/45920627/132510734-41c835fc-2502-4461-b3fd-770668d43c9d.jpg)

❤️ This project has also received funding from Independent Research Fund Denmark (DFF) under agreement No. 9131-00085B.

![dff_logo_uk_vertical_darkblue](https://user-images.githubusercontent.com/45920627/132513579-49a24905-bc53-43d5-a045-5370a7afc18a.png)
