## Segmentation and Geometry Reconstruction Steps 

### Table of contents
* [Segmentation Steps](#segmentation-steps)
* [Teeth-Bone Segmentation Results](#teeth-bone-segmentation-results)
* [Boundary Overlay of the Output Meshes](#Boundary-overlay-of-the-output-meshes)

### 1.Segmentation Steps

To reconstruct the patient-specific geometries, first, the scans are imported in [3DSlicer](https://www.slicer.org/) in the standard Digital Imaging and Communications in Medicine (DICOM) format. Next, according to the pre-evaluation criteria (metal fillings or implant artifacts), we decide on which jaws are suitable to be segmented from the scan.

### 1.1. Pre-processing step for segmentation:
The resolution of the selected CBCT scans is at most 0.3 mm. This is because, based on our experience, accurate teeth-bone segmentation of the scans with slice thicknesses above 0.3 mm is very challenging. Besides, applying smoothing functions with identical kernel sizes results in smoother segments for coarse voxel sizes, making it difficult to remove the segmentation noise (e.g., rugged surfaces) while preserving the fine details in desired regions (e.g., the alveolar crests of teeth sockets and root apexes). As a result, to avoid biases in the geometry reconstruction process, we convert all scans to an identical resolution of 0.15 mm by oversampling a cropped region of interest (ROI) containing the jaws using linear interpolation.

### 1.2. Teeth-bone segmentation using the watershed algorithm:
The resolution of the utilized scans in this study is not high enough to directly delineate the thin periodontal ligament's (PDL) structure (≈ 0.2 mm width) from the scans. Hence, we ignore the PDL structure in the segmentation step and only perform teeth-bone segmentation. Hence, the segmentation provides watertight teeth and bone segments in the contacting regions. The teeth-bone segmentation is performed based on a semi-automatic **watershed algorithm** which is provided in 3DSlicer's **SegmentEditorExtraEffects** extension. 

#### 1.2.1. Initial landmark definition
First, different segments with various colors are used to define the boundaries of the teeth and bone in different slices of the scan. Note that when using the watershed algorithm, one needs to specify a label, a background segment, to identify the uninterested regions, such as skin or background. 

![initial_landmarks](https://user-images.githubusercontent.com/30265621/177006776-3acc0947-11a3-4f7b-894b-259e361a74d1.png)



#### 1.2.2. Initialization
We initialize the algorithm based on the defined initial landmarks. Then, the algorithm grows different segments until different segments reach each other, and all voxels of the scan are labeled with either of the labels/segments. 

#### 1.2.3. Improving segmentation results 
The below image illustrates the result of the watershed algorithm. As it can be seen there are some leakages from teeth segments to the bone. Besides, the root regions are not delineated precisely.

The result of the watershed algorithm is then refined to fix the misclassified teeth and bone segments. Later, the geometries are smoothened using the 3DSlicer's standard **median** and **joint smoothing** modules. The joint smoothing method smooths the adjacent segments simultaneously and enforces watertight interfaces between them. 

The tooth-bone segmentation procedure proceeds until the segmentation accuracy meets our clinical validation criteria. A general criterion for the verification is precise segmentation of roots, crowns, and cervical bone regions, including the teeth sockets, and the miss-annotated regions indicated by the expert are revised until reaching the criterion. Finally, the segmented regions are exported as surface meshes in Stereolithography (STL) file format.

![watershed_result](https://user-images.githubusercontent.com/30265621/177007320-fefd9ec7-c790-404d-bf2a-33e2a603deb1.png)


### 2. Teeth-Bone Segmentation Results
The final and clinically-validated segmentation results are presented below. These segmented geometries are exported as STL files that can be found under each patient's **input** subfolder.

Note that the exported surfaces meshed in the **input** folders are unprocessed and dense meshes. Hence, one needs to reduce and re-mesh them before creating volumetric meshes to have a FE model of the human jaw with a reasonable number of elements. Otherwise, if they are used diectly for simulations, they can provide computationally expensive FE models. 


![P4](https://user-images.githubusercontent.com/30265621/177000615-1b222e3f-7c12-4bb3-826f-1c39a64188a4.png)

![P5](https://user-images.githubusercontent.com/30265621/177000624-0c680724-0a64-4015-ad0d-e914b31da250.png)

![P12](https://user-images.githubusercontent.com/30265621/177000629-f6b0b9ee-8179-48fc-a133-815086734a3e.png)

![P15](https://user-images.githubusercontent.com/30265621/177000638-e0069893-e004-43e0-bec2-6c616bdbae26.png)

![P16](https://user-images.githubusercontent.com/30265621/177000642-fd2d4ff0-2692-4a2b-a625-e183d266d36a.png)

### 3. Boundary Overlay of the Output Meshes 
We have imported the reduced conformal meshes, the **outputs** of our pipeline, into the 3D Slicer and super-imposed them on the patients' scan to demonstrate qualitative accuracy of the output meshes. The teeth, PDLs, and bone boundaries are presented in white, red, and beige colors.

The quantitative results on the accuracy of the reduced meshes were summarized in Table 7 of the manuscript.



![p6_lines_zoom](https://user-images.githubusercontent.com/30265621/177000810-2b61e63e-64ff-4b9d-9727-d639b3f0aba2.png)
