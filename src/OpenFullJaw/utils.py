
import numpy as np
import igl
import meshplot as mp
import matplotlib.pyplot as plt
import math
import pymesh
import json
import sys
import os
import meshio
import time
from scipy.sparse.linalg import spsolve  # for solving the smoothing matrix(in the laplace_beltrami_smoothing function)
from skimage import measure  # for marching cubes
import xml.etree.ElementTree as ET

    
    
# Sub-Functions

#----------------------------------------- 1. sub-functions for geometry processing 

# class mesh_info:
    
#     def __init__(self, v, f, r = 3):
#         self.v = v
#         self.f = f
#         self.r = r
#         self.vprime = []
#         self.kprime = []
#         self.pd1, self.pd2, self.pv1, self.pv2 = igl.principal_curvature(v, f, r, use_k_ring= True)
        
#     def compute_meshinfo(self, v, f, r = 3):
#         self.v = v
#         self.f = f
#         self.r = r
#         self.vprime = []
#         self.kprime = []
#         self.pd1, self.pd2, self.pv1, self.pv2 = igl.principal_curvature(v, f, r, use_k_ring= True)

        
        
def keep_larget_component(v, f):
    
    """
    this function splits the input mesh into connected components and returns the largest component
    
    inputs:
          v:  (3,N)  vertices
          f:  (3,M)  faces
     
    outputs:
          largest_comp.vertices:  (3,N_prime)  vertices of the largest component
          largest_comp.faces:     (3,M_prime)  faces of the largest component
          
    """
    
    input_mesh = pymesh.form_mesh(v, f)
    components_list = pymesh.separate_mesh(input_mesh, connectivity_type='auto')

    s = []
    for component in components_list:
        s.append(component.vertices.shape[0])
    idx = np.argmax(s) # gets the component with biggest vertices size
    # each component has two attributes:
    #     ori_vertex_index : indices of the component's vertices in the original mesh
    #     ori_elem_index   : indices of the component's elements in the original mesh                      
    orig_f_idx = components_list[idx].get_attribute("ori_elem_index").astype(int)    
    largest_comp = pymesh.submesh(input_mesh, orig_f_idx, num_rings = 0)  # this is to adjust the faces matrix based on the new vertices ids

    return largest_comp.vertices, largest_comp.faces



def loose_bb(v, margin): 
    
    """
    this function creates bounding box with a width of "margin" around the input point cloud
    
    inputs:
          v:   (3,N)    vertices
          m:   (float)  how much the bb should be loose
     
    outputs:
          bl:  (3,)  the bottom-left point of the bb
          ur:  (3,)  the upper-right point of the bb
          
    """
    bb_v, bb_f = igl.bounding_box(v)
    
    x_max, y_max, z_max = np.amax(bb_v, axis = 0)
    x_min, y_min, z_min = np.amin(bb_v, axis = 0)
    
    bl = np.array([[x_min - margin, y_min - margin, z_min - margin]])
    ur = np.array([[x_max + margin, y_max + margin, z_max + margin]])
    
    return bl, ur



def compute_mesh_grid_for_bb(bb0_v, margin, spacing):
    
    """
    this function computes a 3D grid with a margin width of "margin" and spacing of "spacing" over the input vertices
    
    inputs:
          bb0_v:    (3,N)     vertices of the bb
          margin:   (float)   how much the bb should be loose
          spacing:  (float)   the spacing or isotropic voxel size
     
    outputs:
          grid_v:   (3,x_dim * y_dim * z_dim)   vertices of the generated 3D grid
          bl:       (3,)    the bottom-left point of the bb
          ur:       (3,)    the upper-right point of the bb
          x_dim:    (int)   number of the generated points in the x dimension
          y_dim:    (int)   number of the generated points in the y dimension
          z_dim:    (int)   number of the generated points in the z dimension
          
    """

    x_max, y_max, z_max = np.amax(bb0_v, axis = 0)
    x_min, y_min, z_min = np.amin(bb0_v, axis = 0)
    
    x = np.arange(x_min - margin, x_max + margin, spacing)
    y = np.arange(y_min - margin, y_max + margin, spacing)
    z = np.arange(z_min - margin, z_max + margin, spacing)

    xx, yy, zz = np.meshgrid(x, y, z, sparse=False)

    X = xx.reshape(-1)
    Y = yy.reshape(-1)
    Z = zz.reshape(-1)

    grid_v = np.vstack((X, Y, Z)).T
    bl = np.array([[x_min, y_min, z_min]])
    ur = np.array([[x_max, y_max, z_max]])
    
    x_dim = xx.shape[0]
    y_dim = xx.shape[1]
    z_dim = xx.shape[2]
    
    return grid_v, bl, ur, x_dim, y_dim, z_dim



def offset_mesh_implicit(v, f, offset, margin, spacing_):
    
    """
    this function offsets the input mesh implicitly by creating a 3D grid with an isotropic voxel size of "spacing_", and the signed distance values
    
    inputs:
          v         (3,N)    vertices
          f         (3,M)    faces
          offset    (float)  the offseting value: { positive : dilation, negative : erosion}
          spacing_  (float)  voxel size or spacing
          margin    (float)  margin of bounding box to create a bigger bounding box around the mesh

    outputs:
          v_prim    (3,N_prime)  vertices of the offseted mesh
          f_prim    (3,M_prime)  faces of the offseted mesh
          
    """
  
    bb_v, bb_f = igl.bounding_box(v)
    bb_volume_v, bl, ur, x_dim, y_dim, z_dim = compute_mesh_grid_for_bb(bb_v, margin, spacing_)
    S, I, C  = igl.signed_distance(bb_volume_v, v, f)

    verts, faces, normals, values = measure.marching_cubes(S.reshape(x_dim, y_dim, z_dim), level = offset, spacing=(spacing_, spacing_, spacing_), gradient_direction = 'descent')

    # addjust the model on the input mesh as "ndimage" has put one of the vertice of the volume to the origin
    d = np.mean(bb_v, axis = 0) + (bl - ur)/2 - np.array([margin, margin, margin])
    v_prim = verts[:, (1, 0, 2)] + d   # adjust the representation differences in ndimage vs ours 
    f_prim = faces[:, (0, 2, 1)]       # adjust the representation differences in ndimage vs ours
    
    # visualize
    frame3 = mp.plot (v_prim, f_prim, shading = sh_true, return_plot = True)
    frame3.add_mesh( v + [10,0,0], f, shading = sh_true)

    
    return v_prim, f_prim



def normalize_(vec):
    
    """
    this function normalizes the length of the input vector
    
    """
    
    norm_vec = vec / np.linalg.norm(vec)
    return norm_vec



def flip_normals(f):
    
    """
    this function flips the normals of the faecs in the input mesh 
 
    """
    f = f[:, (0,2,1)]
    return f



def laplace_beltrami_smoothing1(v, f, num_iter = 30, step_size = 0.0008):
    
    """
    
    this function smoothes the input mesh using laplace_beltrami operation
    see the following link for further details: https://libigl.github.io/libigl-python-bindings/tut-chapter1/
    
    inputs:
          v:          (3,N)   vertices
          f:          (3,M)   faces
          num_iter:   (int)   number iterations for perfoming the smoothing opertation 
          step_size:  (int)   step size for smoothing operation 
    outputs:
          vs:         (list)  list of vertices at each iteration of the smoothing
          cs:         (list)  per-vertex normals
          
    """
    
    l = igl.cotmatrix(v, f)
    n = igl.per_vertex_normals(v, f) * 0.5 + 0.5
    c = np.linalg.norm(n, axis=1)

    vs = [v]
    cs = [c]
    for i in range(num_iter):
        m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_BARYCENTRIC)
        s = (m - step_size * l)
        b = m.dot(v)
        v = spsolve(s, m.dot(v))
        n = igl.per_vertex_normals(v, f) * 0.5 + 0.5
        c = np.linalg.norm(n, axis=1)
        if np.isnan(np.sum(v)):
            break
            print("----------- Error: there was an issue during the smoothing process of this mesh.")
        vs.append(v)
        cs.append(c)

    return vs, cs


def plot_smoothing_result(vs, cs, f):

    frame = mp.plot(vs[0], f, shading={"wireframe": True})
    # frame.add_mesh(vs[-1]+[3,0,0], f, shading={"wireframe": True})
    # frame = mp.plot(vs[0], f, np.abs(vs[0]-vs[-1]), shading={"wireframe": True})

    # compute difference between the original and the smooth version of the mesh
    dist = np.apply_along_axis(np.linalg.norm, 1, vs[0] - vs[-1])

    frame = mp.plot(vs[-1], f, np.log(dist) , shading = {"wireframe": True})

    sh_p = { "point_size": 1.5, "point_color": "red" }
    frame = mp.plot(vs[-1], f, np.log(dist) , shading = {"wireframe": True})
    frame.add_points(vs[-1][dist>0.1], shading = sh_p)
    
    print('minimum difference: ', min(dist))
    print('maximum difference: ', max(dist))
    
    return 



def compute_offset_limit(pv1, pv2):
    
    """
    
    This function computes the offset limit of a mesh, based on principal curvatures and radius curvature formula.
    
    inputs:
           pv1:         (N)  maximal curvature value per vertex
           pv2:         (N)  minimal curvature value per vertex
    outputs:
           offset_lim:  (float)  offset limit based on the curvature information of the mesh
           offset_mag:  (float)  the maximum value suggested for offseting the mesh
          
    """
    
    k = (pv1 + pv2) / 2               # mean curvature
    g = 1 / k                         # radius curvature  
    offset_lim = np.min(np.abs(g))    # surface offset limit

    return offset_lim



def split_teeth_to_separate_meshes(V_ts, F_ts, centers, unn, params):
    
    ## split the teeth and save each remesh tooth
    mesh   = pymesh.form_mesh(V_ts, F_ts)
    meshes = pymesh.separate_mesh(mesh, connectivity_type = 'auto')

    first_time = True
    for i in meshes:
        time.sleep(2)
        if first_time == True :
            frame = mp.plot(i.vertices, i.faces, tooth_c)
            first_time = False
        else:
            frame.add_mesh(i.vertices, i.faces, tooth_c)
        com = np.mean(i.vertices, axis = 0)
        d = centers - com
        dist = np.apply_along_axis(np.linalg.norm, 1, d)
        idx = np.argmin(dist)
        igl.write_triangle_mesh( params.output_dir  +  "/tooth_{}_temp.obj".format(unn[idx]), i.vertices, i.faces)

    return None    




#----------------------------------------- 2.sub-functions for PDL Rim generation

def clean(vertices, faces ) :
    
    epsilon                 = np.min (igl.edge_lengths ( vertices, faces ) ) / 100
    vertices, faces, fwrap  = igl.remove_duplicates( vertices, faces, epsilon )
    
    faces, f1               = igl.resolve_duplicated_faces ( faces ) 
    vertices, faces, IM, IJ = igl.remove_unreferenced  ( vertices, faces )
    
    return vertices, faces



def read_and_clean(path): 
    
    then = time.time()    
    vertices, faces = igl.read_triangle_mesh( path )
    print("number of triangles", len(faces))
    now = time.time()

    time_read = np.round(now - then, 2)    
    print('reading time:',np.round(time_read, 2))
    
    then = time.time()    
    vertices, faces = clean ( vertices, faces )
    now = time.time()

    time_clean = np.round(now - then, 2)    
    print('cleaning time:',np.round(time_clean, 2))
    
    return vertices, faces



def merge_surface_mesh ( vertices_1, faces_1, vertices_2, faces_2 ) :
    
    merged_vertices = np.concatenate ( ( vertices_1, vertices_2 ) )
    merged_faces    = np.concatenate( ( faces_1, faces_2 + len ( vertices_1 ) ) ) 
    
    return merged_vertices, merged_faces



def make_pdl_rim_wider(path_teeth, path_pdl, path_bone, params):
    
    v_t, f_t = igl.read_triangle_mesh(path_teeth)
    v_p, f_p = igl.read_triangle_mesh(path_pdl)
    v_b, f_b = igl.read_triangle_mesh(path_bone)

    sd_value, _, closest_points = igl.signed_distance(v_p, v_t, f_t, return_normals=False)
    idx_in = np.abs(sd_value) < 0.00001
    idx_out = ~idx_in

    frame = mp.plot(v_p, f_p, c = pdl_c, shading = sh_true)
    frame.add_points(closest_points[np.abs(sd_value) < 0.0001], shading = sh_p_x) 

    s, _, c = igl.signed_distance(v_p[idx_in], v_b, f_b, return_normals = False)
    v_p_new = np.copy(v_p)
    v_p_new[idx_in] = v_p[idx_in] - normalize_(c - v_p[idx_in])*3

    frame.add_points(c , shading = sh_p_z)
    frame.add_points(v_p_new[idx_in] , shading = sh_p_y)

    s2, _, c2 = igl.signed_distance(v_p[idx_out], v_t, f_t, return_normals = False)
    v_p_new[idx_out] = v_p[idx_out] - normalize_(c2 - v_p[idx_out])*3
    frame.add_points(v_p_new , shading= sh_p_y)

    print("Zoom-in on one of the PDL rims to see further details.")
    print('Red points   : the points that are in contact with the tooth.')
    print('Blue points  : the points that are in contact with the bone.')
    print('Green points : the points of the wider PDL rim to be used as the input of the multi-domain meshing process.')

    igl.write_triangle_mesh ( params.output_dir  + params.output_fname, v_p_new, f_p )

    return None


#----------------------------------------- 3. sub-functions for remeshing and volumetric mesh

class meshing_parameters:
    
    def __init__(self):
        self.epsilon         =  2e-4    #  ftetwild default value is 1e-3
        self.edge_length     =  0.05    #  ftetwild default value is 0.05
        self.verbose_level   =  3       #  verbose level for log (0 = most verbose, 6 = off)
        self.max_iterations  =  20
        self.json_path       =  mid_o_dir + 'logical_operation.json'
        self.output_fname    =  'unified'
        self.operation_type  =  'union'
        self.ftet_path       =  path_ftetwild
        self.output_dir      =  mid_o_dir 
        
    
    def reset(self):
        self.epsilon         =  3e-4    
        self.edge_length     =  0.05  
        self.verbose_level   =  3
        self.max_iterations  =  20
        self.json_path       =  mid_o_dir + 'logical_operation.json'
        self.output_fname    =  'unified'
        self.operation_type  =  'union'
        self.ftet_path       =  path_ftetwild
        self.output_dir      =  mid_o_dir
        
    def get_info_for_log(self):
        info = {'epsilon'         : self.epsilon ,
                'edge_length'     : self.edge_length,
                'max_iterations'  : self.max_iterations, 
                 }
        return info

    
    
def generate_json_file3(path_json, path_t, path_p, path_b, operation):
    
    """
    this function generates the required json file for performing CSG operation in fTetWild
    
    inputs:
           path_json:  (str)  path to the to be generated json file
           path_t:     (str)  path to the 1st surface mesh
           path_p:     (str)  path to the 2nd surface mesh
           path_b:     (str)  path to the 3rd surface mesh
           operation:  (str)  logical operation applied to the meshes


    outputs:
           does not return anything, will only save the requested json file in the path_json
          
    """

    dictionary = {'operation': operation,
                  'left': path_t,
                  'right': {'operation': operation,
                            'left': path_p,
                            'right': path_b
                           }
                 }

    jsonString = json.dumps(dictionary, indent=4)
    jsonFile = open(path_json  , "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print(jsonString)
    print(path_json)

    return None



def generate_vol_mesh(params, path_input):  
    
    """
    this function generates volumetric mesh for the input surface mesh based on the specified "params". This volumetric mesh is filtered based on the fTetWild's default filtering approach.      
    
    inputs:
           params:       -    an object of Params class that specifies mesh paramteres
           path_input:  (str)  path to the input surface mesh


    outputs:
           V:            (N,3)  vertices of the volumetric mesh
           T:            (M,4)  connectivity matrix of the elements in the 
           path_output:  (str)  path to the output volumetric mesh
          
    """
        
    ### Generate the volumetric mesh
    path_output = params.output_dir + params.output_fname +  '_with_eps_{}_l_{}'.format(params.preprocess_bone_eps, params.preprocess_bone_l)
    
    then = time.time()
    os.system(params.ftet_path + 
              ' -i ' + path_input +
              ' --level ' + str(params.verbose_level) + 
              ' --max-its ' + str(params.max_iter) + 
              ' -e ' + str(params.preprocess_bone_eps) + 
              ' -l ' + str(params.preprocess_bone_l) + 
              ' -o ' + path_output +
              ' --no-binary --no-color')
    now = time.time()
    print("Meshing time(s): ", np.round(now - then, 2))
    
    
    ### 3. Read the generated volumetric mesh
    V, T, _ = read_vol_mesh(path_output + '_.msh')
    
    return V, T, path_output 



def generate_json_file2(path_json, path_1, path_2, operation):
  
    """
    this function generates the required json file for performing CSG operation in fTetWild

    
    inputs:
           path_json:  (str)  path to the to be generated json file
           path_1:     (str)  path to the 1st surface mesh
           path_2:     (str)  path to the 2nd surface mesh
           operation:  (str)  logical operation to be applied on path_1 and path_2

    outputs:
           does not return anything, will only save the requested json file in the path_json
          
    """
    
    dictionary = {'operation': operation,
                  'left': path_1,
                  'right': path_2
                 }

    jsonString = json.dumps(dictionary, indent = 4)
    jsonFile = open(path_json, "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    
    
def extract_tri_mesh_from_vol_mesh(V, T):
    
    """
    this function extracts the surface mesh from the input volumetric mesh
    
    """
    
    f = igl.boundary_facets(T)
    F = flip_normals(f) # fix the normals' orientations
    
    return V, F



def read_vol_mesh(path):
    
    """
    inputs:
           path:  (str)  path to the .msh file

    outputs:
           V:     (N,3)  vertices
           T:     (M,4)  connectivity matrix of tet mesh
           L:     (M)    corresponding mesh labels that specifies different partitions or domains in the mesh
          
    """

    mesh = meshio.read(path, file_format = "gmsh" )
    V    = mesh.points
    T    = mesh.get_cells_type ("tetra")
    try:
        # Try to access label data and store in L
        L = mesh.cell_data["gmsh:geometrical"][0]
    except:
        # If label data inaccessible label everything as zero
        L = np.zeros(len(T))
    return V, T, L



def save_vol_mesh(V, T, L, path):
    
    cells = [('tetra', T)]
    mesh  = meshio.Mesh(V, cells)
    mesh.cell_data = {'gmsh:physical': [L], 'gmsh:geometrical': [L]}
    mesh.write(path, file_format = 'gmsh22', binary = False)


    
def make_cut_plane_view ( V, F, d = 0, s = 0.5 ) :
    
    # d is the direction of the cut: x=0, y=1, z=2
    # s it the size of the cut where 1 is equal to no cut and 0 removes the whole object.
    
    a = np.amin ( V, axis = 0 )
    b = np.amax ( V, axis = 0 )
    h = s * ( b - a ) + a
    c = igl.barycenter ( V, F )
    
    if d == 0 :
        idx = np.where ( c[ :,1 ] < h[ 1 ] )
    elif d == 1 :
        idx = np.where ( c[ :,0 ] < h[ 0 ] )
    else:
        idx = np.where ( c[ :,2 ] < h[ 2 ] )
    
    return idx


### interactive place cut

def make_cut_plane_view_interactive ( V, F, d = 0, s = 0.5 ) :
    
    # d is the direction of the cut: x=0, y=1, z=2
    # s it the size of the cut where 1 is equal to no cut and 0 removes the whole object.
    
    a = np.amin ( V, axis = 0 )
    b = np.amax ( V, axis = 0 )
    h = s * ( b - a ) + a
    c = igl.barycenter ( V, F )
    
    if d == 0 :
        idx = np.where ( c[ :,1 ] < h[ 1 ] )
    elif d == 1 :
        idx = np.where ( c[ :,0 ] < h[ 0 ] )
    else:
        idx = np.where ( c[ :,2 ] < h[ 2 ] )
        
    frame = mp.plot(V, F[idx], c = tooth_c, shading = shading_, return_plot = True)

    return idx,frame

    ##---> you would need the following libraries:
    # %matplotlib widget
    # import ipywidgets as widgets
    # import matplotlib.pyplot as plt
    # import numpy as np
    ##---> you should call the function as follow:
    # widgets.interact(make_cut_plane_view_interactive, V = fixed(V), F = fixed(T), d = {0, 2, 1}, s = (0, 1, .1))


def normal_visualization(v,f):
    
    arrow_length = 0.2
    bc = igl.barycenter(v, f)
    face_normals = igl.per_face_normals(v, f, np.array([1., 1., 1.]))
    end_points = bc + face_normals * arrow_length

    return bc, end_points


def mesh_sanity_check(v, f, d = 0, s = 0.5, name = 'input'):
    
    bc, end_points = normal_visualization(v, f)
    idx_cut = make_cut_plane_view(v, f, d, s)
    print('----> sanitiy check of ({}) mesh'.format(name))
    frame = mp.plot(v, f[idx_cut], shading = sh_true, return_plot = True)
    frame.add_lines(bc[idx_cut], end_points[idx_cut], shading={"line_color": "aqua"})
    
    return frame



#----------------------------------------- 4. FEB generation

def asvoid(arr):
    """
    Based on http://stackoverflow.com/a/16973510/190597 (Jaime, 2013-06)
    View the array as dtype np.void (bytes). The items along the last axis are
    viewed as one value. This allows comparisons to be performed which treat
    entire rows as one value.
    """
    arr = np.copy(np.ascontiguousarray(arr))
    if np.issubdtype(arr.dtype, np.floating):
        """ Care needs to be taken here since
        np.array([-0.]).view(np.void) != np.array([0.]).view(np.void)
        Adding 0. converts -0. to 0.
        """
        arr += 0.
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

## usage example
# X = np.array([[4,  2],
#               [9,  3],
#               [8,  5],
#               [3,  3],
#               [5,  6]])

# searched_values = np.array([[4, 2],
#                             [3, 3],
#                             [5, 6]])

# idx = np.flatnonzero(np.in1d(asvoid(X), asvoid(searched_values)))
# print(idx)



def convert_str_to_numpy(string):
    
    """
    input:
        string: a string that contains 3 floating numbers, example: '1.27, 3.32, 1.67'
    
    output:
        np_array: numpy array extracted from the input string

    """
    
    s = list(map(float, string.split(','))) # split the string based on the "," and convert each of them to float
    np_array = np.array(s).astype('float')
    return np_array



def get_orig_ids(a, b):

    """
    this function gets two arrays which one of them is a subset of the other one and returns the original indices of the elements in the subset
    
    inputs:
        a:  the original_set
        b:  the subset
        
    output:
        idx:  the original indices of the subset's elements in the original set
        
    """

    idx = np.flatnonzero(np.in1d(asvoid(a), asvoid(b)))
    return idx




def get_intersection_indices (a, b):
    
    """
    this function gets two set and returns the indices of their common elements
    inputs:
        a: the small set
        b: the original set
        
    output:
        idx:  the original indices of the subset's elements in the original set
        
    """    
    
    idx_list = []
    for i in b:
        idx_list.append(np.where((a == i).all(axis=1)))
    idx = np.array(idx_list).flatten()
    
    return idx



def xml_pretty_indent(elem, level=2):
    i = "\n" + level*'\t'
    if list(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + '\t'
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            xml_pretty_indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

            
            
class tooth:
    
    def __init__(self):
        self.x_axis              = np.array([[1,0,0]]) 
        self.y_axis              = np.array([[0,1,0]])    
        self.z_axis              = np.array([[0,0,1]])
        self.center              = np.array([[0,0,0]])
        self.load_point_idx      = 0


        
def  add_meterial(material_, mat_id, name_, type_, E_, v_, ro_):
    
    """
    inputs:
        mat_id :  material id
        name_  :  name of the material
        type_  :  type of the material model
        E_     :  Young's modulus of the material
        v_     :  Poisson ration of the material
        
    output:
        mat    : pointer to the insereted material in the xml file
        
    """
    
    mat = ET.SubElement(material_, "material", attrib={"id": "{}".format(mat_id), "name": "{}".format(name_), "type": "{}".format(type_)})
    mat.tail = '\n'
    mat_ro = ET.SubElement(mat, "density")
    mat_ro.text = "{}".format(ro_)
    mat_ro.tail = '\n'
    mat_E = ET.SubElement(mat, "E")
    mat_E.text = "{}".format(E_)
    mat_E.tail = '\n'
    mat_v = ET.SubElement(mat, "v")
    mat_v.text = "{}".format(v_)
    mat_v.tail = '\n'
    
    return mat



class simulation_parameters:
    
    def __init__(self):
        self.step_size             = 0.1   
        self.num_steps             = 10    
        self.verbose_level         = 'static'
        self.default_param         = True
        self.verbose_level         = 3
        
        #TODO: include all simulation parameters of FEBio and remove the template file
        
    def materials(self):
        
        self.tooth_E               = 2000
        self.tooth_v               = 0.3 
        self.tooth_ro              = 2.18e-9 
        self.tooth_mat_id          = 1
        self.tooth_mat_type        = 'neo-Hookean'
        
        self.pdl_E                 = 68.9
        self.pdl_v                 = 0.45 
        self.pdl_ro                = 1.142e-9
        self.pdl_mat_id            = 2
        self.pdl_mat_type          = 'neo-Hookean'
        
        self.bone_E                = 1500
        self.bone_v                = 0.3 
        self.bone_ro               = 1.178e-9
        self.bone_mat_id           = 3
        self.bone_mat_type         = 'neo-Hookean'
        
    def users_materials( E_t, v_t, ro_t, type_t, 
                         E_p, v_p, ro_p, type_p,
                         E_b, v_b, ro_b, type_b ):
        
        self.tooth_E               = E_t
        self.tooth_v               = v_t 
        self.tooth_ro              = ro_t  
        self.tooth_mat_type        = type_t
        
        self.pdl_E                 = E_p
        self.pdl_v                 = v_p
        self.pdl_ro                = ro_p
        self.pdl_mat_type          = type_p
        
        self.bone_E                = E_b
        self.bone_v                = v_b
        self.bone_ro               = ro_b
        self.bone_mat_type         = type_b


        
def get_triangle_area(vertices, faces):
    """
    this function measures the area of each triangle for the the given input surface
    :param vertices: list of vertices
    :param faces: list of the triangle faces in which you want to measure the surface area
    :return: an array containing area of each input triangle
    """
    triangles_area = []
    for i in range(len(faces)):
        u = vertices[faces[i, 1]] - vertices[faces[i, 0]]
        v = vertices[faces[i, 2]] - vertices[faces[i, 0]]
        n = np.cross(u, v)
        area = np.linalg.norm(n, axis=0) / 2
       
        triangles_area.append(area)

    triangles_area = np.array(triangles_area)
    return triangles_area 


#----------------------------------------- 4. old parameters


# class meshing_parameters:
    
#     def __init__(self):
#         self.epsilon         =  2e-4    #  ftetwild default value is 1e-3
#         self.edge_length     =  0.05    #  ftetwild default value is 0.05
#         self.verbose_level   =  3       #  verbose level for log (0 = most verbose, 6 = off)
#         self.max_iterations  =  20
#         self.json_path       =  mid_o_dir + 'logical_operation.json'
#         self.output_fname    =  'unified'
#         self.operation_type  =  'union'
#         self.ftet_path       =  path_ftetwild
#         self.output_dir      =  mid_o_dir 
        
    
#     def reset(self):
#         self.epsilon         =  3e-4    
#         self.edge_length     =  0.05  
#         self.verbose_level   =  3
#         self.max_iterations  =  20
#         self.json_path       =  mid_o_dir + 'logical_operation.json'
#         self.output_fname    =  'unified'
#         self.operation_type  =  'union'
#         self.ftet_path       =  path_ftetwild
#         self.output_dir      =  mid_o_dir
        
#     def get_info_for_log(self):
#         info = {'epsilon'         : self.epsilon ,
#                 'edge_length'     : self.edge_length,
#                 'max_iterations'  : self.max_iterations, 
#                  }
#         return info
    
    
#----------------------------------------- 5. new parameters
    

class Params:
    
    
    """
    Utilized parameters in the pipline, Note, for most of the parameters, a same value is used for all patients.
    
    preprocess_bone_eps: The epsilon value for fTetWild used in preprocessing of the bone mesh (fTetWild default value: 1e-3)
    preprocess_bone_l: The edge_length value for fTetWild used in preprocessing of the bone mesh (fTetWild default value: 0.02)
    preprocess_teeth_eps: The epsilon value for fTetWild used in preprocessing of the teeth mesh
    preprocess_teeth_l: The edge_length value for fTetWild used in preprocessing of the teeth mesh
    preprocess_smooth_bone_iter: The number of iterations for smoothing the bone
    preprocess_smooth_bone_stepsize: The step size used for smoothing the bone
    
    gap_thickness: The gap thickness which indicates the average width of the PDL layer.
    gap_bone_ratio: The ratio that specifies what portion of the gap's width should be deducted from bone geometry. 
    
    rim_distance_rest: The gap distance used in CarGen for all teeth except for molars.
    rim_distance_molars: The gap distance used in CarGen for molars.
    rim_thickness_factor: The thickness factor used in Cargen for creating the top wall of the PDL.
    rim_trim_iter: The number of times the trimming step should be performed on the detected base.
    rim_smooth_iter_base: The number of times the smoothing step should be performed on the boundary of the detected base.
    rim_smooth_iter_extruded_base: The number of times the smoothing step should be performed on the boundary of the extruded base.
    
    volume_mesh_eps: The epsilon value for fTetWild in multi-material meshing step
    volume_mesh_l: The edge_length value for fTetWild in multi-material meshing step
    volume_mesh_json_path: The path where the json file for CSG operation will be stored at
    volume_mesh_output_fname: 
    
    tet_filtering_dist: The distance used for filtering the tetrahedra to get the intersection of teeth and bone hollows. 


    verbose_level: The verbose level for log in fTetWild (0 = most verbose, 6 = off)
    max_iter: The maximum iteration for fTetWild
    output_fname: The name of the output file
    operation_type: The type of the boolean operation. In this repository we use only 'union' operation
    ftet_path: The path to the fTetWild's bin directory
    output_dir: The output directory where the files are going to be saved at 

    """
    def __init__(self):
        
        # the following parameters are fixed across all patients for preparing the Open-Full-Jaw dataset
        self.preprocess_bone_eps      =  1e-4
        self.preprocess_bone_l        =  0.02
        self.preprocess_teeth_eps     =  1e-4
        self.preprocess_teeth_l       =  0.01
        self.preprocess_smooth_bone_iter     =  10
        self.preprocess_smooth_bone_stepsize =  0.0005
        self.gap_thickness   =  0.2
        self.gap_bone_ratio  =  0.5
        self.verbose_level   =  3
        self.max_iter        =  20
        self.volume_mesh_json_path  =  os.getcwd() + '/' + 'mid-output'  + '/'  + 'logical_operation.json'
        self.output_fname    =  'output_mesh'
        self.input_fname     =  'input_mesh'
        self.operation_type  =  'union'
        self.ftet_path       =  '~/fTetWild/build/FloatTetwild_bin'
        self.input_dir       =  os.getcwd() + '/' + 'input'  + '/' 
        self.output_dir      =  os.getcwd() + '/' + 'mid-output'  + '/'  
        self.volume_mesh_eps =  2e-4
        self.volume_mesh_l   =  0.05
        self.verbose         =  True
        self.params_log_info =  {}
        
        # the following parameters can be adjusted based on patient-specific geometries 
        self.rim_distance_rest     =  0.26
        self.rim_distance_molars   =  0.28
        self.rim_thickness_factor  =  1
        self.rim_trim_iter         =  2
        self.rim_smooth_iter_base  =  16
        self.rim_smooth_iter_extruded_base  = 10
        self.tet_filtering_dist    = 0.3

    
    def reset(self):
        
        # params for step1( preprocessing and smoothing step)
        self.preprocess_bone_eps      =  1e-4
        self.preprocess_bone_l        =  0.02
        self.preprocess_teeth_eps     =  1e-4
        self.preprocess_teeth_l       =  0.01
        self.preprocess_smooth_bone_iter      =  10
        self.preprocess_smooth_bone_stepsize  =  0.0005
        
        # params for step2( gap generation step)
        self.gap_thickness   =  0.2
        self.gap_bone_ratio  =  0.5

        # params for step3( PDL rim generation step)
        self.rim_distance_rest        =  0.26
        self.rim_distance_molars      =  0.28
        self.rim_thickness_factor     =  1
        self.rim_trim_iter            =  2
        self.rim_smooth_iter_base     =  16
        self.rim_smooth_iter_extruded_base  = 10
        
        # params for step4( multi-domain volumetric mesh generation step)
        self.volume_mesh_json_path       =  os.getcwd() + '/' + 'mid-output'  + '/' + 'logical_operation.json'
        self.volume_mesh_operation_type  =  'union'
        self.volume_mesh_eps             =  2e-4
        self.volume_mesh_l               =  0.05       
       
        # params for step5( tetrahedra filtering step) 
        self.tet_filtering_dist  = 0.3
        
        # general params for fTetWild
        self.verbose_level =  3
        self.max_iter      =  20
        self.ftet_path     =  '~/fTetWild/build/FloatTetwild_bin'
        self.input_dir     =  os.getcwd() + '/' + 'input'  + '/' 
        self.input_fname   =  'input_mesh'
        self.output_dir    =  os.getcwd() + '/' + 'mid-output'  + '/'  
        self.output_fname  =  'output_mesh'
        self.verbose =  True
        self.params_log_info = {}

    
#colors    
bone_c  = np.array( [240, 220, 170] ) / 255
tooth_c = np.array( [250, 250, 225] ) / 255 
pdl_c   = np.array( [235, 137, 108] ) / 255
mint_c  = np.array( [205, 228, 212] ) / 255
gray_c  = np.array( [211, 211, 211] ) / 255
green   = np.array( [128, 174, 128] ) / 255
blue    = np.array( [111, 184, 210] ) / 255


# mesh plotting properties
sh_true  = {'wireframe': True, 'flat':True, 'side': 'FrontSide', 'reflectivity': 0.1, 'metalness': 0}
sh_false = {'wireframe': False,'flat':True, 'side': 'FrontSide', 'reflectivity': 0.1, 'metalness': 0}

# point plotting properties
sh_p = { "point_size": 0.5, "point_color": "red" }
sh_p_c = { "point_size": 5, "point_color": "gray" }
sh_p_x = { "point_size": 1, "point_color": "red" }
sh_p_y = { "point_size": 0.55, "point_color": "green" }
sh_p_z = { "point_size": 1, "point_color": "blue" }

# color_t = np.array([255/255,250/255,250/255])
# color_p = np.array([166/255,84/255,94/255])
# color_b = np.array([202/256,231/256,193/256])