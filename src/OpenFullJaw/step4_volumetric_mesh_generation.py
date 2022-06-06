import time
import OpenFullJaw
from OpenFullJaw.utils import *

def generate_raw_mesh(path_1, path_2, path_3, params):
    
    """
    this function applies the CSG operation specified in "params" on the input meshes and exports the "raw mesh", which is the unfiltered volumetric mesh covering all input meshes.       
    
    inputs:
           params:    -      an object of meshing_parameters class that specifies mesh paramteres
           path_1:  (str)    path to the 1st surface mesh
           path_2:  (str)    path to the 2nd surface mesh
           path_3:  (str)    path to the 3rd surface mesh

    outputs:
           V :      (N, 3)   vertices of the generated volume mesh including the background mesh   
           T :      (M, 4)   connectivity matrix of the generated volume mesh including the background mesh
           
    """
    t1 = time.time()

    ### 1. Create corresponding .json file for gluing the meshes
    generate_json_file3(params.volume_mesh_json_path, path_1, path_2, path_3, params.operation_type)

        
    ### 2. Generate the volumetric mesh
    path_output = params.output_dir +  params.output_fname +  '_with_eps_{}_l_{}'.format(params.volume_mesh_eps, params.volume_mesh_l)

    print('output_path : ', path_output)

    os.system(params.ftet_path + 
              ' --csg ' + params.volume_mesh_json_path +
              ' --level ' + str(params.verbose_level) + 
              ' --max-its ' + str(params.max_iter) + 
              ' -e ' + str(params.volume_mesh_eps) + 
              ' -l ' + str(params.volume_mesh_l) + 
              ' -o ' + path_output+
              ' --no-binary --no-color'+
              ' --export-raw')

    ### 3. Read the generated raw mesh
    V, T, _ = read_vol_mesh(path_output + '__all.msh')
    
    t2 = time.time()
    glue_time = np.round(t2 - t1, 2)
    print("gluing time (s): ", glue_time)

    gluing_info = {'epsilon'         : params.volume_mesh_eps,
                   'edge_length'     : params.volume_mesh_l,
                   'max_iterations'  : params.max_iter, 
                   'gluing_time'     : glue_time
                  }
   
    
    return V, T, gluing_info
