import time
import OpenFullJaw
from OpenFullJaw.utils import *

def preprocess_bone(path_bone, params):

    then = time.time()
    V_b, T_b, _ = generate_vol_mesh(params, path_bone)
    now = time.time()

    # extract surface mesh and remove any small components like holes, etc. 
    v_b, f_b = extract_tri_mesh_from_vol_mesh(V_b, T_b)
    v_b, f_b = keep_larget_component(v_b, f_b)
    mp.plot(v_b, f_b, bone_c, shading = sh_true)


    # save info for parameters' log and verbose mode

    b_preprocess_time = np.round(now - then, 2)
    print("bone remeshing time (s): ", b_preprocess_time)

    
    
    b_remeshing_info = {'epsilon'             : params.preprocess_bone_eps ,
                        'edge_length'         : params.preprocess_bone_l ,
                        'max_iterations'      : params.max_iter, 
                        'bone_remeshing_time' : b_preprocess_time
                        }
    
    params.params_log_info['bone_remeshing_parameters'] =  b_remeshing_info

    if params.verbose:
        igl.write_triangle_mesh( params.output_dir + params.output_fname + ".obj", v_b, f_b )
    
    return  v_b, f_b


def smooth_bone(v_b, f_b, params):
    
    then = time.time()
    vs, cs = laplace_beltrami_smoothing1(v_b, f_b, params.preprocess_smooth_bone_iter, params.preprocess_smooth_bone_stepsize)
    now = time.time()

    dist = np.apply_along_axis(np.linalg.norm, 1, vs[0] - vs[-1])

    ## plots
    print('---- original mesh superimposed on smoothed mesh')
    frame = mp.plot(vs[-1], f_b, np.log(dist) , shading = sh_true)
    frame.add_mesh(vs[0], f_b , bone_c, shading = sh_true)
    
    print('---- modified regions with highlighted points')
#     frame = mp.plot(vs[-1], f_b, np.log(dist), uv = 0.5, shading = sh_true)
#     frame.add_points(vs[-1][dist>max(dist)/4], shading = { "point_size": 2., "point_color": "red" })

    frame = mp.plot(vs[-1], shading = { "point_size": 1., "point_color": "gray" })
    frame.add_points(vs[-1][dist>max(dist)/4], shading = { "point_size": 2., "point_color": "red" })

    if params.verbose:
        igl.write_triangle_mesh( params.output_dir + params.output_fname + ".obj", vs[-1], f_b )



    b_smooth_time = np.round(now - then, 2)
    print("bone smoothing time (s): ", b_smooth_time)


    b_smoothing_info = {'smoothing_iteration': params.preprocess_smooth_bone_iter,
                        'smoothing_stp_size' : params.preprocess_smooth_bone_stepsize,
                        'min distance': min(dist),
                        'max distance': max(dist),
                        'bone_smoothing_time': b_smooth_time

                     }
    print(b_smoothing_info)

    params.params_log_info['bone_smoothing_parameters'] = b_smoothing_info  
    
    return  vs[-1], f_b


def remesh_teeth(path_all_teeth, params):

    then = time.time()
    V_ts, T_ts, _ = generate_vol_mesh(params, path_all_teeth)
    now = time.time()
    
    # extract the surface mesh 
    v_ts, f_ts = extract_tri_mesh_from_vol_mesh(V_ts, T_ts)
    
    igl.write_triangle_mesh( params.output_dir + "all_teeth_temp.obj", v_ts, f_ts )
        
    t_remesh_time = np.round(now - then, 2)
    print("teeth remeshing time (s): ", t_remesh_time)


    
    teeth_remeshing_info = {'epsilon'             : params.preprocess_teeth_eps ,
                            'edge_length'         : params.preprocess_teeth_l ,
                            'max_iterations'      : params.max_iter, 
                            'bone_remeshing_time' : t_remesh_time
                            }
   
    params.params_log_info['teeth_remeshing_parameters'] =  teeth_remeshing_info

    return v_ts, f_ts

