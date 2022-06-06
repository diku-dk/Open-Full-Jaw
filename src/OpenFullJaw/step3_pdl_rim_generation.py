import cargen
import OpenFullJaw
from OpenFullJaw.utils import *

def generate_pdls_rims(cargen_bone_input, mid_o_teeth_dir, unn, impacted, params):
    
    io_dim = 'mm'
    
    then = time.time()
    s1_vertices, s1_faces = cargen.read_and_clean(cargen_bone_input, io_dim)

    # stop the time
    now = time.time()
    print("bone cleanup time (s): ", np.round(now - then, 2))

    face_idxs, cumulative_sum = igl.vertex_triangle_adjacency ( s1_faces, len( s1_vertices ) )

    param_cargen = cargen.Var()

    # change the default params of cargen for our specific problem
    param_cargen.output_name = "pdl"
    param_cargen.gap_distance = 0.25  #0.3
    param_cargen.trimming_iteration = params.rim_trim_iter
    param_cargen.smoothing_iteration_base = params.rim_smooth_iter_base
    param_cargen.smoothing_iteration_extruded_base = params.rim_smooth_iter_extruded_base
    param_cargen.thickness_factor = params.rim_thickness_factor

    gap_distance = params.rim_distance_rest 
    gap_distance_molars = params.rim_distance_molars 
    t1 = time.time()


    molars = ['17', '18', '19', '30', '31', '32', '1', '2' , '3', '16', '15', '14']
    first = True
    for i in unn:
        print('generating PDL top for Tooth_{}'.format(i))
        if i in molars:
            param_cargen.gap_distance = gap_distance_molars
        else:
            param_cargen.gap_distance = gap_distance

        if i in impacted:
            continue

        s2_vertices, s2_faces = cargen.read_and_clean(params.input_dir +  params.input_fname.format(i), io_dim)

        margin = 0.5
        bl, ur = loose_bb(s2_vertices, margin)

        v_idx_orig  = np.where(np.all(np.logical_and(bl <= s1_vertices, s1_vertices <= ur), axis=1))[0]
        f_idx_orig  = []

        for j in v_idx_orig:
            for k in range ( cumulative_sum[ j ], cumulative_sum[ j+1 ] ) :
                f_idx_orig += [ face_idxs[ k ] ]

        p = mp.plot(s2_vertices, s2_faces, c = tooth_c, shading = sh_false, return_plot = True)
        p.add_mesh(s1_vertices, s1_faces[f_idx_orig], c = bone_c, shading = sh_true)

        f_idx_orig  = np.array(f_idx_orig)
        s1_v        = s1_vertices [v_idx_orig] # original vertex indices is idx and the corresponding new one is np.arrange(0, len(s1_vertices))
        s1_f_temp   = s1_faces[f_idx_orig]
        s1_f = np.copy(s1_f_temp)


        for j in np.arange(len(v_idx_orig)):
            indices = np.where(s1_f_temp == v_idx_orig[j])
            s1_f[indices] = j

        v_idx_new   = np.arange(0, len(v_idx_orig))    
        f_idx_new   = np.arange(0, len(f_idx_orig))

        mesh = pymesh.form_mesh(s1_vertices, s1_faces);
        s1_submesh = pymesh.submesh(mesh, f_idx_orig, num_rings = 0)

        p = mp.plot(s2_vertices, s2_faces, c = tooth_c, shading = sh_false, return_plot = True)
        p.add_mesh(s1_submesh.vertices, s1_submesh.faces, c = bone_c, shading = sh_true)


        # start the timer
        then = time.time()


        s1_v = np.copy(s1_submesh.vertices)
        s1_f = np.copy(s1_submesh.faces)

        # run
        _, _, top_vertices, top_faces = cargen.get_pdl_layer( s1_v, s1_f, s2_vertices, s2_faces, param_cargen)


        # stop the time
        now = time.time()

        print("PDL generation time (s): ", np.round(now - then, 2))


        pdl_top_path  = params.output_dir + params.output_fname.format(i)
        cargen.save_surface ( top_vertices, top_faces, io_dim, pdl_top_path )


    t2 = time.time()
    pdl_generation_time = np.round(t2 - t1, 2)
    print("PDL generation time (s): ", pdl_generation_time)

    cargen_info = {'gap_distance'        : gap_distance,
                   'gap_distance_molars' : gap_distance_molars,
                   'thickness_factor'    : param_cargen.thickness_factor, 
                   'trimming_iteration'  : param_cargen.trimming_iteration, 
                   'smoothing_iteration_base' : param_cargen.smoothing_iteration_base,
                   'smoothing_iteration_extruded_base' : param_cargen.smoothing_iteration_extruded_base,
                   'PDL_generation_time' : pdl_generation_time
                 }

    params.params_log_info['PDL_generation_parameters'] =  cargen_info
    
    return None