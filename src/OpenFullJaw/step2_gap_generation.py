import time
import OpenFullJaw
from OpenFullJaw.utils import *
import igl
import os
import glob



def shrink_bone(v_b_new, f_b, params):
    
    pd1, pd2, pv1, pv2 = igl.principal_curvature(v_b_new, f_b, 5, use_k_ring= True)
    offset_lim_b = compute_offset_limit(pv1, pv2)
    print('---- the offset limit of the bone is: ', offset_lim_b)

    offset_bone = np.floor(offset_lim_b * 70)/100  # to avoid paper thin walls in the bone
    print('---- will assume its offset limit is {} to avoid paper thin walls in the bone mesh. ', offset_bone)

    space = params.gap_thickness
    bone_ratio = params.gap_bone_ratio
    if offset_bone < space * bone_ratio:
        comment = 'bone offset limit({}) is less than desired gap ({}).'.format(offset_bone, space * bone_ratio)
        offset  = offset_bone
        bone_ratio = offset_bone / space
    else:
        comment = 'offset limit is fine.'
        offset  = space * bone_ratio

    print(comment)
    print('is shrinking the bone with {} mm.'.format(offset))

    s1_normals = igl.per_vertex_normals(v_b_new, f_b, 2)  # 2 corresponds to igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE angle weighted
    v_b_shrinked = v_b_new - offset * s1_normals

    frame = mp.plot ( v_b_shrinked, f_b, c = pdl_c, shading = sh_true, return_plot = True)
    igl.write_triangle_mesh ( params.output_dir + params.output_fname + ".obj", v_b_shrinked, f_b )   

    rest = space - offset

    shrinking_info = {'desired_gap'        : params.gap_thickness,
                      'bone_ratio_for_gap' : params.gap_bone_ratio, 
                      'bone_offset_limit'  : offset_bone,
                      'bone_offset'        : offset,
                      'comment'            : comment
                     }
    
    return v_b_shrinked, f_b, offset, shrinking_info




def shrink_teeth(teeth_dir, rest, shrinking_info, params):
    
    space = params.gap_thickness
    voxel_size = 0.1
    
    shrinking_info['teeth_offset'] = rest
    shrinking_info['voxel_size'] = voxel_size
    print('is shrinking the teeth with {} mm.'.format(rest))
    first_time1 = True
    first_time2 = True

    margin = 0.3
    centers = []
    unn = [] # unn IDs
    impacted = [] # IDs of impacted teeth

    then = time.time()
    os.chdir(teeth_dir)

    #----------------------------------------------------- Read all teeth and shrink them
    for file in sorted(glob.glob('*.stl')):
        file, _ = file.split('.')   # file name : 'tooth_31.stl' , we want unn ID which is '31'
        tooth_info = file.split('_')
        tnum = tooth_info[1]  
        unn.append(tnum)
        print('---------------------------------------------', file)

        if tooth_info[-1] == 'impacted':
            impacted.append(tnum)

        v, f = read_and_clean(file + '.stl')

        if tooth_info[-1] == 'impacted':
            impacted.append(tnum)
            print('-----> Impacted tooth! Will shrink it with {} mm instead.'.format(space*0.75))
            v_prime, f_prime = offset_mesh_implicit(v, f, -space*0.75, margin, voxel_size)
        else:
            v_prime, f_prime = offset_mesh_implicit(v, f, -rest, margin, voxel_size)


        centers.append(np.mean(v_prime, axis = 0))


        if first_time1:
            all_v = v
            all_f = f
            first_time1 = False      
        else: 
            all_v, all_f = merge_surface_mesh(all_v, all_f, v, f)


        if first_time2:
            teeth_v = v_prime
            teeth_f = f_prime
            first_time2 = False      
        else: 
            teeth_v, teeth_f = merge_surface_mesh(teeth_v, teeth_f, v_prime, f_prime)

        # save the shrinked tooth
        igl.write_triangle_mesh(params.output_dir + "tooth_{}_shrinked.obj".format(tnum), v_prime, f_prime )
        print('num_vertices of original tooth : ', v.shape[0])
        print('num_vertices of shrinked tooth : ', v_prime.shape[0])
        print('-----------------------------------------------')


    if params.verbose:
        igl.write_triangle_mesh(teeth_dir + "all_teeth.obj", all_v, all_f)
        igl.write_triangle_mesh(params.output_dir + params.output_fname + ".obj", teeth_v, teeth_f )

    frame = mp.plot (all_v,     all_f, c = tooth_c, shading = sh_false, return_plot = True)  # The mesh with all teeth
    frame = mp.plot (teeth_v, teeth_f, c = tooth_c, shading = sh_false, return_plot = True)  # The mesh with all eroded teeth


    now = time.time()

    shrink_time = np.round(now - then, 2)
    print("Time for shrinking all teeth with voxel of {} (s): ".format(voxel_size), shrink_time)

    shrinking_info['shrinking_time'] = shrink_time
    params.params_log_info['shrinking_parameters'] = shrinking_info  

    unn = np.array(unn)
    centers = np.array(centers)    
    impacted = np.array(impacted)

    mid_o_dir = os.path.abspath(os.path.join(params.output_dir, os.pardir))
    np.save(mid_o_dir + 'unn.npy', unn)
    np.save(mid_o_dir + 'centers.npy', centers)
    if impacted.size != 0:
        np.save(mid_o_dir + 'impacted_teeth.npy', impacted)
        
    return teeth_v, teeth_f, unn, centers, impacted
