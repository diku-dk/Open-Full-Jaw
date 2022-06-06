import OpenFullJaw
from OpenFullJaw.utils import *
import igl

def label_raw_mesh(V, T, path_tooth, path_pdl, path_bone, params):
    
    
    L = np.zeros(len(T)).astype(int)

    
    v_tooth, f_tooth = igl.read_triangle_mesh(path_tooth)
    v_bone,  f_bone  = igl.read_triangle_mesh(path_bone)
    v_pdl,   f_pdl   = igl.read_triangle_mesh(path_pdl)

    BC  = igl.barycenter(V,T)
    D_t = igl.signed_distance(BC, v_tooth, f_tooth, return_normals = False)[0]
    D_b = igl.signed_distance(BC, v_bone,  f_bone,  return_normals = False)[0]
    D_p = igl.signed_distance(BC, v_pdl,   f_pdl,   return_normals = False)[0]

    L[D_b <= 0] = 3
    L[D_t <= 0] = 1

    t1 = np.logical_and(D_t > 0, D_t < params.tet_filtering_dist)   # hollow of tets around the tooth with positive sdf values
    t2 = np.logical_and(D_b > 0, D_b < params.tet_filtering_dist)   # hollow of tets around the bone with positive sdf values
    t3 = np.logical_and(np.array(t1), np.array(t2))
    
    frame_t = mesh_sanity_check(v_tooth, f_tooth, d = 2, s = 0.6, name = 'tooth')
    frame_b = mesh_sanity_check(v_bone,  f_bone,  d = 2, s = 0.7, name = 'bone')
    frame_p = mesh_sanity_check(v_pdl,   f_pdl,   d = 2, s = 0.7, name = 'pdl')
    
    
    print('----> step0: hollow of teeth')
    frame = mp.plot(V, T[np.where(t1)], c = tooth_c, shading = sh_true, return_plot = True)
#     c = color_t
    print('----> step0: hollow of bone')
    frame = mp.plot(V, T[np.where(t2)], c = bone_c, shading = sh_true, return_plot = True)
    
    TB = T[np.where(t2)]
    idx = make_cut_plane_view(V, TB, 2, 0.7)
    print('----> step0: hollow of bone, plane cut ')
    frame = mp.plot(V, TB[idx], c = bone_c, shading = sh_true, return_plot = True)
#     frame.add_mesh(v_tooth, f_tooth)

    print('----> step1: intersection of hollow of teeth and the hollow of the bone')
    frame = mp.plot(V, T[np.where(t3)], c = mint_c, shading = sh_true, return_plot = True)
    frame.add_mesh(v_pdl, f_pdl, c = pdl_c, shading = sh_true)
 
    TB = T[np.where(t3)]
    idx2 = make_cut_plane_view(V, TB, 2, 0.7)
    print('----> step0: hollow of bone, plane cut ')
    frame = mp.plot(V, TB[idx2], c = pdl_c, shading = sh_true, return_plot = True)    
    
    
    t4 = igl.fast_winding_number_for_meshes(v_pdl, f_pdl, BC)
    t3 = np.logical_and(t3, t4 >= 0) 

    print('----> step2: positive winding numbers w.r.t. PDL rings')
    frame = mp.plot(V, T[np.where(t4 >= 0)], c = gray_c, shading = sh_true, return_plot = True)
    frame.add_mesh(v_pdl, f_pdl, c = pdl_c, shading = sh_true)
    
    print('----> step3: intersection of step1 and step2')
    frame = mp.plot(V, T[np.where(t3)], c = pdl_c, shading = sh_true, return_plot = True)
#     frame.add_mesh(v_pdl, f_pdl, c = color_p, shading = sh_true)
    
    print('----> final result for PDL geometries')
    frame = mp.plot(V, T[np.where(t3)], c = pdl_c, shading = sh_true, return_plot = True)

    L[t3] = 2
    # labels[D_p <= 0] = 3
    
    print('----> Pipeline input meshes:')

    print('Bone_temp, PDL_rims_wide, Teeth_temp')
    frame = mp.plot(v_bone, f_bone , c = bone_c, shading = sh_true, return_plot = True)
    frame.add_mesh (v_pdl + np.array([0,0,5]),  f_pdl, c = pdl_c, shading = sh_true)
    frame.add_mesh (v_tooth + np.array([0,0,10]), f_tooth, c = tooth_c, shading = sh_true)
    
    
    print('----> The labelled tetrahedra:')
    print('Bone, PDL, Teeth')
    frame = mp.plot(V, T[np.where(D_b <= 0)], c = bone_c, shading = sh_true, return_plot = True)    
    frame = mp.plot(V + np.array([0,0,5]), T[np.where(t3)], c = pdl_c, shading = sh_true, return_plot = True)
    frame = mp.plot(V + np.array([0,0,10]), T[np.where(D_t <= 0)], c = tooth_c, shading = sh_true, return_plot = True)

    T2 = T[L != 0]
    L_new = L[L != 0]
    V_new, T_new, _, _ = igl.remove_unreferenced(V, T2)

    return V_new, T_new, L_new



def label_impacted_pdl(V, T, L, dir_orig, dir_shrink, impacted):

    BC = igl.barycenter(V,T)

    for i in impacted:
        
        tooth_orig   = dir_orig   + "tooth_" + i + "_impacted.stl"
        tooth_shrink = dir_shrink + "tooth_" + i + "_temp.obj"
        
        v_orig,   f_orig    = igl.read_triangle_mesh(tooth_orig)
        v_shrink, f_shrink  = igl.read_triangle_mesh(tooth_shrink)

        D_o = igl.signed_distance(BC, v_orig,   f_orig,    return_normals = False)[0]
        D_s = igl.signed_distance(BC, v_shrink, f_shrink,  return_normals = False)[0]

        idx_o = np.logical_and(D_o > -0.8, D_o < 0)   # hollow of tets inside the original tooth
        idx_s = np.logical_and(D_s > 0,  D_s < 1)    # hollow of tets outside of the shrinked tooth
        idx = np.logical_and(np.array(idx_o), np.array(idx_s))
        
        TB = T[np.where(idx_o)]
        idx_cut = make_cut_plane_view(V, TB, 2, 0.7)
        print('----> step0: hollow of orig tooth with place cut')
        frame = mp.plot(V, TB[idx_cut], c = tooth_c, shading = sh_true, return_plot = True)


        TB = T[np.where(idx_s)]
        idx_cut = make_cut_plane_view(V, TB, 2, 0.7)
        print('----> step0: hollow of shrink tooth with plane cut')
        frame = mp.plot(V, TB[idx_cut], c = tooth_c, shading = sh_true, return_plot = True)

        TB = T[np.where(idx)]
        idx_cut = make_cut_plane_view(V, TB, 2, 0.7)
        print('----> step0: hollow of selected PDL')
        frame = mp.plot(V, TB[idx_cut], c = tooth_c, shading = sh_true, return_plot = True)

        L_new = L
        L_new[idx] = 2   # label those tets as 2 which corresponds to the pdl layer
        
        return idx, L