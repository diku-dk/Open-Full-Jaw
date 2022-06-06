import cargen
import numpy as np
import igl
import meshplot as mp
import math


def get_pdl_layer(vertices_p,
                  faces_p,
                  vertices_s,
                  faces_s,
                  param):
    """
    This function generates gap-based cartilages based on the bone geometries

    :param vertices_p: list of vertex positions of the primary surface
    :param faces_p: list of triangle indices of the primary surface
    :param vertices_s: list of vertex positions of the secondary surface
    :param faces_s: list of triangle indices of the secondary surface
    :param param: list of parameters needed to generate hip joint cartilage models from "all_params.py"

    :return: vertices and faces of hip joint articulating cartilages
    """

    " Step A. distance filtering "

    # save primary bone vertices separately
    vertices_bp = np.copy(vertices_p)
    vertices_bs = np.copy(vertices_s)

    # bone adjacency faces
    face_adjacency, cumulative_sum = igl.vertex_triangle_adjacency(faces_p, len(vertices_p))

    # initial cartilage surface definition
    intt_face_idxs = cargen.get_initial_surface(vertices_p,
                                                faces_p,
                                                vertices_s,
                                                faces_s,
                                                param.gap_distance)

    # trim the cartilage boundary layer
    int_face_idxs = cargen.trim_boundary(faces_p,
                                         intt_face_idxs,
                                         face_adjacency,
                                         cumulative_sum,
                                         param.trimming_iteration)

    # removing extra components
    int_face_idxs = cargen.get_largest_component(faces_p,
                                                 int_face_idxs)

    # remove ears
    for i in range(3):
        int_face_idxs = cargen.remove_ears(faces_p,
                                           int_face_idxs)

    base_face_idxs = np.copy(int_face_idxs)

    # viz
    frame = mp.plot(vertices_p, faces_p, c=cargen.mandible, shading=cargen.sh_false, return_plot = True)
    print(frame)
    frame.add_mesh(vertices_p, faces_p[base_face_idxs], c=cargen.blue, shading=cargen.sh_true)

    " Step A-2: smoothing + quality control for the base layer "

    # neighbour info
    base_b_vertex_idxs, boundary_face_idxs, neigh_face_list = cargen.neighbouring_info(vertices_p,
                                                                                       faces_p[base_face_idxs])
    # measure dihedral angle
    max_angles = cargen.get_dihedral_angle(vertices_p, faces_p, base_face_idxs, neigh_face_list)
    max_angle = np.max(max_angles)
    max_angle = np.round(max_angle, 2)

    print("Base layer: max dihedral angle before smoothing is ", max_angle,
          "radians (", np.round(math.degrees(max_angle), 2), "degrees).")

    # norm visualization
    centroids, end_points = cargen.norm_visualization(vertices_p, faces_p[base_face_idxs])
    frame = mp.plot(vertices_p, faces_p[base_face_idxs], c=cargen.pastel_yellow, shading=cargen.sh_true, return_plot = True)
    frame.add_lines(centroids, end_points, shading={"line_color": "aqua"})

    # apply the smoothing + remove penetration/gap
    iteration = []
    for ss in range(param.smoothing_iteration_base):

        vertices_p = cargen.smooth_boundary(vertices_p, base_b_vertex_idxs, param.smoothing_factor)
        vertices_p = cargen.snap_to_surface(vertices_p, vertices_bp, faces_p)
        smoothed_max_angles = cargen.get_dihedral_angle(vertices_p, faces_p, base_face_idxs, neigh_face_list)

        folded_vertex_idxs = []
        # np.float64(2) # max_angle
        for count, i in enumerate(smoothed_max_angles):
            if i > np.float64(2):
                folded_vertex_idxs.append(count)
                iteration.append(ss - 1)

    smoothed_max_angle = np.max(smoothed_max_angles)
    smoothed_max_angle = np.round(smoothed_max_angle, 2)

    print("Base layer: max dihedral angle after smoothing is ", smoothed_max_angle,
          "radians (", np.round(math.degrees(smoothed_max_angle), 2), "degrees).")

    # norm visualization
    centroids, end_points = cargen.norm_visualization(vertices_p, faces_p[base_face_idxs])
    frame = mp.plot(vertices_p, faces_p[base_face_idxs], c=cargen.pastel_yellow, shading=cargen.sh_true, return_plot = True)
    frame.add_lines(centroids, end_points, shading={"line_color": "aqua"})

    print("Quality control results for the base layer: ")
    print("")
    if len(folded_vertex_idxs) != 0:
        print("- There are issues with these vertex indices", folded_vertex_idxs)
        print("- Solution 1: To prevent this issue, decrease your 'param.smoothing_iteration' to ", iteration[0])
        print("- Solution 2: By default the faulty triangles will be removed from the face index list ")
        print("")

        # vz
        print("faulty vertices & neighbouring triangles:")
        frame = mp.plot(vertices_p, faces_p[base_face_idxs], c=cargen.blue, shading=cargen.sh_false, return_plot = True)
        frame.add_points(vertices_p[base_b_vertex_idxs[folded_vertex_idxs]],
                         shading={"point_size": 0.2, "point_color": "red"})

        faces = faces_p[base_face_idxs]
        for i in folded_vertex_idxs:
            frame.add_mesh(vertices_p, faces[np.array(neigh_face_list[i])], c=cargen.sweet_pink,
                           shading=cargen.sh_true)
            frame.add_mesh(vertices_p, faces[np.array(boundary_face_idxs[i])], c=cargen.pastel_yellow,
                           shading=cargen.sh_true)

        if param.fix_boundary:
            base_face_idxs = cargen.fix_boundary(vertices_p, faces_p, base_face_idxs, base_b_vertex_idxs,
                                                 folded_vertex_idxs)
            base_b_vertex_idxs = igl.boundary_loop(faces_p[base_face_idxs])
            print("normal visualization of the fixed result:")
            centroids, end_points = cargen.norm_visualization(vertices_p, faces_p[base_face_idxs])
            frame = mp.plot(vertices_p, faces_p[base_face_idxs], c=cargen.pastel_yellow, shading=cargen.sh_true, return_plot = True)
            frame.add_lines(centroids, end_points, shading={"line_color": "aqua"})

    else:
        print("Everything is clean in the base layer. we will now continue to extrusion step:")
        print("")

    # cartilage area
    cartilage_area = cargen.get_area(vertices_p,
                                     faces_p[base_face_idxs])
    " Part B. Extrusion "
    thickness_profile = cargen.assign_thickness(vertices_p,
                                                faces_p,
                                                vertices_bs,
                                                faces_s,
                                                base_face_idxs,
                                                param.thickness_factor)

    base_vertex_idxs = np.unique(faces_p[base_face_idxs].flatten())
    weights = thickness_profile[base_vertex_idxs]

    # extrude surface
    ex_vertices_p = cargen.extrude_cartilage(vertices_p,
                                             faces_p,
                                             base_face_idxs,
                                             weights)

    # snap to the opposite surface
    ex_vertices_p = cargen.snap_to_surface(ex_vertices_p, vertices_bs, faces_s)

    ex_base_face_idxs = np.copy(base_face_idxs)

    # neighbour info
    ex_base_b_vertex_idxs, boundary_face_idxs, neigh_face_list = cargen.neighbouring_info(ex_vertices_p,
                                                                                          faces_p[ex_base_face_idxs])
    # measure dihedral angle
    max_angles = cargen.get_dihedral_angle(ex_vertices_p, faces_p, ex_base_face_idxs, neigh_face_list)
    max_angle = np.max(max_angles)
    max_angle = np.round(max_angle, 2)

    print("Extruded layer: max dihedral angle before smoothing is ", max_angle,
          "radians (", np.round(math.degrees(max_angle), 2), "degrees).")

    # norm visualization
    centroids, end_points = cargen.norm_visualization(ex_vertices_p, faces_p[ex_base_face_idxs])
    frame = mp.plot(ex_vertices_p, faces_p[base_face_idxs], c=cargen.pastel_yellow, shading=cargen.sh_true, return_plot = True)
    frame.add_lines(centroids, end_points, shading={"line_color": "aqua"})

    if param.smoothing_iteration_extruded_base != 0:
        # apply the smoothing + remove penetration/gap
        iteration = []
        for ss in range(param.smoothing_iteration_extruded_base):
            ex_vertices_p = cargen.smooth_boundary(ex_vertices_p, ex_base_b_vertex_idxs, param.smoothing_factor)
            ex_vertices_p = cargen.snap_to_surface(ex_vertices_p, vertices_bs, faces_s)
            smoothed_max_angles = cargen.get_dihedral_angle(ex_vertices_p, faces_p, ex_base_face_idxs, neigh_face_list)

            folded_vertex_idxs = []
            # np.float64(2) # max_angle
            for count, i in enumerate(smoothed_max_angles):
                if i > np.float64(2):
                    folded_vertex_idxs.append(count)
                    iteration.append(ss - 1)

        smoothed_max_angle = np.max(smoothed_max_angles)
        smoothed_max_angle = np.round(smoothed_max_angle, 2)

        print("Extruded layer: max dihedral angle after smoothing is", smoothed_max_angle,
              "radians (", np.round(math.degrees(smoothed_max_angle), 2), "degrees).")
    else:
        smoothed_max_angles = cargen.get_dihedral_angle(ex_vertices_p, faces_p, ex_base_face_idxs, neigh_face_list)

        folded_vertex_idxs = []
        # np.float64(2) # max_angle
        for count, i in enumerate(smoothed_max_angles):
            if i > np.float64(2):
                folded_vertex_idxs.append(count)
                iteration.append(ss - 1)

        smoothed_max_angle = np.max(smoothed_max_angles)
        smoothed_max_angle = np.round(smoothed_max_angle, 2)

        print("Extruded layer: max dihedral angle after smoothing is", smoothed_max_angle,
              "radians (", np.round(math.degrees(smoothed_max_angle), 2), "degrees).")

    # norm visualization
    centroids, end_points = cargen.norm_visualization(ex_vertices_p, faces_p[ex_base_face_idxs])
    frame = mp.plot(ex_vertices_p, faces_p[ex_base_face_idxs], c=cargen.pastel_yellow, shading=cargen.sh_true, return_plot = True)
    frame.add_lines(centroids, end_points, shading={"line_color": "aqua"})

    print("Quality control results  for the extruded layer: ")
    print("")
    if len(folded_vertex_idxs) != 0:
        print("- There are issues with these vertex indices", folded_vertex_idxs)
        print("- Solution 1: To prevent this issue, decrease your 'param.smoothing_iteration' to ", iteration[0])
        print("- Solution 2: By default the faulty triangles will be removed from the face index list ")

        # vz
        print("faulty vertices & neighbouring triangles:")
        frame = mp.plot(ex_vertices_p, faces_p[ex_base_face_idxs], c=cargen.pastel_blue, shading=cargen.sh_false, return_plot = True)
        frame.add_points(ex_vertices_p[ex_base_b_vertex_idxs[folded_vertex_idxs]],
                         shading={"point_size": 0.2, "point_color": "red"})

        faces = faces_p[ex_base_face_idxs]
        for i in folded_vertex_idxs:
            frame.add_mesh(ex_vertices_p, faces[np.array(neigh_face_list[i])], c=cargen.sweet_pink,
                           shading=cargen.sh_true)
            frame.add_mesh(ex_vertices_p, faces[np.array(boundary_face_idxs[i])], c=cargen.pastel_yellow,
                           shading=cargen.sh_true)

        if param.fix_boundary:
            ex_base_face_idxs = cargen.fix_boundary(ex_vertices_p, faces_p, ex_base_face_idxs, ex_base_b_vertex_idxs,
                                                    folded_vertex_idxs)
            ex_base_b_vertex_idxs = igl.boundary_loop(faces_p[ex_base_face_idxs])
            print("normal visualization of the fixed result:")
            centroids, end_points = cargen.norm_visualization(ex_vertices_p, faces_p[ex_base_face_idxs])
            frame = mp.plot(ex_vertices_p, faces_p[ex_base_face_idxs], c=cargen.pastel_yellow, shading=cargen.sh_true, return_plot = True)
            frame.add_lines(centroids, end_points, shading={"line_color": "aqua"})

    else:
        print("Everything is clean in the extruded base layer. We will now continue to create the roof.")
        print("")

    # viz
    frame = mp.plot(vertices_p, faces_p, c=cargen.mandible, shading=cargen.sh_false, return_plot = True)
    frame.add_mesh(vertices_p, faces_p[base_face_idxs], c=cargen.blue, shading=cargen.sh_true)
    frame.add_mesh(ex_vertices_p, faces_p[ex_base_face_idxs], c=cargen.green, shading=cargen.sh_true)

    " Part C. Build the closed surface"

    # flip normals of the bottom surface
    base_faces = faces_p[base_face_idxs]
    base_faces[:, [0, 1]] = base_faces[:, [1, 0]]

    # build wall
    roof_faces = cargen.get_wall(vertices_p,
                                 base_b_vertex_idxs,
                                 ex_base_b_vertex_idxs)

    # merge left, top, and right faces
    si_vertices = np.concatenate((ex_vertices_p, vertices_p))
    si_faces = np.concatenate((faces_p[ex_base_face_idxs], roof_faces, base_faces + len(ex_vertices_p)))

    # increase the number of facet rows in the wall
    usi_vertices, usi_faces = igl.upsample(si_vertices, si_faces, param.upsampling_iteration)
    roof_vertices, roof_faces = igl.upsample(usi_vertices, roof_faces, param.upsampling_iteration)

    # final touch
    usi_vertices, usi_faces = cargen.clean(usi_vertices,
                                           usi_faces)

    roof_vertices, roof_faces = cargen.clean(roof_vertices,
                                             roof_faces)

    # viz
    frame = mp.plot(ex_vertices_p, faces_p[base_face_idxs], c=cargen.green, shading=cargen.sh_true, return_plot = True)
    frame.add_mesh(vertices_p, base_faces, c=cargen.blue, shading=cargen.sh_true)
    frame.add_mesh(roof_vertices, roof_faces, c=cargen.organ, shading=cargen.sh_true)
    frame.add_points(vertices_p[base_b_vertex_idxs], shading={"point_size": 0.3, "point_color": "red"})
    frame.add_points(ex_vertices_p[ex_base_b_vertex_idxs], shading={"point_size": 0.3, "point_color": "red"})

    f = mp.plot(roof_vertices, roof_faces, c=cargen.organ, shading=cargen.sh_true, return_plot = True)
    f.add_mesh(vertices_s, faces_s, c=cargen.tooth, shading=cargen.sh_false)

    frame = mp.plot(vertices_p, faces_p, c=cargen.mandible, shading=cargen.sh_false, return_plot = True)
    frame.add_mesh(usi_vertices, usi_faces, c=cargen.organ, shading=cargen.sh_true)

    # geometry
    print("pdl base area is: ", np.round(cartilage_area, 2))
    print("mean pdl thickness is: ", np.round(np.mean(thickness_profile[base_vertex_idxs]), 2))
    print("maximum pdl thickness is: ", np.round(np.max(thickness_profile[base_vertex_idxs]), 2))

    return usi_vertices, usi_faces, roof_vertices, roof_faces
