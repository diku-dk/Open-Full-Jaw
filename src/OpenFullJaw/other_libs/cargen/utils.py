import numpy as np
import igl
import meshplot as mp
import matplotlib.pyplot as plt
import math

def clean(vertices,
          faces):
    """
    this function removes duplicated faces and removes unreferenced vertices from the input mesh

    :param vertices: list of vertex positions
    :param faces: list of triangle indices

    :return: cleaned vertices and faces
    """
    # epsilon:  minimal distance to consider two vertices identical
    epsilon = np.min(igl.edge_lengths(vertices, faces)) / 100
    vertices, faces, _ = igl.remove_duplicates(vertices, faces, epsilon)
    faces, _ = igl.resolve_duplicated_faces(faces)
    vertices, faces, _, _ = igl.remove_unreferenced(vertices, faces)

    return vertices, faces


def read_and_clean(path,
                   input_dimension):

    """
    this function reads vertex and face information from an input surface mesh

    :param path: a path where the surface meshes are stored in
    :param input_dimension: the dimension of the input mesh ("mm" = millimeters, "m" = meters)

    :return: the vertices and faces corresponding to the input mesh
    """

    vertices, faces = igl.read_triangle_mesh(path, 'float')

    if input_dimension == "m":
        vertices = vertices * 1000

    vertices, faces = clean(vertices, faces)

    print("number of triangles after cleaning", len(faces))

    return vertices, faces


def get_curvature_measures(vertices,
                           faces,
                           neighbourhood_size,
                           curvature_type,
                           curve_info):

    """
    computes the curvature of a surface mesh and sets a corresponding curvature field for it

    :param vertices: list of vertex positions
    :param faces: list of triangle indices
    :param neighbourhood_size: controls the size of the neighbourhood used
    :param curvature_type: choose among "gaussian", "mean", "minimum" or "maximum"
    :param curve_info: If set to True, will visualize the curvature type on your model
    :return: assigned curvature value for for each face

    """

    # max_pd_v : #v by 3 maximal curvature direction for each vertex
    # min_pd_v : #v by 3 minimal curvature direction for each vertex
    # max_pv_v : #v by 1 maximal curvature value for each vertex
    # min_pv_v : #v by 1 minimal curvature value for each vertex
    max_pd_v, min_pd_v, max_pv_v, min_pv_v = igl.principal_curvature(vertices, faces, neighbourhood_size)

    # curvature onto face
    min_pv_f = igl.average_onto_faces(faces, min_pv_v)
    max_pv_f = igl.average_onto_faces(faces, max_pv_v)
    mean_pv_f = (min_pv_f + max_pv_f) / 2.0
    gaussian_pv_f = min_pv_f * max_pv_f

    # dictionary enabling selection of curvature measure
    selector = {"gaussian": gaussian_pv_f, "mean": mean_pv_f, "minimum": min_pv_f, "maximum": max_pv_f}

    # assign the selected curvature type
    curvature_value = selector[curvature_type]

    if curve_info:
        print(np.min(curvature_value))
        print(np.max(curvature_value))
        print(np.mean(curvature_value))

        frame = mp.subplot(vertices, faces, min_pv_f, s=[2, 2, 0])
        mp.subplot(vertices, faces, max_pv_f, s=[2, 2, 1], data=frame)
        mp.subplot(vertices, faces, mean_pv_f, s=[2, 2, 2], data=frame)
        mp.subplot(vertices, faces, gaussian_pv_f, s=[2, 2, 3], data=frame)

    return curvature_value


def get_initial_surface(vertices_p,
                        faces_p,
                        vertices_s,
                        faces_s,
                        gap_distance):

    """
    this function selects the initial subset of the primary surface

    :param vertices_p: list of vertex positions of the primary surface
    :param faces_p: list of triangle indices of the primary surface
    :param vertices_s: list of vertex positions of the secondary surface
    :param faces_s: list of triangle indices of the secondary surface
    :param gap_distance: distance threshold to select a subset of the primary surface as tissue attachment region
    :return: list of facet indices corresponding to the initial subset of the primary surface
    """
    # barycenter coordinate of each face
    triangle_centroids = igl.barycenter(vertices_p, faces_p)

    # sd_value: list of smallest signed distances
    # sd_face_idxs: list of facet indices corresponding to smallest distances
    # closest_points: closest points on the secondary surface to each point in triangle_centroids
    sd_value, sd_face_idxs, closest_points = igl.signed_distance(triangle_centroids,
                                                                 vertices_s, faces_s,
                                                                 return_normals=False)

    # list of facet indices below a distance threshold
    initial_face_idxs = np.where(sd_value < gap_distance)[0]

    # histogram
    plt.hist(sd_value, bins="doane")
    plt.xlabel("distance from the primary to the secondary bone")
    plt.ylabel("number of facets")
    plt.show()

    return initial_face_idxs


def get_boundary_faces(faces,
                       sub_face_idxs,
                       face_adjacency, cumulative_sum):
    """
    determines facet indices belonging to the boundary of a sub-region .

    :param faces: list of faces where the sub-region is a part of
    :param sub_face_idxs: list of facet indices corresponding to the sub-region you want to find the boundary
    :param face_adjacency: The face adjacency matrix
    :param cumulative_sum: cumulative sum of "face-vertex visiting procedure" from libigl

    :return: Boundary face indices on both sides of the boundary and the inner boundary face indices
    """

    edge_vertex_idxs = igl.boundary_facets(faces[sub_face_idxs])
    boundary_vertex_idxs = np.unique(edge_vertex_idxs.flatten())

    boundary_face_idxs = []
    for j in boundary_vertex_idxs:
        for k in range(cumulative_sum[j], cumulative_sum[j + 1]):
            boundary_face_idxs += [face_adjacency[k]]
    inner_boundary_face_idxs = np.intersect1d(boundary_face_idxs, sub_face_idxs)

    return boundary_face_idxs, inner_boundary_face_idxs


def trim_boundary(faces,
                  sub_face_idxs,
                  face_adjacency,
                  cumulative_sum,
                  trimming_iteration):
    """
    Trim the boundary of a sub-region

    :param faces: list of faces where the sub-region is a part of
    :param sub_face_idxs: list of facet indices corresponding to the region you want to trim
    :param face_adjacency: the face adjacency matrix
    :param cumulative_sum: cumulative sum of "face-vertex visiting procedure" from libigl
    :param trimming_iteration: Number of trimming iterations to perform

    :return: list of facet indices corresponding to the trimmed sub-region
    """
    for i in range(trimming_iteration):
        b_face_idxs, ib_face_idxs = get_boundary_faces(faces, sub_face_idxs, face_adjacency, cumulative_sum)
        sub_face_idxs = np.setxor1d(sub_face_idxs, ib_face_idxs)

    return sub_face_idxs


def get_largest_component(faces,
                          sub_face_idxs):
    """
    only keeps the largest component of a sub-region

    :param faces: list of faces where the sub-region is a part of
    :param sub_face_idxs: list of facet indices corresponding to the sub-region

    :return: list of facet indices corresponding to the largest component of the sub-region
    """

    components = igl.face_components(faces[sub_face_idxs])
    components_count = np.bincount(components)
    z = np.argmax(components_count)
    max_count = np.where(components == z)[0]
    single_face_idxs = sub_face_idxs[max_count]

    return single_face_idxs


def remove_ears(faces,
                sub_face_idxs):
    """
    Removes any "ears" of the the sub-region.
    An "ear" refer to a single triangle with a vastly different normal than adjacent triangles
    See the illustration below for an example:

            ____/\____

    :param faces: list of faces where the sub-region is a part of
    :param sub_face_idxs: list of facet indices corresponding to the sub-region

    :return: list of facet indices corresponding to the sub-region with all the rears removed
    """

    ears = igl.ears(faces[sub_face_idxs])[0]
    cleaned_faces = np.delete(faces[sub_face_idxs], ears, axis=0)
    cleaned_face_idxs = []
    # TODO: Remove this for loop.
    for i in range(len(cleaned_faces)):
        ind = np.where(faces == cleaned_faces[i])[0]
        m = np.unique(ind, return_counts=True)[1]
        o = np.where(m == 3)[0][0]
        cleaned_face_idxs.append(ind[o])

    return np.array(cleaned_face_idxs)


def grow_cartilage(faces,
                   sub_face_idxs,
                   face_adjacency, cumulative_sum,
                   curvature_value,
                   min_curvature_threshold,
                   max_curvature_threshold):

    """
    grows the sub-region surface based on global curvature values
    TODO: Change this to be based on local curvature

    :param faces: list of faces where the sub-region is a part of
    :param sub_face_idxs: list of facet indices corresponding to the sub-region
    :param face_adjacency: the face adjacency matrix
    :param cumulative_sum: cumulative sum of "face-vertex visiting procedure" from libigl
    :param curvature_value: assigned curvature value for for each face
    :param min_curvature_threshold: minimum curvature threshold for the grown region
    :param max_curvature_threshold: maximum curvature threshold for the grown region

    :return: list of facet indices corresponding to the grown sub-region
    """

    pending_face_idxs = sub_face_idxs
    faces_count = sub_face_idxs.shape[0]

    for n in range(200):

        # b = both side boundary, ib = inner boundary
        grow_b_face_idxs, grow_ib_face_idxs = get_boundary_faces(faces,
                                                                 pending_face_idxs,
                                                                 face_adjacency,
                                                                 cumulative_sum)

        # ob = the outer boundary
        grow_ob_face_idxs = np.setxor1d(grow_b_face_idxs, grow_ib_face_idxs)

        # neighbours with appropriate curvature range
        grow_measure = np.where(np.logical_and(curvature_value[grow_ob_face_idxs] > min_curvature_threshold,
                                               curvature_value[grow_ob_face_idxs] < max_curvature_threshold))
        grow_face_idxs = grow_ob_face_idxs[grow_measure]

        # add to the initial sub-region
        extended_face_idxs = np.concatenate((pending_face_idxs, grow_face_idxs))

        if extended_face_idxs.shape[0] == faces_count:
            break

        else:
            pending_face_idxs = extended_face_idxs
            faces_count = pending_face_idxs.shape[0]

    return extended_face_idxs


def assign_thickness(vertices_p,
                     faces_p,
                     vertices_s,
                     faces_s,
                     sub_face_idxs,
                     thickness_factor):
    """
    this function assigns a thickness to all the primary vertices.
    The thickness of all the vertices outside the sub-region is set to zero
    The thickness of all the vertices inside the sub-region is set to half of closest distance to the secondary surface

    :param vertices_p: list of vertex positions of the primary surface
    :param faces_p: list of triangle indices of the primary surface
    :param vertices_s: list of vertex positions of the secondary surface
    :param faces_s: list of triangle indices of the secondary surface
    :param sub_face_idxs: list of facet indices corresponding to the sub-region
    :param thickness_factor: a constant which will be multiplied by the distance between two surfaces.
    This allows you to control the thickness value

    :return: the thickness profile

    """
    sub_vertex_idxs = np.unique(faces_p[sub_face_idxs].flatten())
    sd_value = igl.signed_distance(vertices_p[sub_vertex_idxs], vertices_s, faces_s, return_normals=False)[0]
    sd_value = sd_value * thickness_factor

    # thickness of all the vertices outside the sub-region
    thickness_profile = np.zeros(vertices_p.shape[0])

    # thickness of all the vertices inside the sub-region
    j = 0
    for i in sub_vertex_idxs:
        thickness_profile[i] = sd_value[j]
        j = j + 1

    return thickness_profile


def boundary_value(vertices,
                   faces,
                   external_boundary,
                   internal_boundary,
                   thickness_profile,
                   blending_order):
    """
    Computes the value at the boundary

    :param vertices:
    :param faces:
    :param external_boundary: external boundary vertex indices
    :param internal_boundary: internal boundary vertex indices
    :param thickness_profile: boundary values for all the vertices
    :param blending_order: power of harmonic operation (1: harmonic, 2: bi-harmonic, etc)

    :return: The harmonic weights determining how to interpolate between extruded and base surface
    """

    boundaries = np.concatenate((internal_boundary, external_boundary))

    boundary_thickness_value_outer = thickness_profile[external_boundary]
    boundary_thickness_value_inner = thickness_profile[internal_boundary]
    boundary_thickness_value = np.concatenate((boundary_thickness_value_inner, boundary_thickness_value_outer))

    weights = igl.harmonic_weights(vertices, faces, boundaries, boundary_thickness_value, blending_order)

    return weights


def extrude_cartilage(vertices_p,
                      faces_p,
                      sub_face_idxs,
                      harmonic_weights):
    """
    Extrudes the subset surface based on harmonic weights

    :param vertices_p: list of vertex positions of the primary surface
    :param faces_p: list of triangle indices of the primary surface
    :param sub_face_idxs: list of facet indices corresponding to the sub-region
    :param harmonic_weights: The harmonic weights computed by 'boundary_value'

    :return: extruded subset vertices
    """

    sub_vertex_idxs = np.unique(faces_p[sub_face_idxs].flatten())
    vertex_normals = igl.per_vertex_normals(vertices_p, faces_p)
    surface_normals = vertex_normals[sub_vertex_idxs]

    thickness = []
    for i in range(len(sub_vertex_idxs)):
        thickness.append(surface_normals[i] * harmonic_weights[i])
    extruded_vertices = np.copy(vertices_p)
    extruded_vertices[sub_vertex_idxs] = vertices_p[sub_vertex_idxs] + thickness

    return extruded_vertices


def norm_visualization(vertices,
                       faces):
    """

    :param vertices: list of vertices
    :param faces: list of faces

    :return: visualizes the norm of each facet
    """
    # starting point
    centroids = igl.barycenter(vertices, faces)
    face_normals = igl.per_face_normals(vertices, faces, np.array([1., 1., 1.]))
    avg_edge_length = igl.avg_edge_length(vertices, faces)

    # end point
    arrow_length = 2
    end_points = centroids + face_normals * avg_edge_length * arrow_length

    # visualization
#     frame = mp.plot(vertices, faces, c = pastel_yellow, shading = sh_true)
#     frame.add_lines(centroids, end_points, shading={"line_color": "aqua"})
    return centroids, end_points

def merge_surface_mesh(vertices_1,
                       faces_1,
                       vertices_2,
                       faces_2):
    """
    Merges two surface meshes into one

    :param vertices_1: list of vertex positions of the first surface
    :param faces_1: list of faces of the first surface
    :param vertices_2: list of vertex positions of the second surface
    :param faces_2: list of faces of the second surface

    :return: The merged vertices and surfaces of the mesh
    """
    merged_vertices = np.concatenate((vertices_1, vertices_2))
    merged_faces = np.concatenate((faces_1, faces_2 + len(vertices_1)))

    return merged_vertices, merged_faces


def smooth_and_separate_boundaries(vertices,
                                   boundary_edges,
                                   smoothing_factor,
                                   smoothing_iteration):
    """
    smooths the subset boundaries and ensures no penetration to the underlying surface

    :param vertices: list of vertex positions
    :param boundary_edges: list of the boundary edge indices
    :param smoothing_factor:
    :param smoothing_iteration: number of smoothing iterations to perform

    :return: list of separated, smooth and non-penetrating vertices
        
    """

    # determine and smooth first boundary
    boundary_1 = igl.edges_to_path(boundary_edges)[0]

    for i in range(smoothing_iteration):
        vertices = smooth_boundary(vertices, boundary_1, smoothing_factor)

    # determine if we have more boundaries and smooth them
    boundary_1_idxs = []
    for j in boundary_1:
        idx = np.where(boundary_edges == j)[0]
        boundary_1_idxs.append(idx)
    boundary_2_idxs = np.delete(boundary_edges, boundary_1_idxs, axis=0)

    if len(boundary_2_idxs) != 0:
        boundary_2 = igl.edges_to_path(boundary_2_idxs)[0]

        for i in range(smoothing_iteration):
            vertices = smooth_boundary(vertices, boundary_2, smoothing_factor)

    return vertices


def smooth_boundary(vertices,
                    b_idxs,
                    smoothing_factor):
    """
    Smooths the boundary of the surface using Laplacian smoothing

    :param vertices: list of vertex positions
    :param boundary_vertex_idxs: list of the boundary vertex indices
    :param smoothing_factor:

    :return: list of smooth and non-penetrating vertices
    """
    # add the last vertex idxs to the beginning and the first index to the end
    b_idxs = np.insert(b_idxs, 0, b_idxs[-1])
    b_idxs = np.insert(b_idxs, len(b_idxs), b_idxs[1])
    
    # new vertices
    new_vertices = np.zeros((len(b_idxs) - 2, 3))
    
    # Loop through all boundary vertices and apply smoothing
    
    for i in range(1, len(b_idxs) - 1):
        
        delta = 0.5 * ( vertices[ b_idxs[i - 1] ] + vertices[ b_idxs[i + 1] ] ) - vertices[ b_idxs[i] ]
        
        new_vertices[i - 1] = vertices[b_idxs[i]] + smoothing_factor * delta

    vertices[b_idxs[1: -1]] = new_vertices

    return vertices


def save_surface(vertices,
                 faces,
                 output_dim,
                 path
                 ):
    """
    this function saves a surface mesh (obj file format) of the generated model

    :param vertices: list of vertex positions
    :param faces: list of the triangle faces
    :param output_dim: output dimension
    :param output_name: name of the output file
    :param output_dir: the directory where the out put will be stored in

    """

    if output_dim == "m":
        vertices = vertices / 1000

    igl.write_triangle_mesh(path, vertices, faces)


def get_area(vertices,
             faces):
    """
    this function measures the surface area of a given surface mesh

    :param vertices: list of vertex positions
    :param faces: list of the triangle faces in which you want to measure the surface area

    :return: the surface area of the input triangles
    """
    surface_area = 0
    for i in range(len(faces)):
        u = vertices[faces[i, 1]] - vertices[faces[i, 0]]
        v = vertices[faces[i, 2]] - vertices[faces[i, 0]]
        n = np.cross(u, v)
        mag = np.linalg.norm(n, axis=0)
        mag = mag / 2

        surface_area += mag

    return surface_area


def make_cut_plane_view(vertices,
                        faces,
                        d=0,
                        s=0.5):
    """
    this function visualizes a plane cut view of a given mesh

    :param vertices: list of vertex positions
    :param faces: list of the triangle faces in which you want to see the cut-plane
    :param d: is the direction of the cut: x=0, y=1, z=2
    :param s: s it the size of the cut where 1 is equal to no cut and 0 removes the whole object
    """

    a = np.amin(vertices, axis=0)
    b = np.amax(vertices, axis=0)
    h = s * (b - a) + a
    c = igl.barycenter(vertices, faces)

    if d == 0:
        idx = np.where(c[:, 1] < h[1])
    elif d == 1:
        idx = np.where(c[:, 0] < h[0])
    else:
        idx = np.where(c[:, 2] < h[2])

    return idx


def get_wall(vertices_p,
             boundary_vertex_idxs1,
             boundary_vertex_idxs2 ):
    """

    :param vertices_p:
    :param boundary_vertex_idxs:
    :return:
    """
    internal_boundary = np.copy(boundary_vertex_idxs1)
    external_boundary = np.copy(boundary_vertex_idxs2)

    # closing the loop
    internal_boundary = np.append(internal_boundary, internal_boundary[0])
    external_boundary = np.append(external_boundary, external_boundary[0])

    # build the wall
    wall_faces = []

    for i in range(len(internal_boundary) - 1):
        z = [internal_boundary[i], external_boundary[i] + len(vertices_p),
             external_boundary[i + 1] + len(vertices_p)]
        x = [internal_boundary[i], external_boundary[i + 1] + len(vertices_p), internal_boundary[i + 1]]
        wall_faces.append(z)
        wall_faces.append(x)

    wall_faces = np.array(wall_faces)

    return wall_faces

def snap_to_surface ( vertices, vertices_r, faces_r ):
    """

    :param vertices:
    :param vertices_r:
    :param faces_r:
    :param penetration:
    :param gap:
    :return:
    """

    vertices_p = np.copy (vertices)
    sd_value, _, closest_points = igl.signed_distance(vertices_p, vertices_r, faces_r, return_normals=False)
    vertices_p = closest_points

    return vertices_p


def remove_penetration (vertices, vertices_r, faces_r):
    """

    :param vertices:
    :param vertices_r:
    :param faces_r:
    :return:
    """
    vertices_p = np.copy(vertices)
    sd_value, _, closest_points = igl.signed_distance(vertices_p, vertices_r, faces_r, return_normals=False)
    penetrating_vertices = np.where(sd_value <= 0)[0]
    vertices_p[penetrating_vertices] = closest_points[penetrating_vertices]

    return vertices_p


def contact_surface(vertices_1,
                    faces_1,
                    vertices_2,
                    faces_2,
                    epsilon):

    """
    This function measured the contact surface in the cartilage-cartilage interface.

    :param vertices_1: vertices: list of vertex positions
    :param face_1:
    :param vertices_2: list of vertex positions
    :param faces_2:
    :param epsilon:

    """

    # triangle centroids
    triangle_centroids = igl.barycenter(vertices_1, faces_1)

    # point to surface distance
    sd_value, _, _ = igl.signed_distance(triangle_centroids,
                                         vertices_2,
                                         faces_2,
                                         return_normals=False)

    # faces below a distance threshold
    contact_face_idxs = np.where(sd_value < epsilon)[0]

    # viz
    frame = mp.plot(vertices_1, faces_1, c=bone, shading=sh_false)
    frame.add_mesh(vertices_1, faces_1[contact_face_idxs], c=organ, shading=sh_true)

    contact_area = get_area(vertices_1, faces_1[contact_face_idxs])

    print("contact surface area is: ", np.round(contact_area, 2))


def neighbouring_info(vertices, faces):

    """
    This function measured the contact surface in the cartilage-cartilage interface.

    :param vertices:
    :param faces:

    :return:

    """

    # adjacency info
    face_adjacency, cumulative_sum = igl.vertex_triangle_adjacency(faces, len(vertices))

    # part.1 boundary vertex indices
    boundary_vertex_idxs = igl.boundary_loop(faces)

    # part.2 neighboring faces to these vertices
    boundary_face_idxs = []
    container = []
    for j in boundary_vertex_idxs:
        for k in range(cumulative_sum[j], cumulative_sum[j + 1]):
            container += [face_adjacency[k]]
        boundary_face_idxs.append(container)
        container = []

    # part.3 find the face neighbors to these faces
    tt_info = igl.triangle_triangle_adjacency(faces)[0]

    container = []
    neigh_face_list = []
    for i in boundary_face_idxs:
        a = tt_info[i]
        a = np.unique(a.flatten())
        a = a.tolist()
        neigh_face_list.append(a[1:])
        a = 0

    return boundary_vertex_idxs, boundary_face_idxs, neigh_face_list


def get_dihedral_angle (vertices, faces, face_idxs, neigh_face_list ):

    """
    This function measured the contact surface in the cartilage-cartilage interface.

    :param vertices:
    :param faces:

    :return:

    """

    # face normals
    face_normals = igl.per_face_normals(vertices, faces[face_idxs], np.array([1., 1., 1.]))

    # measure dihedral angles
    container = []
    angles = []

    for i in neigh_face_list:
        # make all possible pairs
        pairs = []
        for j in range(len(i)):
            for k in range(len(i)):
                a = [i[j], i[k]]
                pairs.append(a)
        pairs = np.array(pairs)

        # find the dihedral angle of each pair
        for l in pairs:
            cos = np.dot(face_normals[l[0]], face_normals[l[1]])
            cos = np.clip(cos, -1, 1)
            container.append(np.arccos(cos))
        angles.append(container)
        container = []

    max_angles = []
    for i in angles:
        container = np.max(i)
        max_angles.append(container)
        container = []

    return max_angles

def fix_boundary(vertices, faces, face_idxs, boundary_vertex_idxs, folded_vertex_idxs ):

    """
    This function measured the contact surface in the cartilage-cartilage interface.

    :param vertices:
    :param faces:

    :return:

    """

    # adjacency info
    face_adjacency, cumulative_sum = igl.vertex_triangle_adjacency(faces[face_idxs], len(vertices))

    container = []
    per_vertex_neighbour_face_idxs = []
    for i in boundary_vertex_idxs[folded_vertex_idxs]:
        for k in range(cumulative_sum[i], cumulative_sum[i + 1]):
            container += [face_adjacency[k]]
        per_vertex_neighbour_face_idxs.append(container)
        container = []

    all_neighbour_face_idxs = []
    for i in per_vertex_neighbour_face_idxs:
        for j in i:
            all_neighbour_face_idxs.append(j)

    all_array = np.array(all_neighbour_face_idxs)
    count = []
    for i in all_neighbour_face_idxs:
        count.append(all_neighbour_face_idxs.count(i))

    count = np.array(count)

    mutual = np.where(count == 2)[0]

    faulty_face_idxs = all_array[mutual]
    faulty_face_idxs = np.unique(faulty_face_idxs)

    # vz
    print("The pink triangles will be removed:")
    print("No.........................")

    frame = mp.plot(vertices, faces[face_idxs], c=pastel_blue, shading=sh_false)
    frame.add_mesh(vertices, faces[face_idxs[faulty_face_idxs]], c=sweet_pink, shading=sh_true)

#     face_idxs = np.delete(face_idxs, faulty_face_idxs, axis=0)

    return face_idxs


"""
Cartilage generation parameters

Sets the parameter of the joint for future cartilage generation.

@param neighbourhood_size: How far away (in terms of edges) a vertex can be, while still considered
a neighbour. Also referred to as 'k-ring' in some literature.
@param curvature_type: choose between "gaussian", "mean", "minimum", "maximum"
Refers to the gaussian, mean, the maximum and the minimum of the principal curvatures, respectively.
@param gap_distance: The distance between the primary bone and the secondaries.
@param trimming_iteration: number of times the trimming step should be performed.
@param min_curvature_threshold: The minimum curvature where we will consider the surface to be part of the cartilage
@param max_curvature_threshold: The maximum curvature where we will consider the surface to be part of the cartilage
@param blending_order: Order of the harmonic weight computation during cartilage generation.
@param smoothing_factor: The size of the smoothing step (this is similar to the 'h' parameter in mean curvature flow)
@param smoothing_iteration_base: The number of times the smoothing step should be performed on the base layer.
@param smoothing_iteration_extruded_base: The number of times the smoothing step should be performed on the extruded layer.
@param output_dimension: The scale of the output ("mm" = millimeters, "m" = meters).
@param thickness_factor: a constant which will be multiplied by the distance between two surfaces.This allows you to 
control the thickness value.
"""


class Var:

    def __init__(self):
        self.neighbourhood_size: int = 20
        self.curvature_type: str = "mean"
        self.gap_distance: float = 4.2
        self.trimming_iteration: int = 7
        self.min_curvature_threshold: float = 0.026
        self.max_curvature_threshold: float = np.inf
        self.blending_order: int = 2
        self.smoothing_factor: float = 0.5
        self.smoothing_iteration_base: int = 3
        self.smoothing_iteration_extruded_base: int = 3
        self.extend_cartilage_flag = True
        self.curve_info = False
        self.upsampling_iteration: int = 0
        self.thickness_factor: float = 0.5
        self.fix_boundary = True
        self.no_extend_trimming_iteration: int = 3

    def reset(self):
        self.neighbourhood_size = 20
        self.curvature_type = "mean"
        self.gap_distance = 4.2
        self.trimming_iteration = 7
        self.min_curvature_threshold = 0.026
        self.max_curvature_threshold = np.inf
        self.blending_order = 2
        self.smoothing_factor = 0.5
        self.smoothing_iteration_base = 3
        self.smoothing_iteration_extruded_base = 3
        self.extend_cartilage_flag = True
        self.curve_info = False
        self.upsampling_iteration = 0
        self.thickness_factor: float = 0.5
        self.fix_boundary = True
        self.no_extend_trimming_iteration = 2


" Colors and Eye-candies"

# color definitions
pastel_light_blue = np.array([179, 205, 226]) / 255.
pastel_blue = np.array([111, 184, 210]) / 255.
bone = np.array([0.92, 0.90, 0.85])
pastel_orange = np.array([255, 126, 35]) / 255.
pastel_yellow = np.array([241, 214, 145]) / 255.
pastel_green = np.array([128, 174, 128]) / 255.
mandible = np.array([222, 198, 101]) / 255
tooth = np.array([255, 250, 220]) / 255
organ = np.array([221, 130, 101]) / 255.
green = np.array([128, 174, 128]) / 255.
blue = np.array([111, 184, 210]) / 255.
sweet_pink = np.array([0.9, 0.4, 0.45])  #230, 102, 115
rib = np.array([253, 232, 158]) / 255.
skin = np.array([242, 209, 177]) / 255.
chest = np.array([188, 95, 76]) / 255.


# Meshplot settings
sh_true = {'wireframe': True, 'flat': True, 'side': 'FrontSide', 'reflectivity': 0.1, 'metalness': 0}
sh_false = {'wireframe': False, 'flat': True, 'side': 'FrontSide', 'reflectivity': 0.1, 'metalness': 0}
