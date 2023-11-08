from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista_tools import pyvista_faces_to_1d, pyvista_faces_to_2d
import skimage
from omegaconf import DictConfig
import pymeshfix


def extract_section(image_data, section_value):
    """
    Select individual section from segmented data based on pre-defined section values

    Parameters
    ----------
    image_data
    section_value

    Returns
    -------
    lobe_array

    """

    section_array = np.where(image_data == section_value, 1, 0)

    return section_array


def voxel_to_mesh(dataarray, spacing=None, direction=None, offset=None, step_size=1) -> pv.PolyData:
    """
    Creat mesh from discrete scalar array


    Parameters
    ----------
    dataarray
        3D discrete scalar array
    spacing
        Spacing parameter for array in X, Y, and Z
    direction
        Affine transformation matrix
    offset
        Offset position of mesh
    step_size
        step_size for marching cubes

    Returns
    -------
    mesh

    """
    if spacing is None:
        spacing = [1, 1, 1]

    if direction is None:
        direction = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if offset is None:
        offset = np.array([0, 0, 0])

    verts, faces, norms, vals = skimage.measure.marching_cubes(dataarray, step_size=step_size, spacing=spacing,
                                                               allow_degenerate=False)

    verts = np.matmul(verts, direction)
    verts = verts + offset
    norms = np.matmul(norms, direction)

    surf = pv.PolyData(verts, pyvista_faces_to_1d(faces))
    surf.point_data.active_normals = norms

    surf = surf.compute_normals(auto_orient_normals=True)

    return surf


def fix_mesh(mesh: pv.DataSet, repair_kwargs: dict = None, return_holes=False) -> pv.DataSet | tuple[
    pv.DataSet, pv.PolyData]:
    """
    Call the meshfix.repair function on a Pyvista dataset and return a fixed copy of the dataset along with the meshfix
    holes

    Parameters
    ----------
    mesh
        Pyvista Dataset
    repair_kwargs
        Kwargs for meshfix.repair
    return_holes
        Flag to return holes if desired

    Returns
    -------
        Fixed copy of the input mesh, holes identified by meshfix

    """

    if repair_kwargs is None:
        repair_kwargs = {}
    meshfix = pymeshfix.MeshFix(mesh.points, pyvista_faces_to_2d(mesh.faces))
    holes = meshfix.extract_holes()
    meshfix.repair(**repair_kwargs)
    fixed_mesh = meshfix.mesh

    if return_holes:
        return fixed_mesh, holes
    else:
        return fixed_mesh


def refine_mesh(mesh: pv.PolyData, params: DictConfig = None):
    """
    Refine a mesh

    Parameters
    ----------
    mesh
        Mesh to refine
    params
        Parameters

    Returns
    -------
    Refined lobe surface

    """
    if params is None:
        params = DictConfig(
            {"n_iter": 10, "feature_smoothing": False, "edge_angle": 15, "feature_angle": 45, "relaxation_factor": 1,
             "target_reduction": 0.01, "volume_preservation": True, "hole_size": 100, "fix_mesh": False})

    # Can try remeshing using pyacvd
    # clus = pyacvd.Clustering(mesh)
    # clus.subdivide(3)
    # clus.cluster(20000)
    # remesh = clus.create_mesh()

    mesh_smooth = mesh.smooth(n_iter=params.n_iter, feature_smoothing=params.feature_smoothing,
                              edge_angle=params.edge_angle, feature_angle=params.feature_angle,
                              relaxation_factor=params.relaxation_factor)

    mesh_dec = mesh_smooth.decimate(target_reduction=params.target_reduction,
                                    volume_preservation=params.volume_preservation)
    mesh_filled = mesh_dec.fill_holes(hole_size=params.hole_size)

    mesh_fixed = fix_mesh(mesh_filled) if params.fix_mesh else mesh_filled

    # Recomputed normals just in case..
    mesh_fixed = mesh_fixed.compute_normals(auto_orient_normals=True)

    return mesh_fixed


def find_connected_faces(faces: list, return_points=False) -> dict | [dict, dict]:
    """
    Find all groups of connected faces in a list

    Parameters
    ----------
    faces
        List of faces, each represented by a list of point indices
    return_points
        Option to return groups of connected points also

    Returns
    -------
    connected_faces
        Dict of groups of connected faces
    connected_points
        Dict of groups of connected points

    """
    point_groups, connected_faces = initial_point_grouping(faces)

    # Now iteratively merge connected groups until there are none left to merge
    num_groups = len(point_groups)
    while True:
        point_groups, new_connected_groups = merge_groups(point_groups)

        new_connected_faces = {}
        for group, groups in new_connected_groups.items():
            new_connected_faces[group] = list(set(flatten([connected_faces[i] for i in groups])))

        connected_faces = new_connected_faces

        # If we didn't merge any, we're done
        if len(point_groups) == num_groups:
            break
        else:
            num_groups = len(point_groups)

    if return_points:
        return connected_faces, point_groups
    else:
        return connected_faces


def initial_point_grouping(faces: list):
    """
    Create an initial grouping of points using a quick naive approach. Loop through the faces and group each new one
    with faces that we have already seen points from.

    Parameters
    ----------
    faces
        List of faces, each represented by a list of point indices


    Returns
    -------
    point_groups
        Groups of points that lie in connected faces
    face_groups
        Indices of the faces that make up each point group

    """
    # Assign initial grouping to points
    next_group = 0
    grouped_points = {}
    point_groups = {}
    face_groups = {}
    for i, face in enumerate(faces):
        for point in face:
            if group := grouped_points.get(point):
                # All the rest of the points in the face are in the same group
                for point in face:
                    grouped_points[point] = group

                # Add all points in the face to the group list, and record the face index
                point_groups[group] = point_groups[group] + list(face)
                face_groups[group].append(i)
                break

        else:
            # No point in the face is in an existing group. Start a new one.
            for point in face:
                grouped_points[point] = next_group

            # Start a new list of points for the group, and record the face index
            point_groups[next_group] = list(face)
            face_groups[next_group] = [i]
            next_group += 1

    return point_groups, face_groups


def merge_groups(point_groups: dict) -> [dict, dict]:
    """
    Run one iteration merging dict values which contain intersecting points

    Parameters
    ----------
    point_groups
        Groups of points that lie in connected faces

    Returns
    -------
    merged_point_groups
        point_groups after one iteration of merging

    merged_group_original_keys
        keys in original point_groups that were merged together
    """
    merged_point_groups = {}
    merged_group_original_keys = {}

    while True:
        # Loop through like this because we will be removing items from the dict
        check_group, check_points = next(iter(point_groups.items()))

        # Check each group for intersections with all others, cumulatively adding points
        points_to_merge = check_points
        groups_to_merge = [check_group]
        for group, points in point_groups.items():
            if group == check_group:
                continue

            if len(np.intersect1d(points_to_merge, points)) > 0:
                points_to_merge.extend(points)
                groups_to_merge.append(group)

        # Merge connected points into the first original group
        merged_point_groups[check_group] = points_to_merge
        merged_group_original_keys[check_group] = groups_to_merge

        # Delete already merged groups from the search dict
        point_groups = {group: points for group, points in point_groups.items() if group not in groups_to_merge}

        # If nothing left, break out
        if len(point_groups) == 0:
            break

    return merged_point_groups, merged_group_original_keys


def flatten(l: list) -> list:
    """
    Flatten a list

    Parameters
    ----------
    l
        list to flatten

    Returns
    -------
    Flattened list

    """
    return [item for sublist in l for item in sublist]
