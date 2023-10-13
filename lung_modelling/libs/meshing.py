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


def voxel_to_mesh(dataarray, spacing=None, direction=None, offset=None) -> pv.PolyData:
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

    verts, faces, norms, vals = skimage.measure.marching_cubes(dataarray, step_size=1, spacing=spacing,
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
    fixed_mesh = mesh.copy()
    fixed_mesh.points = meshfix.v
    fixed_mesh.faces = pyvista_faces_to_1d(meshfix.f)

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
