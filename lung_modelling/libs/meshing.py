import numpy as np
import pyvista as pv
from pyvista_tools import pyvista_faces_to_1d
import skimage


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
