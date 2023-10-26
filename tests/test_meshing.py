import numpy as np
import medpy.io
from lung_modelling import voxel_to_mesh, find_connected_faces
import numpy.testing
from pathlib import Path
import matplotlib
import pyvista as pv
from pyvista_tools import pyvista_faces_to_2d

parent_dir = Path(__file__).parent


def test_voxel_to_mesh():
    """
    This test demonstrates the use of the spacing, offset and direction parameters of the medpy header.
    To really check that it is working, test that the generated .mhd file is aligned with the generated .stl file in
    3D Slicer. This shows that our interpretation of spacing, offset, and direction are the same as 3D slicer's.
    """

    # Orientable image in scalar array form
    voxel_matrix = np.array([
        [[0, 0, 0, 0, ],
         [0, 0, 0, 0, ],
         [0, 0, 0, 0, ],
         [0, 0, 1, 0, ], ],

        [[0, 0, 1, 0, ],
         [0, 0, 1, 0, ],
         [0, 1, 1, 0, ],
         [0, 1, 1, 0, ], ],
    ])

    voxel_matrix = np.pad(voxel_matrix, 1)  # Pad so that faces can be generated on all sides
    voxel_matrix = np.kron(voxel_matrix, np.ones((2, 2, 2)))  # Scale up array to make visualization easier

    spacing = (1, 2, 3)
    offset = (1, 2, 3)
    direction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)

    header = medpy.io.header.Header(spacing=spacing, offset=offset)
    header.set_direction(direction)

    medpy.io.save(voxel_matrix, str(parent_dir / Path("test_data/test_file.mhd")), hdr=header, use_compression=True)

    image_data, header = medpy.io.load(str(parent_dir / Path("test_data/test_file.mhd")))

    mesh = voxel_to_mesh(image_data, header.spacing, header.direction, header.offset)
    mesh.save("test_data/test_file.stl")

    np.testing.assert_array_equal(mesh.points, correct_points)


correct_points = np.array([[2.5, -14., 21.],
                           [3., -14., 19.5],
                           [3., -13., 21.],
                           [3., -13., 24.],
                           [2.5, -14., 24.],
                           [3., -14., 25.5],
                           [2.5, -16., 21.],
                           [3., -16., 19.5],
                           [2.5, -16., 24.],
                           [3., -16., 25.5],
                           [3., -17., 21.],
                           [3., -17., 24.],
                           [4., -14., 19.5],
                           [4., -13., 21.],
                           [4., -13., 24.],
                           [4., -14., 25.5],
                           [4., -16., 19.5],
                           [4., -16., 25.5],
                           [4., -17., 21.],
                           [4., -17., 24.],
                           [4.5, -2., 21.],
                           [5., -2., 19.5],
                           [5., -1., 21.],
                           [5., -1., 24.],
                           [4.5, -2., 24.],
                           [5., -2., 25.5],
                           [4.5, -4., 21.],
                           [5., -4., 19.5],
                           [4.5, -4., 24.],
                           [5., -4., 25.5],
                           [4.5, -6., 21.],
                           [5., -6., 19.5],
                           [4.5, -6., 24.],
                           [5., -6., 25.5],
                           [4.5, -8., 21.],
                           [5., -8., 19.5],
                           [4.5, -8., 24.],
                           [5., -8., 25.5],
                           [4.5, -10., 15.],
                           [5., -10., 13.5],
                           [5., -9., 15.],
                           [5., -9., 18.],
                           [4.5, -10., 18.],
                           [4.5, -10., 21.],
                           [4.5, -10., 24.],
                           [5., -10., 25.5],
                           [4.5, -12., 15.],
                           [5., -12., 13.5],
                           [4.5, -12., 18.],
                           [4.5, -12., 21.],
                           [4.5, -12., 24.],
                           [5., -12., 25.5],
                           [4.5, -14., 15.],
                           [5., -14., 13.5],
                           [4.5, -14., 18.],
                           [5., -14., 25.5],
                           [4.5, -16., 15.],
                           [5., -16., 13.5],
                           [4.5, -16., 18.],
                           [5., -16., 25.5],
                           [5., -17., 15.],
                           [5., -17., 18.],
                           [5., -17., 21.],
                           [5., -17., 24.],
                           [6., -2., 19.5],
                           [6., -1., 21.],
                           [6., -1., 24.],
                           [6., -2., 25.5],
                           [6., -4., 19.5],
                           [6., -4., 25.5],
                           [6., -6., 19.5],
                           [6., -6., 25.5],
                           [6., -8., 19.5],
                           [6., -8., 25.5],
                           [6., -10., 13.5],
                           [6., -9., 15.],
                           [6., -9., 18.],
                           [6., -10., 25.5],
                           [6., -12., 13.5],
                           [6., -12., 25.5],
                           [6., -14., 13.5],
                           [6., -14., 25.5],
                           [6., -16., 13.5],
                           [6., -16., 25.5],
                           [6., -17., 15.],
                           [6., -17., 18.],
                           [6., -17., 21.],
                           [6., -17., 24.],
                           [6.5, -2., 21.],
                           [6.5, -2., 24.],
                           [6.5, -4., 21.],
                           [6.5, -4., 24.],
                           [6.5, -6., 21.],
                           [6.5, -6., 24.],
                           [6.5, -8., 21.],
                           [6.5, -8., 24.],
                           [6.5, -10., 15.],
                           [6.5, -10., 18.],
                           [6.5, -10., 21.],
                           [6.5, -10., 24.],
                           [6.5, -12., 15.],
                           [6.5, -12., 18.],
                           [6.5, -12., 21.],
                           [6.5, -12., 24.],
                           [6.5, -14., 15.],
                           [6.5, -14., 18.],
                           [6.5, -14., 21.],
                           [6.5, -14., 24.],
                           [6.5, -16., 15.],
                           [6.5, -16., 18.],
                           [6.5, -16., 21.],
                           [6.5, -16., 24.]])


def test_find_connected_faces():
    inner = pv.Box(quads=False)
    outer = pv.Box(quads=False).scale([2, 2, 2])

    merged = pv.merge([inner, outer])

    groups = find_connected_faces(pyvista_faces_to_2d(merged.faces))

    g1 = {0, 1, 2, 3, 4, 5, 6, 7}
    g2 = {8, 9, 10, 11, 12, 13, 14, 15}

    # p = pv.Plotter()
    # p.add_mesh(merged.extract_all_edges())
    # c = matplotlib.colormaps["hsv"]
    # for group, points in groups.items():
    #     color = c((group + 1) / (len(groups)))
    #     p.add_points(merged.points[points], color=color)
    #
    # p.show()

    assert set(np.array(groups[0]).ravel()) == g1
    assert set(np.array(groups[1]).ravel()) == g2
