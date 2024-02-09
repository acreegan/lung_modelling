import shapeworks as sw
import pyvista as pv
import matplotlib
from lung_modelling.shapeworks_libs import PCA_Embbeder
import DataAugmentationUtils
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from glob import glob


def prepare_meshes():
    # center = [-5000, 500, 0]

    # mesh_1 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([0.8, 1, 1],
    #                                                                                             inplace=False)
    # mesh_2 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([0.9, 1, 1],
    #                                                                                             inplace=False)
    # mesh_3 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([1, 1, 1],
    #                                                                                             inplace=False)
    # mesh_4 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([1.1, 1, 1],
    #                                                                                             inplace=False)
    # mesh_5 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([1.2, 1, 1],
    #                                                                                             inplace=False)

    center = [10, 10, 0]

    mesh_1 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([0.8, 1.3, 1],
                                                                                                inplace=False)
    mesh_2 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([0.9, 1.2, 1],
                                                                                                inplace=False)
    mesh_3 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([1, 1, 1],
                                                                                                inplace=False)
    mesh_4 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([1.1, 0.8, 1],
                                                                                                inplace=False)
    mesh_5 = pv.Sphere(theta_resolution=10, phi_resolution=10, radius=1.5, center=center).scale([1.2, 0.7, 1],
                                                                                                inplace=False)

    meshes = [mesh_1, mesh_2, mesh_3, mesh_4, mesh_5]
    middle_mesh = mesh_3

    return meshes, middle_mesh


def test_pca():
    meshes, middle_mesh = prepare_meshes()

    all_points = np.array([mesh.points for mesh in meshes])
    embedder = PCA_Embbeder(all_points, num_dim=len(meshes) - 1)

    mean_data = embedder.mean_data
    project_zeros = embedder.project(np.zeros(len(meshes) - 1))

    np.testing.assert_allclose(mean_data, middle_mesh.points)  # In this case our middle mesh is the mean
    np.testing.assert_allclose(project_zeros,
                               middle_mesh.points)  # Check that projecting all zero scores results in the mean

    # p = pv.Plotter()
    # c = matplotlib.colormaps["Set1"]
    # for i, mesh in enumerate(meshes):
    #     p.add_mesh(mesh.extract_all_edges(), color=c(i), label=f"mesh {i + 1}")
    # p.show_axes()
    # p.add_legend()
    # p.show_bounds()
    # p.show()
    #
    # p = pv.Plotter(shape=(1, len(meshes)), window_size=(5 * 300, 300))
    # for i in range(len(meshes)):
    #     p.subplot(0, i)
    #     p.add_title(f"mode {i + 1} eigenvectors", font_size=8)
    #     eigenvectors = embedder.eigen_vectors[:, i].reshape(middle_mesh.points.shape)
    #     p.add_points(eigenvectors)
    #     p.show_axes()
    #     p.show_bounds()
    #
    # p.link_views()
    # p.show()


def test_compare_pca_methods():
    meshes, middle_mesh = prepare_meshes()

    all_points = np.array([mesh.points for mesh in meshes])
    embedder = PCA_Embbeder(all_points, num_dim=len(meshes) - 1)

    # Go through temp directory because ParticleSystem can only be created with files
    with tempfile.TemporaryDirectory() as td:
        for i, mesh in enumerate(meshes):
            filename = str(Path(td) / f"{i}_particles")
            np.savetxt(filename, mesh.points)

        files = glob(str(Path(td) / "*particles"))
        particle_system = sw.ParticleSystem(files)

    shape_statistics = sw.ParticleShapeStatistics()
    shape_statistics.PCA(particleSystem=particle_system, domainsPerShape=1)

    # ShapeStatistics is backwards from Embedder, thus the flip
    # Test only eigenvectors from modes with shape changes (e.g., the first 2)
    # Todo: not sure why they sometimes have different signs
    a = np.abs(np.flip(shape_statistics.eigenVectors(), 1))[:, :2]
    b = np.abs(embedder.eigen_vectors)[:, :2]

    assert np.allclose(a, b)

    # Todo: not sure why there is a factor of 2
    c = embedder.eigen_values[0] / 2
    d = shape_statistics.eigenValues()[4]

    assert np.allclose(c, d)
