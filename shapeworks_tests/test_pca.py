import math

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
import pyacvd
from sklearn.decomposition import PCA

parent_dir = Path(__file__).parent


def prepare_meshes():
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


def test_pca_ellipsoid():
    particle_files_dir = parent_dir / "test_data" / "ellipsoid_particles"
    particle_files_list = glob(str(particle_files_dir / "*world.particles"))
    ps = sw.ParticleSystem(particle_files_list)

    particles = ps.Particles().T
    pv_points = [pv.PointSet(p) for p in particles]
    points = np.array([p.points for p in pv_points])

    embedder = PCA_Embbeder(points, num_dim=len(points) - 1)

    mean_data = embedder.mean_data
    project_zeros = embedder.project(np.zeros(len(points) - 1))

    np.testing.assert_allclose(project_zeros, mean_data)

    for scores, p in zip(embedder.PCA_scores, points):
        np.testing.assert_allclose(embedder.project(scores), p)

    # # Plot meshes
    # c = matplotlib.colormaps["viridis"]
    # side = math.ceil(math.sqrt(len(pv_points)))
    # shape = (side, side)
    # p = pv.Plotter(shape=shape)
    # for i, points in enumerate(pv_points):
    #     p.subplot(*np.unravel_index(i, shape))
    #     p.add_mesh(points, color=c((i * len(pv_points) / c.N) % c.N), label=f"ellipse {i + 1}",
    #                render_points_as_spheres=True, point_size=5)
    #     p.show_axes()
    #     p.add_legend()
    #     p.show_bounds()
    # p.link_views()
    # p.show()

    # Plot meshes overlapping
    c = matplotlib.colormaps["viridis"]
    p = pv.Plotter()
    for i, points in enumerate(pv_points):
        p.add_mesh(points, color=c((i * len(pv_points) / c.N) % c.N), label=f"ellipse {i + 1}",
                   render_points_as_spheres=True, point_size=5)

    p.show_axes()
    p.add_legend()
    p.show_bounds()
    p.show()

    # # Plot eigenvectors
    # p = pv.Plotter(shape=(1, len(pv_points)), window_size=(5 * 300, 300))
    # for i in range(len(pv_points)):
    #     p.subplot(0, i)
    #     p.add_title(f"mode {i + 1} eigenvectors", font_size=8)
    #     eigenvectors = embedder.eigen_vectors[:, i].reshape(mean_data.shape)
    #     p.add_points(pv.PointSet(eigenvectors))
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


# Todo finish this
def test_pca_load_and_save():
    pass


# Todo finish this
def test_pca_project_with_stdev():
    # Prepare meshes with known stdev
    std = 0.5
    mean = 1.5
    n_samples = 40

    rng = np.random.default_rng(0)
    scales = rng.normal(mean, std, n_samples)
    scales = np.sort(scales)

    meshes = []
    for scale in scales:
        mesh = pv.Sphere(theta_resolution=20, phi_resolution=20, radius=1.5, center=[0, 0, 0]).scale([scale, 1, 1],
                                                                                                     inplace=False)
        meshes.append(mesh)

    points = np.array([mesh.points for mesh in meshes])
    # Add some noise. The test fails without this
    points = points + rng.normal(0, 0.01, points.shape)

    pv_points = [pv.PointSet(points) for points in points]
    embedder = PCA_Embbeder(points, num_dim=len(meshes) - 1)

    mean_data = embedder.mean_data
    project_zeros = embedder.project(np.zeros(len(points) - 1))

    np.testing.assert_allclose(project_zeros, mean_data)

    for scores, p in zip(embedder.PCA_scores, points):
        np.testing.assert_allclose(embedder.project(scores), p)

    # Test with stdevs
    largest_scale = scales[-1]
    stds_from_mean = (largest_scale - mean) / std
    test_scores = np.zeros(len(meshes) - 1)
    test_scores[0] = stds_from_mean

    projected = embedder.project_with_stdev(test_scores)

    scale_max, scale_min = np.max(scales), np.min(scales)
    score_max, score_min = np.max(embedder.PCA_scores[:, 0]), np.min(embedder.PCA_scores[:, 0])

    scale_std = np.std(scales)
    score_std = np.std(embedder.PCA_scores[:, 0])

    largest_scale_in_score_stds = stds_from_mean * score_std

    pca = PCA(svd_solver="auto")
    centered_points = points
    pca_loadings = pca.fit_transform(centered_points.reshape([centered_points.shape[0], -1]))

    np.testing.assert_allclose(pca_loadings[:, 0], embedder.PCA_scores[:, 0])

    for scores, p in zip(pca_loadings, points):
        np.testing.assert_allclose(pca.inverse_transform(scores).reshape([-1, 3]), p)

    print("hello")
    pass

    # # Plot meshes overlapping
    # c = matplotlib.colormaps["viridis"]
    # p = pv.Plotter()
    # for i, points in enumerate(pv_points):
    #     p.add_mesh(points, color=c((i * len(pv_points) / c.N) % c.N), label=f"ellipse {i + 1}",
    #                render_points_as_spheres=True, point_size=5)
    #
    # p.show_axes()
    # p.add_legend()
    # p.show_bounds()
    # p.show()

    # # Plot eigenvectors
    # p = pv.Plotter(shape=(1, len(meshes)), window_size=(5 * 300, 300))
    # for i in range(len(meshes)):
    #     p.subplot(0, i)
    #     p.add_title(f"mode {i + 1} eigenvectors", font_size=8)
    #     eigenvectors = embedder.eigen_vectors[:, i].reshape(embedder.mean_data.shape)
    #     p.add_points(pv.PointSet(eigenvectors))
    #     p.show_axes()
    #     p.show_bounds()
    #
    # p.link_views()
    # p.show()

    # # Plot meshes individually
    # c = matplotlib.colormaps["viridis"]
    # side = math.ceil(math.sqrt(len(pv_points)))
    # shape = (side, side)
    # p = pv.Plotter(shape=shape)
    # for i, points in enumerate(pv_points):
    #     p.subplot(*np.unravel_index(i, shape))
    #     p.add_mesh(points, color=c((i * len(pv_points) / c.N) % c.N), label=f"ellipse {i + 1}",
    #                render_points_as_spheres=True, point_size=5)
    #     p.show_axes()
    #     p.add_legend()
    #     p.show_bounds()
    # p.link_views()
    # p.show()

    pass
