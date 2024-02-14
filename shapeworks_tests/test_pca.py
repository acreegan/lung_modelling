import shapeworks as sw
import pyvista as pv
from lung_modelling.shapeworks_libs import PCA_Embbeder
import numpy as np
import tempfile
from pathlib import Path
from glob import glob
from sklearn.decomposition import PCA

parent_dir = Path(__file__).parent


def test_compare_pca_methods():
    # Prepare meshes with known stdev
    # ------------------------------------------------------------------------------------------------------------------
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

    # Method 1: Shapeworks PCA embedder
    # ------------------------------------------------------------------------------------------------------------------
    embedder = PCA_Embbeder(points, num_dim=len(meshes) - 1)

    mean_data = embedder.mean_data
    project_zeros = embedder.project(np.zeros(len(points) - 1))

    np.testing.assert_allclose(project_zeros, mean_data)

    for scores, p in zip(embedder.PCA_scores, points):
        np.testing.assert_allclose(embedder.project(scores), p)

    # Method 2: sklearn PCA
    # ------------------------------------------------------------------------------------------------------------------
    pca = PCA(svd_solver="auto")
    pca_loadings = pca.fit_transform(points.reshape([points.shape[0], -1]))

    np.testing.assert_allclose(pca_loadings[:, 0], embedder.PCA_scores[:, 0])

    for scores, p in zip(pca_loadings, points):
        np.testing.assert_allclose(pca.inverse_transform(scores).reshape([-1, 3]), p)

    # Method 3: Shapeworks ShapeStatistics
    # Go through temp directory because ParticleSystem can only be created with files
    # ------------------------------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        for i, p in enumerate(points):
            filename = str(Path(td) / f"{i}_particles")
            np.savetxt(filename, p)

        files = glob(str(Path(td) / "*particles"))
        particle_system = sw.ParticleSystem(files)

    shape_statistics = sw.ParticleShapeStatistics()
    shape_statistics.PCA(particleSystem=particle_system, domainsPerShape=1)
    shape_statistics.principalComponentProjections()

    loadings = np.flip(np.sort(shape_statistics.pcaLoadings()[:, 0]))
    # This API does not yet have an inverse function

    # Compare loadings of all methods
    # ------------------------------------------------------------------------------------------------------------------
    np.testing.assert_allclose(loadings, embedder.PCA_scores[:, 0])
    np.testing.assert_allclose(pca_loadings[:, 0], embedder.PCA_scores[:, 0])


# Todo finish this
def test_pca_load_and_save():
    # Prepare meshes...
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

    # Create PCA embedder
    embedder = PCA_Embbeder(points, num_dim=len(meshes) - 1)

    # Write and read from file
    with tempfile.TemporaryDirectory() as td:
        embedder.write_PCA(Path(td), score_option="full")
        embedder2 = PCA_Embbeder.from_directory(Path(td))

    for scores1, scores2, p in zip(embedder.PCA_scores, embedder2.PCA_scores, points):
        np.testing.assert_allclose(embedder.project(scores1), p)
        np.testing.assert_allclose(embedder2.project(scores2), p)

    # Write and read from file without scores
    with tempfile.TemporaryDirectory() as td:
        embedder.write_PCA(Path(td), score_option="none")
        embedder_2 = PCA_Embbeder.from_directory(Path(td))

    for scores, p in zip(embedder.PCA_scores, points):
        np.testing.assert_allclose(embedder.project(scores), p)
        np.testing.assert_allclose(embedder_2.project(scores), p)
