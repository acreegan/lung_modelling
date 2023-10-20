import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import pyvista as pv
from omegaconf import DictConfig
from pyvista_tools import pyvista_faces_to_2d

from lung_modelling.app.shapeworks_tasks import MeshTransformSW
from lung_modelling.workflow_manager import DatasetLocator
import shapeworks as sw
import matplotlib


def test_mesh_transform_sw_single_domain():
    # Create some sample meshes
    left = pv.Box().translate([-4, 0, 0], inplace=False)
    center = pv.Box()
    right = pv.Box().translate([3, 0, 0], inplace=False)

    # Create SPARC data structure in temp files
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory(dir=root) as derivative, \
            tempfile.TemporaryDirectory(dir=root) as pool_derivative, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="left_") as left_dir, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="center_") as center_dir, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="right_") as right_dir:
        dataloc = DatasetLocator(Path(root), "", Path(derivative).relative_to(root), "")

        # Save meshes in directory structure and create configs
        left.save(str(dataloc.abs_derivative / Path(left_dir).stem / "left.vtk"))
        center.save(str(dataloc.abs_derivative / Path(center_dir).stem / "center.vtk"))
        right.save(str(dataloc.abs_derivative / Path(right_dir).stem / "right.vtk"))

        # Use center as the reference
        initialize_result = {"reference_meshes":
                                 {"combined_reference_mesh": {"points": np.array(center.points),
                                                              "faces": np.array(pyvista_faces_to_2d(center.faces))}}}

        dirs_list = [(Path(left_dir).stem, "", ""), (Path(center_dir).stem, "", ""),
                     (Path(right_dir).stem, "", "")]

        output_directory = "."
        task_config = DictConfig({"source_directory": ".", "params": {"iterations": 10}})

        results = []
        for dir, _, _ in dirs_list:
            result = MeshTransformSW.work(source_directory_primary=dataloc.abs_primary / dir,
                                          source_directory_derivative=dataloc.abs_derivative / dir,
                                          output_directory=dataloc.abs_derivative / dir / output_directory,
                                          dataset_config=None,
                                          task_config=task_config,
                                          initialize_result=initialize_result)
            results.append(result)

        # Load results and apply transform to original meshes
        loaded_results = []
        meshes = [left, center, right]
        transformed_meshes = []
        for result, mesh in zip(results, meshes):
            result = np.load(glob(f"{result[0]}*")[0])
            loaded_results.append(result)
            transformed_meshes.append(mesh.transform(result, inplace=False))

        # p = pv.Plotter()
        # for i, mesh in enumerate(transformed_meshes):
        #     p.add_mesh(meshes[i], color="blue", label=f"mesh {i} pre transform")
        #     p.add_mesh(mesh, color="red", label=f"mesh {i} transformed")
        #
        # p.add_legend()
        # p.show()

        # All meshes should now be equal
        for mesh in transformed_meshes[1:]:
            assert np.array_equal(transformed_meshes[0].points, mesh.points)


def test_mesh_transform_sw_multi_domain():
    # Create some sample meshes
    left_l = pv.Box(quads=False).scale([0.5, 1, 1], inplace=False).translate([-4.5, 0, 0], inplace=False)
    left_r = pv.Box(quads=False).scale([0.5, 1, 1], inplace=False).translate([-3.5, 0, 0], inplace=False)

    center_l = pv.Box(quads=False).scale([0.5, 1, 1], inplace=False).translate([-0.5, 0, 0], inplace=False)
    center_r = pv.Box(quads=False).scale([0.5, 1, 1], inplace=False).translate([0.5, 0, 0], inplace=False)

    right_l = pv.Box(quads=False).scale([0.5, 1, 1], inplace=False).translate([2.5, 0, 0], inplace=False)
    right_r = pv.Box(quads=False).scale([0.5, 1, 1], inplace=False).translate([3.5, 0, 0], inplace=False)

    # Create SPARC data structure in temp files
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory(dir=root) as derivative, \
            tempfile.TemporaryDirectory(dir=root) as pool_derivative, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="left_") as left_dir, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="center_") as center_dir, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="right_") as right_dir:
        dataloc = DatasetLocator(Path(root), "", Path(derivative).relative_to(root), "")

        # Save meshes in directory structure and create configs
        left_l.save(str(dataloc.abs_derivative / Path(left_dir).stem / "mesh_l.vtk"))
        left_r.save(str(dataloc.abs_derivative / Path(left_dir).stem / "mesh_r.vtk"))
        center_l.save(str(dataloc.abs_derivative / Path(center_dir).stem / "mesh_l.vtk"))
        center_r.save(str(dataloc.abs_derivative / Path(center_dir).stem / "mesh_r.vtk"))
        right_l.save(str(dataloc.abs_derivative / Path(right_dir).stem / "mesh_l.vtk"))
        right_r.save(str(dataloc.abs_derivative / Path(right_dir).stem / "mesh_r.vtk"))

        meshes = [left_l, left_r, center_l, center_r, right_l, right_r]
        names = ["left_l", "left_r", "center_l", "center_r", "right_l", "right_r"]

        combined_center = center_l.merge(center_r)

        # Use center as the reference
        initialize_result = {"reference_meshes": {
            "combined_reference_mesh": {"points": np.array(combined_center.points),
                                        "faces": np.array(
                                            pyvista_faces_to_2d(combined_center.faces))},
            "mesh_l_reference_mesh": {"points": np.array(center_l.points),
                                      "faces": np.array(pyvista_faces_to_2d(center_l.faces))},
            "mesh_r_reference_mesh": {"points": np.array(center_r.points),
                                      "faces": np.array(pyvista_faces_to_2d(center_r.faces))}}}

        dirs_list = [(Path(left_dir).stem, "", ""), (Path(center_dir).stem, "", ""),
                     (Path(right_dir).stem, "", "")]

        output_directory = "."
        task_config = DictConfig({"source_directory": ".", "params": {"iterations": 10}})

        results = []
        for dir, _, _ in dirs_list:
            result = MeshTransformSW.work(source_directory_primary=dataloc.abs_primary / dir,
                                          source_directory_derivative=dataloc.abs_derivative / dir,
                                          output_directory=dataloc.abs_derivative / dir / output_directory,
                                          dataset_config=None,
                                          task_config=task_config,
                                          initialize_result=initialize_result)
            results.append(result)

        all_results = np.array(results).ravel()
        all_loaded_results = []
        for result in all_results:
            result = np.load(glob(f"{result}*")[0])
            all_loaded_results.append(result)

        individual_mesh_results = np.array(results)[:, 1:].ravel()
        individual_loaded_results = []
        for result in individual_mesh_results:
            result = np.load(glob(f"{result}*")[0])
            individual_loaded_results.append(result)

        transformed_meshes = []
        for result, mesh in zip(individual_loaded_results, meshes):
            transformed_meshes.append(mesh.transform(result, inplace=False))

        # p = pv.Plotter()
        # c = matplotlib.colormaps["hsv"]
        # for i, (mesh, name) in enumerate(zip(meshes, names)):
        #     p.add_mesh(mesh, color=c((i + 1) / (len(meshes))), label=name)
        #     p.add_mesh(transformed_meshes[i], label=f"{name} transformed")
        #
        # p.add_legend()
        # p.show()

        # print("")
        for mesh_l, mesh_r in zip(transformed_meshes[::2], transformed_meshes[1::2]):
            # print(np.array_equal(mesh_l.points, center_l.points))
            # print(np.array_equal(mesh_r.points, center_r.points))
            assert np.array_equal(mesh_l.points, center_l.points)
            assert np.array_equal(mesh_r.points, center_r.points)


def test_create_transform():
    # Test that the shapeworks createTransform function works as we expect
    mesh = pv.Box(quads=False).scale([0.5, 1, 1], inplace=False).translate([-4.5, 0, 0], inplace=False)
    reference_mesh = pv.Box(quads=False).scale([0.5, 1, 1], inplace=False).translate([-0.5, 0, 0], inplace=False)

    sw_mesh = sw.Mesh(mesh.points, pyvista_faces_to_2d(mesh.faces))
    sw_reference_mesh = sw.Mesh(reference_mesh.points, pyvista_faces_to_2d(reference_mesh.faces))

    transform = sw_mesh.createTransform(sw_reference_mesh, sw.Mesh.AlignmentType.Rigid, 10)

    pv_mesh = sw.sw2vtkMesh(sw_mesh)
    pv_reference_mesh = sw.sw2vtkMesh(sw_reference_mesh)

    transformed_pv_mesh = pv_mesh.transform(transform)

    assert np.array_equal(transformed_pv_mesh.points, pv_reference_mesh.points)
