from pathlib import Path
from lung_modelling.app.shapeworks_tasks import ReferenceSelectionMeshSW, MeshTransformSW
from lung_modelling.workflow_manager import DatasetLocator
import pyvista as pv
import tempfile
from omegaconf import DictConfig
import shapeworks as sw
import numpy as np
from pyvista_tools import pyvista_faces_to_2d, pyvista_faces_to_1d
from glob import glob

parent_dir = Path(__file__).parent


def test_reference_selection_mesh_sw_single_domain():
    # Create some sample meshes
    left = pv.Box().translate([-4, 0, 0], inplace=False)
    center = pv.Box()
    right = pv.Box().translate([3, 0, 0], inplace=False)

    # Create SPARC data structure in temp files
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory(dir=root) as derivative, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="left_") as left_dir, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="center_") as center_dir, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="right_") as right_dir:
        dataloc = DatasetLocator(Path(root), "", Path(derivative).relative_to(root), "")

        # Save meshes in directory structure and create configs
        left.save(str(dataloc.abs_derivative / Path(left_dir).stem / "left.vtk"))
        center.save(str(dataloc.abs_derivative / Path(center_dir).stem / "center.vtk"))
        right.save(str(dataloc.abs_derivative / Path(right_dir).stem / "right.vtk"))

        dirs_list = [(Path(left_dir).stem, "", ""), (Path(center_dir).stem, "", ""),
                     (Path(right_dir).stem, "", "")]

        output_directory = Path(root)
        task_config = DictConfig({"source_directory": "."})

        # Run the actual task!
        result = ReferenceSelectionMeshSW.work(dataloc, dirs_list, output_directory, None, task_config)

        ref_mesh = pv.read(str(result[0]))

        # # Plot results
        # p = pv.Plotter()
        # p.add_mesh(left, label="left", color="blue")
        # p.add_mesh(center, label="center", color="green")
        # p.add_mesh(right, label="right", color="yellow")
        # p.add_mesh(ref_mesh, label=result[0].stem, color="red")
        # p.add_legend()
        # p.show_bounds(bounds=[-5, 5, -1, 1, -1, 1])
        # p.show()

        # Ref mesh shoudl be center
        assert np.array_equal(ref_mesh.points, center.points)


def test_find_reference_mesh():
    # Snippet to check that find_reference_mesh works how we expect it to

    # Create some pyvista meshes. Center should be the medoid
    left = pv.Box().translate([-4, -4, -4], inplace=False)
    center = pv.Box().translate([1, 1, 1], inplace=False)
    right = pv.Box().translate([3, 3, 3], inplace=False)

    # Save to .vtk and read to convert to Shapeworks mesh
    sw_meshes = []
    for pv_mesh in [left, center, right]:
        with tempfile.TemporaryDirectory() as d:
            pv_mesh.save(f"{d}/temp_mesh.vtk")
            sw_meshes.append(sw.Mesh(f"{d}/temp_mesh.vtk"))

    # Demonstrate that the Shapeworks meshes are loaded with the correct center of masses
    print("\n")
    for i, mesh in enumerate(sw_meshes):
        print(f"Mesh {i} center of mass: {mesh.centerOfMass()}")

    # Try find_reference_mesh_index with different ordered mesh lists
    ref_index_1 = sw.find_reference_mesh_index(sw_meshes, domains_per_shape=1)
    ref_index_2 = sw.find_reference_mesh_index([sw_meshes[1], sw_meshes[2], sw_meshes[0]], domains_per_shape=1)
    ref_index_3 = sw.find_reference_mesh_index([sw_meshes[2], sw_meshes[0], sw_meshes[1]], domains_per_shape=1)

    # Print what we found. We expect the indices to be 1, 0, 2
    for i, ref_index in enumerate([ref_index_1, ref_index_2, ref_index_3]):
        print(f"Ref index {i}: {ref_index}")


    # Plot
    # pv_mesh_list = [sw.sw2vtkMesh(mesh) for mesh in sw_meshes]
    # p = pv.Plotter()
    # p.add_mesh(pv_mesh_list[0], label="0", color="blue")
    # p.add_mesh(pv_mesh_list[1], label="1", color="green")
    # p.add_mesh(pv_mesh_list[2], label="2", color="yellow")
    # p.add_legend()
    # p.show_bounds()
    # p.show()

    # Reference mesh should be center. No matter what order we test in.
    assert np.array_equal(np.array([ref_index_1, ref_index_2, ref_index_3]), np.array([1, 0, 2]))

    pass


def test_find_reference_mesh_1():
    test_data_dir = Path("test_data")
    mesh1 = sw.Mesh(str(Path(__file__).parent / test_data_dir / "m03_L_femur.ply"))
    mesh2 = sw.Mesh(str(Path(__file__).parent / test_data_dir / "m04_L_femur.ply"))
    mesh3 = sw.Mesh(str(Path(__file__).parent / test_data_dir / "m03.vtk"))

    meshList = []
    meshList.append(mesh1)
    meshList.append(mesh2)
    meshList.append(mesh3)

    ref_index = sw.find_reference_mesh_index(meshList, domains_per_shape=1)

    # pv_mesh_list = [sw.sw2vtkMesh(mesh) for mesh in meshList]
    #
    # p = pv.Plotter()
    # p.add_mesh(pv_mesh_list[0], label="0", color="blue")
    # p.add_mesh(pv_mesh_list[1], label="1", color="green")
    # p.add_mesh(pv_mesh_list[2], label="2", color="yellow")
    # p.add_mesh(pv_mesh_list[ref_index], label="ref_mesh", color="red")
    # p.add_legend()
    # p.show_bounds()
    # p.show()

    assert ref_index == 2


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
