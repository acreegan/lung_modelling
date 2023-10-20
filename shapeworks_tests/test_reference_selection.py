from pathlib import Path
from lung_modelling.app.shapeworks_tasks import ReferenceSelectionMeshSW
from lung_modelling.workflow_manager import DatasetLocator
import pyvista as pv
import tempfile
from omegaconf import DictConfig
import shapeworks as sw
import numpy as np

parent_dir = Path(__file__).parent


def test_reference_selection_mesh_sw_single_domain():
    # Create some pyvista meshes. Middle should be the medoid
    # Use quads=False if saving to vtk to save all the faces
    small = pv.Box(quads=False).scale([100, 100, 100], inplace=False)
    middle = pv.Box(quads=False).scale([101, 101, 101], inplace=False)
    large = pv.Box(quads=False).scale([102, 102, 102], inplace=False)

    # Create SPARC data structure in temp files
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory(dir=root) as derivative, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="small_") as small_dir, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="middle_") as middle_dir, \
            tempfile.TemporaryDirectory(dir=derivative, prefix="large_") as large_dir:
        dataloc = DatasetLocator(Path(root), "", Path(derivative).relative_to(root), "")

        # Save meshes in directory structure and create configs
        small.save(str(dataloc.abs_derivative / Path(small_dir).stem / "small.vtk"))
        middle.save(str(dataloc.abs_derivative / Path(middle_dir).stem / "middle.vtk"))
        large.save(str(dataloc.abs_derivative / Path(large_dir).stem / "large.vtk"))

        dirs_list = [(Path(small_dir).stem, "", ""), (Path(middle_dir).stem, "", ""),
                     (Path(large_dir).stem, "", "")]

        output_directory = Path(root)
        task_config = DictConfig({"source_directory": "."})

        # Run the actual task!
        result = ReferenceSelectionMeshSW.work(dataloc, dirs_list, output_directory, None, task_config)

        ref_mesh = pv.read(str(result[0]))

        # # Plot results
        # p = pv.Plotter()
        # p.add_mesh(small.extract_all_edges(), label="small", color="blue")
        # p.add_mesh(middle.extract_all_edges(), label="middle", color="green")
        # p.add_mesh(large.extract_all_edges(), label="large", color="yellow")
        # p.add_mesh(ref_mesh.extract_all_edges(), label=result[0].stem, color="red")
        # p.add_legend()
        # p.show()

        # Ref mesh should be middle
        assert np.array_equal(ref_mesh.points, middle.points)


def test_find_reference_mesh():
    # Snippet to check that find_reference_mesh works how we expect it to

    # Create some pyvista meshes. Middle should be the medoid
    small = pv.Box(quads=False).scale([100, 100, 100], inplace=False)
    middle = pv.Box(quads=False).scale([101, 101, 101], inplace=False)
    large = pv.Box(quads=False).scale([102, 102, 102], inplace=False)

    # Zip with names to keep track
    original_meshes = zip([small, middle, large], ["small", "middle", "large"])

    # Save to .vtk and read to convert to Shapeworks mesh
    sw_meshes = []
    for pv_mesh, name in original_meshes:
        with tempfile.TemporaryDirectory() as d:
            pv_mesh.save(f"{d}/{name}.stl")
            sw_meshes.append((sw.Mesh(f"{d}/{name}.stl"), name))

    # Try find_reference_mesh_index with different ordered mesh lists
    sw_meshes = np.array(sw_meshes)
    ref_mesh_names = []
    for i in [0, 1, 2]:
        test_array = np.roll(sw_meshes, i, axis=0)
        ref_index = sw.find_reference_mesh_index(test_array[:, 0], domains_per_shape=1)
        ref_mesh_names.append(test_array[ref_index, 1])
        # print(f"Ref mesh {i}: {test_array[ref_index, 1]}")

    # # Plot
    # pv_mesh_list = [sw.sw2vtkMesh(mesh) for mesh in sw_meshes[:, 0]]
    # p = pv.Plotter()
    # p.add_mesh(pv_mesh_list[0].extract_all_edges(), label="0", color="blue")
    # p.add_mesh(pv_mesh_list[1].extract_all_edges(), label="1", color="green")
    # p.add_mesh(pv_mesh_list[2].extract_all_edges(), label="2", color="yellow")
    # p.add_legend()
    # p.show_bounds()
    # p.show()

    # Reference mesh should be center. No matter what order we test in.
    assert np.array_equal(np.array(ref_mesh_names), np.array(["middle", "middle", "middle"]))


# Todo
# def test_reference_selection_mesh_sw_multi_domain():
#     assert False
