from lung_modelling.workflow_manager import EachItemTask, DatasetLocator, AllItemsTask
from pathlib import Path
from omegaconf import DictConfig
import os
import shapeworks as sw
from glob import glob
import pyvista as pv
import numpy as np
import subprocess
from pyvista_tools import pyvista_faces_to_2d, remove_shared_faces_with_merge
import csv
from lung_modelling import find_connected_faces, flatten


class SmoothLungLobesSW(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @property
    def name(self):
        return "smooth_lung_lobes_sw"

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Pre-process lung lobe images by applying antialiasing using Shapeworks libraries

        Parameters
        ----------
        source_directory_primary
            Absolute path of the source directory in the primary folder of the dataset
        source_directory_derivative
            Absolute path of the source directory in the derivative folder of the dataset
        output_directory
            Directory in which to save results of the work
        dataset_config
            Config relating to the entire dataset
        task_config
            **results_directory**: subdirectory for results

            **output_filenames**: dict providing a mapping from lobe mapping (in dataset config) to output filenames

            **params**: (Dict)
                **maximumRMSError**, **numberOfIterations**:
                    Parameters to apply to antialias

        Returns
        -------
        smoothed_lobes
            list of Path objects representing the files created.
        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file = glob(str(source_directory_primary / dataset_config.lung_image_glob))[0]
        shape_seg = sw.Image(image_file)

        suffix = Path(image_file).suffix

        smoothed_lobes = []
        for lobe, name in task_config.output_filenames.items():
            s = shape_seg.copy()  # Copy because extractLabel is destructive
            lobe_image = s.extractLabel(dataset_config.lobe_mapping[lobe])
            iso_spacing = [1, 1, 1]
            lobe_image.antialias(task_config.params.numberOfIterations, task_config.params.maximumRMSError).resample(
                iso_spacing, sw.InterpolationType.Linear).binarize().isolate()

            filename = f"{str(output_directory / name)}{suffix}"
            lobe_image.write(filename)
            smoothed_lobes.append(Path(filename))

        return smoothed_lobes


class CreateMeshesSW(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @property
    def name(self):
        return "create_meshes_sw"

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Convert medical image files to meshes and apply smoothing using Shapeworks libraries.

        To meet the shapeworks requirement that all groomed files have unique names, the top level parent folder (which
        should be the subject number) is appended to the file names, delimited by a dash character (-).

        Parameters
        ----------
        source_directory_primary
            Absolute path of the source directory in the primary folder of the dataset
        source_directory_derivative
            Absolute path of the source directory in the derivative folder of the dataset
        output_directory
            Directory in which to save results of the work
        dataset_config
            Config relating to the entire dataset
        task_config
            **source_directory**: subdirectory within derivative source folder to find source files

            **results_directory**: subdirectory for results

            **params**: (Dict)
                **decimate**, **target_reduction**, **volume_preservation**
                    Option to decimate and parameters for pyvvista mesh decimate
                **remesh**, **remesh_percentage**, **adaptivity**:
                    Option to remesh and parameters for shapeworks remesh
                **smooth**, **smooth_iterations**, **relaxation**:
                    Option to smooth and parameters for shapeworks smooth
                **fill_holes**, **hole_size**:
                    Option to fill holes andd parameters for shapeworks fill_holes
                **remove_shared_faces**:
                    Option to remove duplicate faces in the mesh
                **isolate_mesh**:
                    Option to remove islands leaving only the larges connected region in the mesh.
                    If use_geodesic_distance is selected in the optimizer, it is essential that the mesh does not
                    have islands.

        Returns
        -------
        mesh_files
            list of Path objects representing the files created.
        """
        params = task_config.params

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_files = glob(str(source_directory_derivative / task_config.source_directory / "*"))

        if len(image_files) == 0:
            raise RuntimeError("No files found")

        mesh_files = []
        for image_file in image_files:
            image_data = sw.Image(image_file).pad(10)
            mesh = image_data.toMesh(1)  # Seems to get the mesh in the right orientation without manual realignment

            if params.decimate:
                mesh = sw.sw2vtkMesh(mesh)
                mesh = mesh.decimate(target_reduction=params.target_reduction,
                                     volume_preservation=params.volume_preservation)
                mesh = sw.Mesh(mesh.points, pyvista_faces_to_2d(mesh.faces))

            if params.remesh:
                # Shapeworks remeshing uses ACVD (vtkIsotropicDiscreteRemeshing). Should be the same as pyACVD with pyvista
                mesh = mesh.remeshPercent(percentage=params.remesh_percentage,
                                          adaptivity=params.adaptivity)

            if params.smooth:
                # Shapeworks smooth uses vtkSmoothPolyData. Should be the same as pyVista
                mesh = mesh.smooth(iterations=params.smooth_iterations,
                                   relaxation=params.relaxation)

            if params.fill_holes:
                # Shapeworks fillHoles uses vtkFillHolesFilter. Should be the same as pyVista
                mesh = mesh.fillHoles(hole_size=params.hole_size)

            if params.remove_shared_faces:
                mesh = sw.sw2vtkMesh(mesh) if isinstance(mesh, sw.Mesh) else mesh
                mesh = remove_shared_faces_with_merge([mesh])
                if mesh.n_faces == 0:
                    raise ValueError("Generated mesh is empty")
            else:
                mesh = sw.sw2vtkMesh(mesh) if isinstance(mesh, sw.Mesh) else mesh

            if params.isolate_mesh:
                _, connected_points = find_connected_faces(list(pyvista_faces_to_2d(mesh.faces)))
                if len(connected_points) > 1:
                    connected_points = [list(set(item)) for item in list(connected_points.values())]
                    connected_points.sort(key=len, reverse=True)
                    to_remove = flatten(connected_points[1:])
                    mesh, _ = mesh.remove_points(to_remove)

            output_filename = f"{output_directory / Path(image_file).stem}-{output_directory.parents[0].stem}.vtk"
            mesh.save(output_filename)

            mesh_files.append(Path(output_filename))

        return mesh_files


class SmoothWholeLungsSW(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @property
    def name(self):
        return "smooth_whole_lungs_sw"

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Pre-process lung segmentation by extracting whole lung and applying antialiasing using Shapeworks libraries.

        Parameters
        ----------
        source_directory_primary
            Absolute path of the source directory in the primary folder of the dataset
        source_directory_derivative
            Absolute path of the source directory in the derivative folder of the dataset
        output_directory
            Directory in which to save results of the work
        dataset_config
            Config relating to the entire dataset
        task_config
            **results_directory**: subdirectory for results

            **output_filenames**: dict providing a mapping from lobe mapping (in dataset config) to output filenames

            **params**: (Dict)
                **maximumRMSError**, **numberOfIterations**:
                    Parameters to apply to antialias

        Returns
        -------
        smoothed_lobes
            list of Path objects representing the files created.
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file = glob(str(source_directory_primary / dataset_config.lung_image_glob))[0]
        shape_seg = sw.Image(image_file)

        suffix = Path(image_file).suffix

        smoothed_files = []
        for name, lobes in task_config.output_filenames.items():

            lobe_images = []
            for lobe in lobes:
                s = shape_seg.copy()
                image = s.extractLabel(dataset_config.lobe_mapping[lobe])
                lobe_images.append(image)

            merged = lobe_images[0].copy()
            for image in lobe_images[1:]:
                merged = merged + image

            iso_spacing = [1, 1, 1]
            merged.antialias(task_config.params.numberOfIterations, task_config.params.maximumRMSError).resample(
                iso_spacing, sw.InterpolationType.Linear).binarize().isolate()

            filename = f"{str(output_directory / name)}{suffix}"
            merged.write(filename)
            smoothed_files.append(Path(filename))

        return smoothed_files


class ReferenceSelectionMeshSW(AllItemsTask):
    @property
    def name(self):
        return "reference_selection_mesh_sw"

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Load all meshes at once so the shape closest to the mean can be found and selected as the reference

        The subject that was used as the reference is indicated in the output filename, surrounded by ()

        Parameters
        ----------
        dataloc
            Dataset locator for the dataset
        dirs_list
            List of relative paths to the source directories
        output_directory
            Directory in which to save results of the work
        dataset_config
            Config relating to the entire dataset
        task_config
            **source_directory**: subdirectory within derivative source folder to find source files

            **results_directory**: subdirectory for results


        Returns
        -------
        List of reference mesh filenames. The first element is the combined reference mesh, and the following elements
        are the domain reference meshes.

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        all_mesh_files = []
        for dir, _, _ in dirs_list:
            mesh_files = glob(str(dataloc.abs_derivative / dir / task_config.source_directory / "*"))
            all_mesh_files.append(mesh_files)

        lengths = [len(files) for files in all_mesh_files]
        if len(set(lengths)) != 1:
            raise ValueError(f"Not all source directories have the same number of meshes")

        domains_per_shape = lengths[0]

        all_meshes = [sw.Mesh(file) for file in np.array(all_mesh_files).ravel()]

        domain_reference_filenames = []
        if domains_per_shape == 1:
            ref_index = sw.find_reference_mesh_index(all_meshes, domains_per_shape)
            ref_mesh_combined = all_meshes[ref_index]
            ref_dir = Path(dirs_list[ref_index][0]).stem

        else:
            ref_index, combined_meshes = sw.find_reference_mesh_index(all_meshes, domains_per_shape)
            ref_mesh_combined = combined_meshes[ref_index]
            ref_dir = Path(dirs_list[ref_index][0]).stem

            for i in range(domains_per_shape):
                # find_reference_mesh_index destroys all_meshes during the combining process. So we have to create them again
                all_meshes = [sw.Mesh(file) for file in np.array(all_mesh_files).ravel()]
                domain_reference_mesh = all_meshes[ref_index * domains_per_shape + i]
                domain_reference_name = str(
                    Path(np.array(all_mesh_files).ravel()[ref_index * domains_per_shape + i]).stem).split("-")[0]

                domain_reference_filename = output_directory / f"{domain_reference_name}_reference_mesh.vtk"
                domain_reference_mesh.write(str(domain_reference_filename))
                domain_reference_filenames.append(domain_reference_filename)

        ref_mesh_combined_filename = output_directory / f"combined_reference_mesh.vtk"
        ref_mesh_combined.write(str(ref_mesh_combined_filename))

        ref_dir_filename = output_directory / "ref_dir.txt"
        with open(str(ref_dir_filename), "w") as ref_dir_file:
            writer = csv.writer(ref_dir_file)
            writer.writerow([str(ref_dir)])

        return [ref_mesh_combined_filename, *domain_reference_filenames]


class MeshTransformSW(EachItemTask):

    @property
    def name(self):
        return "mesh_transform_sw"

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        """
        Load reference meshes and convert to points and faces so they can be pickled and sent to the work function

        Parameters
        ----------
        dataloc
            Dataset Locator
        dataset_config
            Dataset config
        task_config
            **source_directory_initialize**
                directory from which to load reference meshes

        Returns
        -------
        Dict of reference meshes
            {**reference_meshes**:{**points**:[points], **faces**:[faces]}}

        """
        reference_mesh_files = glob(
            str(dataloc.abs_pooled_derivative / task_config.source_directory_initialize / "*.vtk"))

        reference_meshes = {}
        for file in reference_mesh_files:
            reference_mesh = pv.read(file)
            reference_meshes[str(Path(file).stem)] = {"points": np.array(reference_mesh.points),
                                                      "faces": np.array(pyvista_faces_to_2d(reference_mesh.faces))}

        return {"reference_meshes": reference_meshes}

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Calculate alignment transforms for global alignment and per domain alignment. Uses shapeworks rigid alignment
        for both.

        Parameters
        ----------
        source_directory_primary
            Absolute path of the source directory in the primary folder of the dataset
        source_directory_derivative
            Absolute path of the source directory in the derivative folder of the dataset
        output_directory
            Directory in which to save results of the work
        dataset_config
            Config relating to the entire dataset
        task_config
            **results_directory**: subdirectory for results

            **output_filenames**: dict providing a mapping from lobe mapping (in dataset config) to output filenames

            **params**: (Dict)
                **iterations**
                    Iterations for shapeworks alignment function

        initialize_result
            Return dict from the initialize function

        Returns
        -------
        List of transform filenames. The first is for the global alignment, and the following are for domain alignments.

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        mesh_files = glob(str(source_directory_derivative / task_config.source_directory / "*"))

        if len(mesh_files) == 0:
            raise RuntimeError("No files found")

        meshes = []
        for mesh_file in mesh_files:
            meshes.append(sw.Mesh(mesh_file))

        # Copy so we don't destroy meshes[0] for later!
        combined_mesh = meshes[0].copy()
        for mesh in meshes[1:]:
            combined_mesh += mesh

        combined_reference_mesh = sw.Mesh(initialize_result["reference_meshes"]["combined_reference_mesh"]["points"],
                                          initialize_result["reference_meshes"]["combined_reference_mesh"]["faces"])
        combined_transform = combined_mesh.createTransform(combined_reference_mesh, sw.Mesh.AlignmentType.Rigid,
                                                           task_config.params.iterations)

        # # Center method:
        # com1 = combined_reference_mesh.centerOfMass()
        # com2 = combined_mesh.centerOfMass()
        # centered_mesh = combined_mesh.copy().translate(
        #     [-(com2[0] - com1[0]), -(com2[1] - com1[1]), -(com2[2] - com1[2])])
        #
        # combined_transform = combined_mesh.createTransform(centered_mesh, sw.Mesh.AlignmentType.Rigid,
        #                                                    1).flatten()

        combined_transform_filename = output_directory / "combined_mesh_transform.txt"
        np.savetxt(str(combined_transform_filename), combined_transform)

        domain_transform_filenames = []
        if len(meshes) > 1:
            for mesh, file in zip(meshes, mesh_files):
                domain_reference_mesh = sw.Mesh(
                    initialize_result["reference_meshes"][f"{str(Path(file).stem).split('-')[0]}_reference_mesh"][
                        "points"],
                    initialize_result["reference_meshes"][f"{str(Path(file).stem).split('-')[0]}_reference_mesh"][
                        "faces"])
                domain_transform = mesh.createTransform(domain_reference_mesh, sw.Mesh.AlignmentType.Rigid,
                                                        task_config.params.iterations)

                # # Center method:
                # com = mesh.centerOfMass()
                # centered_mesh = mesh.copy().translate([-com[0], -com[1], -com[2]])
                # domain_transform = mesh.createTransform(centered_mesh, sw.Mesh.AlignmentType.Rigid, 1)

                domain_transform_filename = output_directory / f"{str(Path(file).stem)}_transform.txt"
                np.savetxt(str(domain_transform_filename), domain_transform)
                domain_transform_filenames.append(domain_transform_filename)

        return [combined_transform_filename, *domain_transform_filenames]


class OptimizeMeshesSW(AllItemsTask):

    @property
    def name(self):
        return "optimize_meshes_sw"

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Run the shapeworks optimize command

        Parameters
        ----------
        dataloc
            Dataset Locator
        dirs_list
            List of relative paths to the source directories
        output_directory
            Directory in which to save results of the work
        dataset_config
            Dataset config
        task_config
            **source_directory_transform**
                directory for transform files
            **source_directory_mesh**
                directory for mesh files
            **source_directory_original**
                directory for original (pre-grooming) files
            **results_directory**:
                subdirectory for results
            **params**
                shapeworks optimization params

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        subjects = []
        for dir, _, _ in dirs_list:
            subject = sw.Subject()
            mesh_files = glob(str(dataloc.abs_derivative / dir / task_config.source_directory_mesh / "*"))
            original_files = glob(str(dataloc.abs_derivative / dir / task_config.source_directory_original / "*"))
            subject.set_number_of_domains(len(mesh_files))
            transforms = []
            for file in mesh_files:
                glob_string = str(
                    dataloc.abs_derivative / dir / task_config.source_directory_transform / f"*{Path(file).stem}*")
                transform_file = glob(glob_string)[0]
                transform = np.loadtxt(transform_file).flatten()
                transforms.append(transform)

            combined_transform_file = \
                glob(str(dataloc.abs_derivative / dir / task_config.source_directory_transform / "*combined*"))[0]
            combined_transform = np.loadtxt(combined_transform_file).flatten()
            transforms.append(combined_transform)
            subject.set_groomed_transforms(transforms)
            subject.set_groomed_filenames(mesh_files)
            subject.set_original_filenames(original_files)
            subject.set_display_name(str(Path(dir).stem))
            subjects.append(subject)

        project = sw.Project()
        project.set_subjects(subjects)
        parameters = sw.Parameters()
        for key, value in task_config.params.items():
            parameters.set(key, sw.Variant(value))

        project.set_parameters("optimize", parameters)

        spreadsheet_file = output_directory / "shapeworks_project.swproj"
        project.save(str(spreadsheet_file))

        wd = os.getcwd()
        os.chdir(output_directory)
        optimize_cmd = ("shapeworks optimize --progress --name " + str(spreadsheet_file)).split()
        subprocess.check_call(optimize_cmd)
        os.chdir(wd)


all_tasks = [SmoothLungLobesSW, CreateMeshesSW, SmoothWholeLungsSW, ReferenceSelectionMeshSW, MeshTransformSW,
             OptimizeMeshesSW]
