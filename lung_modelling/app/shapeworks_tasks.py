import itertools

from lung_modelling.workflow_manager import EachItemTask, DatasetLocator, AllItemsTask
from pathlib import Path, PurePosixPath
from omegaconf import DictConfig
import os
import shapeworks as sw
from glob import glob
import pyvista as pv
import numpy as np
import subprocess
from pyvista_tools import pyvista_faces_to_2d, remove_shared_faces_with_merge
import csv
from lung_modelling import find_connected_faces, flatten, voxel_to_mesh, fix_mesh, mesh_rms_error, load_with_category
from lung_modelling.shapeworks_libs import PCA_Embbeder
import medpy.io
import pandas as pd
from DataAugmentationUtils import Utils, Embedder
from fnmatch import fnmatch
import re
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
from scipy import stats
import pickle
from sklearn.feature_selection import RFECV
from loguru import logger
from matplotlib import colormaps
import statsmodels.api as sm


class ExtractLungLobesSW(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        """

        Parameters
        ----------
        dataloc
        dataset_config
        task_config

        Returns
        -------

        """
        pass

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
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Config relating to the entire dataset
        task_config
            **results_directory**: Name of the results folder (Stem of output_directory)

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
        """

        Parameters
        ----------
        dataloc
        dataset_config
        task_config

        Returns
        -------

        """
        pass

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
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Config relating to the entire dataset
        task_config
            **source_directory**: subdirectory within derivative source folder to find source files

            **results_directory**: Name of the results folder (Stem of output_directory)

            **params**: (Dict)
                **step_size**
                    Step size to use for marching cubes. Higher values result in coarser geometry but can prevent meshes
                    from taking up too much RAM
                **decimate**, **decimate_target_faces**, **volume_preservation**, **subdivide_passes**
                    Option to decimate and parameters for pyvvista mesh decimate. If subdivide_passes is greater than 0,
                    the mesh is decimated first then subdivided. The initial decimation is calcuated such that the
                    result after subdivision is the target number of faces
                **remesh**, **remesh_target_points**, **adaptivity**:
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

        image_files = glob(str(source_directory_derivative / task_config.source_directory / task_config.image_glob))

        if len(image_files) == 0:
            raise RuntimeError("No files found")

        mesh_files = []
        for image_file in image_files:
            image_data, header = medpy.io.load(image_file)
            if params.pad:
                image_data = np.pad(image_data, 5)
            mesh = voxel_to_mesh(image_data, spacing=header.spacing, direction=header.direction, offset=header.offset,
                                 step_size=params.step_size)

            if params.fix_first:
                mesh = mesh.clean()
                mesh = fix_mesh(mesh)

            mesh = sw.Mesh(mesh.points, pyvista_faces_to_2d(mesh.faces))

            if params.decimate:
                mesh = sw.sw2vtkMesh(mesh)
                mesh = mesh.clean()

                t_faces = params.decimate_target_faces / (4 ** params.subdivide_passes)
                target_reduction = 1 - (t_faces / mesh.n_faces_strict)
                mesh = mesh.decimate(target_reduction=target_reduction,
                                     volume_preservation=params.volume_preservation)

                if params.subdivide_passes > 0:
                    mesh = fix_mesh(mesh)
                    mesh = mesh.subdivide(params.subdivide_passes)

                mesh = sw.Mesh(mesh.points, pyvista_faces_to_2d(mesh.faces))

            if params.remesh:
                # Shapeworks remeshing uses ACVD (vtkIsotropicDiscreteRemeshing). Should be the same as pyACVD with pyvista
                mesh = mesh.remesh(numVertices=params.remesh_target_points,
                                   adaptivity=params.adaptivity)

            if params.smooth:
                # Shapeworks smooth uses vtkSmoothPolyData. Should be the same as pyVista
                mesh = mesh.smooth(iterations=params.smooth_iterations,
                                   relaxation=params.relaxation)

            if params.fill_holes:
                # Shapeworks fillHoles uses vtkFillHolesFilter. Should be the same as pyVista
                mesh = mesh.fillHoles(hole_size=params.hole_size)

            if params.remove_shared_faces:
                pv_mesh = sw.sw2vtkMesh(mesh) if isinstance(mesh, sw.Mesh) else mesh
                pv_mesh = remove_shared_faces_with_merge([pv_mesh])
                if pv_mesh.n_faces_strict == 0:
                    raise ValueError("Generated mesh is empty")
            else:
                pv_mesh = sw.sw2vtkMesh(mesh) if isinstance(mesh, sw.Mesh) else mesh

            # Todo: Pyvista already has a function for this called connectivity
            if params.isolate_mesh:
                _, connected_points = find_connected_faces(list(pyvista_faces_to_2d(pv_mesh.faces)), return_points=True)
                if len(connected_points) > 1:
                    connected_points = [list(set(item)) for item in list(connected_points.values())]
                    connected_points.sort(key=len, reverse=True)
                    to_remove = flatten(connected_points[1:])
                    pv_mesh, _ = pv_mesh.remove_points(to_remove)

            output_filename = f"{output_directory / Path(image_file).stem}-{output_directory.parents[0].stem}.vtk"
            pv_mesh.save(output_filename)

            mesh_files.append(Path(output_filename))

        return mesh_files


class ExtractWholeLungsSW(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        """

        Parameters
        ----------
        dataloc
        dataset_config
        task_config

        Returns
        -------

        """
        pass

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
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Config relating to the entire dataset
        task_config
            **results_directory**: Name of the results folder (Stem of output_directory)

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
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Config relating to the entire dataset
        task_config
            **source_directories**: subdirectories within derivative source folder to find source files

            **results_directory**: Name of the results folder (Stem of output_directory)


        Returns
        -------
        List of reference mesh filenames. The first element is the combined reference mesh, and the following elements
        are the domain reference meshes.

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        all_mesh_files = []
        for dir, _, _ in dirs_list:
            mesh_files = []
            # Combine files from all sub source directories
            for source_directory in task_config.source_directories:
                mesh_files.extend(glob(str(dataloc.abs_derivative / dir / source_directory / "*")))
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

                domain_reference_filename = output_directory / f"{domain_reference_name}-reference_mesh.vtk"
                domain_reference_mesh.write(str(domain_reference_filename))
                domain_reference_filenames.append(domain_reference_filename)

        ref_mesh_combined_filename = output_directory / f"combined-reference_mesh.vtk"
        ref_mesh_combined.write(str(ref_mesh_combined_filename))

        ref_dir_filename = output_directory / "ref_dir.txt"
        with open(str(ref_dir_filename), "w") as ref_dir_file:
            writer = csv.writer(ref_dir_file)
            writer.writerow([str(dirs_list[ref_index][0])])

        return [ref_mesh_combined_filename, *domain_reference_filenames]


class MeshTransformSW(EachItemTask):

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
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Config relating to the entire dataset
        task_config
            **source_directories**: subdirectories within derivative source folder to find source files

            **results_directory**: Name of the results folder (Stem of output_directory)

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

        mesh_files = []
        for source_directory in task_config.source_directories:
            mesh_files.extend(glob(str(source_directory_derivative / source_directory / "*")))

        if len(mesh_files) == 0:
            raise RuntimeError("No files found")

        meshes = []
        for mesh_file in mesh_files:
            meshes.append(sw.Mesh(mesh_file))

        # Copy so we don't destroy meshes[0] for later!
        combined_mesh = meshes[0].copy()
        for mesh in meshes[1:]:
            combined_mesh += mesh

        combined_reference_mesh = sw.Mesh(initialize_result["reference_meshes"]["combined-reference_mesh"]["points"],
                                          initialize_result["reference_meshes"]["combined-reference_mesh"]["faces"])
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
                    initialize_result["reference_meshes"][f"{str(Path(file).stem).split('-')[0]}-reference_mesh"][
                        "points"],
                    initialize_result["reference_meshes"][f"{str(Path(file).stem).split('-')[0]}-reference_mesh"][
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

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Run the shapeworks optimize command

        NOTE! Path to project file cannot have spaces

        Todo: This should return the files it creates (at least the shapeworks project...)

        Parameters
        ----------
        dataloc
            Dataset Locator
        dirs_list
            List of relative paths to the source directories
        output_directory
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Dataset config
        task_config
            **source_directory_transform**
                directory for transform files
            **source_directories_mesh**
                directories for mesh files
            **source_directories_original**
                directories for original (pre-grooming) files
            **source_directory_subject_data**
                optional source directory to add subject groups to shapeworks project
            **image_globs**
                glob to find original files
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **params**
                shapeworks optimization params

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if "source_directory_subject_data" in task_config:
            group_data_file = \
                glob(str(dataloc.abs_pooled_derivative / task_config.source_directory_subject_data / "*"))[0]
            group_data = pd.read_csv(group_data_file)

        subjects = []
        for dir, _, _ in dirs_list:
            subject = sw.Subject()
            mesh_files = []
            for source_directory_mesh in task_config.source_directories_mesh:
                abs_mesh_files = glob(str(dataloc.abs_derivative / dir / source_directory_mesh / "*"))
                mesh_files.extend([os.path.relpath(file, output_directory) for file in abs_mesh_files])

            original_files = []
            for source_directory_original in task_config.source_directories_original:
                for image_glob in task_config.image_globs:
                    abs_original_files = glob(
                        str(dataloc.abs_derivative / dir / source_directory_original / image_glob))
                    original_files.extend([os.path.relpath(file, output_directory) for file in abs_original_files])

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

            landmark_files = []
            for source_directory_landmarks in task_config.source_directories_landmarks:
                abs_landmark_files = glob(str(dataloc.abs_derivative / dir / source_directory_landmarks / "*"))
                landmark_files.extend([os.path.relpath(file, output_directory) for file in abs_landmark_files])

            if "source_directory_subject_data" in task_config:
                subject_dir = str(PurePosixPath(dir))
                subject_group_data = group_data.loc[group_data.dir == subject_dir]
                subject.set_group_values(
                    {column: subject_group_data[column].values[0] for column in subject_group_data.columns[1:]})
            subject.set_landmarks_filenames(landmark_files)
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

        # Set studio parameters
        studio_dictionary = {
            "tool_state": "analysis"
        }
        studio_parameters = sw.Parameters()
        for key in studio_dictionary:
            studio_parameters.set(key, sw.Variant(studio_dictionary[key]))
        project.set_parameters("studio", studio_parameters)

        spreadsheet_file = output_directory / f"{output_directory.stem}_shapeworks_project.swproj"
        project.save(str(spreadsheet_file))

        wd = os.getcwd()
        os.chdir(output_directory)
        optimize_cmd = ["shapeworks", "optimize", "--progress", "--name", str(spreadsheet_file)]
        subprocess.check_call(optimize_cmd)
        os.chdir(wd)


class ComputePCASW(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Generate PCA model from the optimized shapeworks particle system. The complete PCA model is written to the
        results directory (Eigenvalues, eigenvectors, mean coordinates, and subject scores). The mean mesh is also
        computed by warping a subject mesh to the PCA mean points.

        Parameters
        ----------
        dataloc
            Dataset Locator
        dirs_list
            List of relative paths to the source directories
        output_directory
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Dataset config
        task_config
            **source_directory**
                source directory of the shapeworks particle system
            **source_directory_reference**
                source directory for the reference shape
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **mesh_file_domain_name_regex**:
                Regex pattern to match domain name within mesh file name
            **params**
                **warp_mesh_subject_id**
                    Index of subject mesh to use as the warp mesh

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        project_directory = dataloc.abs_pooled_derivative / task_config.source_directory
        shapeworks_project_file = glob(str(project_directory / "*.swproj"))[0]
        ref_dir_file = glob(str(dataloc.abs_pooled_derivative / task_config.source_directory_reference / "*.txt"))[0]
        ref_dir = Path(str(np.loadtxt(ref_dir_file, dtype=str)))

        project = sw.Project()
        project.load(shapeworks_project_file)
        subjects = project.get_subjects()

        # Apply transform to points and reference mesh
        # --------------------------------------------------------------------------------------------------------------
        # Get the local points and global transforms
        all_points = []
        subject_transforms = []
        ref_subject_index = None
        for i, subject in enumerate(subjects):
            # Global transform is the last in the list of subject transforms
            subject_global_transform = np.array(subject.get_groomed_transforms()[-1]).reshape(4, 4)
            subject_transforms.append(subject_global_transform)

            local_point_files = subject.get_local_particle_filenames()
            subject_points = []
            for file in local_point_files:
                domain_points = np.loadtxt(project_directory / file)  # File paths stored relative to project directory
                subject_points.append(domain_points)

            all_points.append(subject_points)

            # Find reference subject by comparing against directory we stored earlier
            if fnmatch(Path(subject.get_groomed_filenames()[0]).parents[1], f"*{ref_dir}"):
                ref_subject_index = i

        # Check the number of points in each domain
        domain_n_points = [len(p) for p in all_points[0]]

        # Merge points for each subject and apply transform
        all_points_transformed = []
        for subject_points, transform in zip(all_points, subject_transforms):
            subject_points = pv.PointSet(list(itertools.chain.from_iterable(subject_points)))
            subject_points_transformed = subject_points.transform(transform).points
            all_points_transformed.append(subject_points_transformed)

        all_points_transformed = np.array(all_points_transformed)

        if "warp_mesh_subject_id" in task_config.params:
            ref_subject_index = task_config.params["warp_mesh_subject_id"]

        # Load reference meshes and apply transform
        ref_meshes_transformed = []
        for relative_filename in subjects[ref_subject_index].get_groomed_filenames():
            # File paths stored relative to project directory
            ref_mesh_filename = os.path.normpath(os.path.join(project_directory, relative_filename))
            ref_mesh = sw.Mesh(ref_mesh_filename)
            ref_mesh_transformed = ref_mesh.applyTransform(subject_transforms[ref_subject_index])
            ref_meshes_transformed.append(ref_mesh_transformed)

        # Do PCA
        # --------------------------------------------------------------------------------------------------------------
        point_embedder = PCA_Embbeder(all_points_transformed, percent_variability=1)

        # Generate mean mesh for saving / subsequent warping
        # --------------------------------------------------------------------------------------------------------------
        # Project mean points
        mean_points = point_embedder.project(np.zeros(len(subjects) - 1))

        # Break points back into domains
        mean_points_split = np.split(mean_points, np.cumsum(domain_n_points))
        ref_points_split = np.split(all_points_transformed[ref_subject_index], np.cumsum(domain_n_points))

        # Do warping
        mean_meshes = []
        for ref_mesh, ref_points, m_points in zip(ref_meshes_transformed, ref_points_split, mean_points_split):
            warper = sw.MeshWarper()
            warper.generateWarp(ref_mesh, ref_points)
            warped_mesh = warper.buildMesh(m_points)
            warped_mesh = sw.sw2vtkMesh(warped_mesh)
            mean_meshes.append(warped_mesh)

        # Get path relative to derivative from path relative to shapeworks project. Then get sids
        mesh_dirpaths = [subject.get_groomed_filenames()[0].split(str(dataloc.rel_derivative))[-1] for subject in
                         subjects]
        sids = [Path(path).parts[dataset_config.subject_id_folder_depth - 1] for path in
                mesh_dirpaths]  # -1 this time because it starts with a slash

        point_embedder.write_PCA(output_directory, pca_score_ids=sids)

        # Write domain names and domain n_points. These are in the order used in the pca embedder, so we can use these
        # to split back into domains
        domain_names = [re.search(task_config.mesh_file_domain_name_regex, str(Path(f).stem)).group()
                        for f in subjects[0].get_local_particle_filenames()]

        mean_mesh_filenames = []
        for name, mean_mesh in zip(domain_names, mean_meshes):
            mean_mesh_filename = output_directory / f"{name}-mean.vtk"
            mean_mesh.save(str(mean_mesh_filename))
            mean_mesh_filenames.append(mean_mesh_filename)  # For returning created filenames

        domain_df = pd.DataFrame(data=np.array([domain_names, domain_n_points]).T, columns=["domain_name", "n_points"])
        domain_df.to_csv(output_directory / "domains.csv", index=False)


class SubjectDataPCACorrelationSW(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Find correlations between subject data and PCA components. Each mode analysed separately. First, RFECV is used
        to find the highest scoring set of subject data parameters for each mode. Then the f test is run to determine
        statistical significance of the model. Significant models are written as pickles, along with the regression
        analysis statistics.

        Parameters
        ----------
        dataloc
            Dataset Locator
        dirs_list
            List of relative paths to the source directories
        output_directory
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Dataset config
        task_config
            **source_directory_pca**
                source directory of the PCA model
            **source_directory_subject_data**
                source directory in which to find subject data
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **subject_data_filename**
                Filename of the subject data file
            **subject_data_keys**
                List of keys in the subject data to test for correlation with PCA modes
            **study_phase**
                Phase of the COPDGene study to draw subject data from
            **params**
                **percent_variability**
                    Threshold for cumulative proportion of variability. The lowest N modes that meet this threshold are
                    selected for inclusion
                **ftest_significance**
                    Significance threshold for f test. (Test of statistical significance for each linear regression model)

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        pca_directory = dataloc.abs_pooled_derivative / task_config.source_directory_pca
        embedder = PCA_Embbeder.from_directory(pca_directory)
        scores_df = pd.read_csv(glob(str(pca_directory / "original_PCA_scores*"))[0])
        # domain_df = pd.read_csv(pca_directory / "domains.csv")

        # This should be incorporated into PCA_Embedder. i.e., we want to save all the scores, but later compute
        # how many to use to preserve a given percent_variability. so a method for num_dim from percent_variability
        # also maybe access cumDst so we can plot compactness
        # task_config.params.percent_variability = 0.7
        cumDst = np.cumsum(embedder.eigen_values) / np.sum(embedder.eigen_values)
        num_dim = np.where(np.logical_or(cumDst > float(task_config.params.percent_variability),
                                         np.isclose(cumDst, float(task_config.params.percent_variability))))[0][0] + 1

        data_file = dataloc.abs_pooled_primary / task_config.source_directory_subject_data / task_config.subject_data_filename
        data = pd.read_csv(data_file, sep="\t", low_memory=False)

        # # Just whack it all in a sklearn linearRegression model
        # --------------------------------------------------------------------------------------------------------
        # Get subject data at the right ids and cols

        # task_config.subject_data_keys = ['gender', 'age_visit', 'Height_CM', 'Weight_KG']
        subject_data = data.loc[data.sid.isin(scores_df.id)].loc[data.Phase_study == task_config.study_phase][
            task_config.subject_data_keys]
        pca_scores = scores_df.filter(like="mode").iloc[:, :num_dim]

        # Drop rows where we don't have all subject data
        # (otherwise see https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values)
        subject_data = subject_data.dropna()
        pca_scores = pca_scores.loc[scores_df.id.isin(data.loc[subject_data.index].sid)]

        # Create a linear model for each mode, since we may need different predictors to predict each mode
        estimator = linear_model.LinearRegression()
        selector = RFECV(estimator)

        models = []
        reg_scores = []
        cv_mean_scores = []
        all_feature_names = []
        f_pvalues = []
        for mode in pca_scores:
            selector = selector.fit(subject_data, pca_scores[mode])
            cv_mean_scores.append(max(selector.cv_results_["mean_test_score"]))
            feature_names = selector.get_feature_names_out()
            model = linear_model.LinearRegression().fit(subject_data[feature_names], pca_scores[mode])
            models.append(model)
            reg_scores.append(model.score(subject_data[feature_names], pca_scores[mode]))
            all_feature_names.append(list(feature_names))

            X = sm.add_constant(subject_data[feature_names].values)
            sm_model = sm.OLS(pca_scores[mode].values, X)
            sm_results = sm_model.fit()
            f_pvalue = sm_results.f_pvalue
            f_pvalues.append(f_pvalue)

        significant_models = {mode: model for mode, model, f_pvalue in zip(pca_scores, models, f_pvalues)
                              if f_pvalue < task_config.params.ftest_significance}

        all_feature_names_flat = list(set(list(itertools.chain.from_iterable(all_feature_names))))
        mode_weighting = embedder.eigen_values / np.sum(embedder.eigen_values)
        total_weighted_score = sum([reg_scores[i] * mode_weighting[i] for i in range(len(f_pvalues)) if
                                    f_pvalues[i] < task_config.params.ftest_significance])

        all_feature_names_str = [", ".join(feature_names) for feature_names in all_feature_names]

        model_statistics = pd.DataFrame(data=np.array([pca_scores.columns.values,
                                                       mode_weighting[:len(pca_scores.columns)], reg_scores,
                                                       cv_mean_scores, f_pvalues, all_feature_names_str]).T,
                                        columns=["mode", "mode_weighting", "regression_scores",
                                                 "cross_validation_mean_scores", "f-test_p-values", "feature_names"])
        model_statistics_file = output_directory / "model_statistics.csv"
        model_statistics.to_csv(model_statistics_file, index=False)

        # Make sure to delete old files since these modes may now be excluded
        old_models = glob(str(output_directory / "*linear_model.pickle"))
        for model in old_models:
            Path(model).unlink()

        model_files = []
        for mode, model in significant_models.items():
            output_path_linear_model = output_directory / f"{mode}-linear_model.pickle"
            with open(output_path_linear_model, "wb") as f:
                pickle.dump(model, f)
            model_files.append(output_path_linear_model)

        output_path_mean_subject_data = output_directory / "mean_subject_data.csv"
        subject_data.mean().to_csv(output_path_mean_subject_data, index_label="cols")

        return [model_statistics_file, output_path_linear_model, output_path_mean_subject_data]


class GenerateMeshesMatchingSubjectsSW(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:

        """
        Use pre built PCA and Linear Regression models to generate meshes that estimate the current subjects in dirs_list
        using the specified subject data. Then calculate the RMS error between reference and predicted meshes, and
        between reference and mean meshes, for each domain.

        Note: this could be an each item task with an initialize if initialize got the dirs list

        Parameters
        ----------
        dataloc
            Dataset Locator
        dirs_list
            List of relative paths to the source directories
        output_directory
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Dataset config
        task_config
            **source_directory_pca**
                source directory of the PCA model
            **source_directory_subject_data**
                Source directory in which to find subject data
            **source_directory_linear_model**
                Source directory in which to find linear regression model pickles
            **source_directories_reference_mesh**
                List of directory names in which to search for reference meshes
            **mesh_file_domain_name_regex**:
                Regex pattern to match domain name within mesh file name
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **subject_data_filename**
                Filename of the subject data file
            **subject_data_keys**
                List of keys in the subject data to test for correlation with PCA modes
            **study_phase**
                Phase of the COPDGene study to draw subject data from
            **params**
                **alignment_iterations**
                    Iterations for rigid alignment between reference meshes and predicted and mean meshes

        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Load PCA model and linear regression models
        # --------------------------------------------------------------------------------------------------------------
        pca_directory = dataloc.abs_pooled_derivative / task_config.source_directory_pca
        embedder = PCA_Embbeder.from_directory(pca_directory)
        domain_df = pd.read_csv(pca_directory / "domains.csv")

        data_file = dataloc.abs_pooled_primary / task_config.source_directory_subject_data / task_config.subject_data_filename
        data = pd.read_csv(data_file, sep="\t", low_memory=False)

        def load_pickle(file):
            with open(file, "rb") as f:
                return pickle.load(f)

        linear_models = load_with_category(
            search_dirs=[dataloc.abs_pooled_derivative / task_config.source_directory_linear_model],
            category_regex=task_config.mesh_file_domain_name_regex,
            load_glob="*.pickle",
            loader=load_pickle)
        linear_models = dict(sorted(linear_models.items(), key=lambda item: int(item[0].split()[-1])))

        mean_subject_data_file = dataloc.abs_pooled_derivative / task_config.source_directory_linear_model / "mean_subject_data.csv"
        mean_subject_data = pd.read_csv(mean_subject_data_file, index_col="cols")

        sids = [Path(dirpath).parts[dataset_config.subject_id_folder_depth - 2] for dirpath, _, _ in dirs_list]

        # Get subject data for new subjects to predict
        # --------------------------------------------------------------------------------------------------------------
        subject_data = data.loc[data.sid.isin(sids)].loc[data.Phase_study == task_config.study_phase][
            task_config.subject_data_keys]

        # If we don't have all the predictors for the regression model, fill them in with the means
        missing_cols = mean_subject_data.index[~mean_subject_data.index.isin(subject_data.columns)]
        if missing_cols.size > 0:
            for col in missing_cols:
                subject_data[col] = mean_subject_data.loc[col].values[0]

        # Generate predicted points
        # --------------------------------------------------------------------------------------------------------------

        all_predicted_scores = {key: model.predict(subject_data[model.feature_names_in_]) for key, model in
                                linear_models.items()}

        # Fill in missing modes with zeros
        mode_indices = {int(k.split()[-1]) - 1: k for k in linear_models.keys()}  # Embedder names modes starting at 1
        last_mode_index = max(mode_indices.keys())

        all_predicted_scores_filled = []
        for i in range(last_mode_index + 1):
            if i in mode_indices.keys():
                all_predicted_scores_filled.append(all_predicted_scores[mode_indices[i]])

            else:
                all_predicted_scores_filled.append(np.zeros(len(all_predicted_scores[mode_indices[last_mode_index]])))

        all_projected_points = [embedder.project(scores) for scores in np.array(all_predicted_scores_filled).T]

        # Generate predicted meshes
        # --------------------------------------------------------------------------------------------------------------
        # Split predicted points back into domains
        all_points_split = []
        for projected_points in all_projected_points:
            points_split = dict(zip(domain_df.domain_name, np.split(projected_points, np.cumsum(domain_df.n_points))))
            all_points_split.append(points_split)

        mean_points = embedder.project(np.zeros(len(all_predicted_scores_filled[0])))
        mean_points_split = dict(zip(domain_df.domain_name, np.split(mean_points, np.cumsum(domain_df.n_points))))

        # Load the mean meshes to use as the base for warping
        mean_meshes = load_with_category(search_dirs=[dataloc.abs_pooled_derivative / task_config.source_directory_pca],
                                         category_regex=task_config.mesh_file_domain_name_regex,
                                         load_glob="*.vtk",
                                         loader=sw.Mesh)

        # Warp mean meshes to predicted points for all subjects, for all domains
        all_predicted_meshes = []
        for points_split, (dirpath, _, _) in zip(all_points_split, dirs_list):
            domain_meshes = {}
            for domain in points_split.keys():
                warper = sw.MeshWarper()
                warper.generateWarp(mean_meshes[domain], mean_points_split[domain])
                warped_mesh = warper.buildMesh(points_split[domain])
                domain_meshes[domain] = warped_mesh

            all_predicted_meshes.append(domain_meshes)

        # sw.sw2vtkMesh(all_predicted_meshes[0]["left_lung"]).plot(show_edges=True)

        # Load reference meshes
        # --------------------------------------------------------------------------------------------------------------
        all_reference_meshes = []
        for dir_path, _, _ in dirs_list:
            reference_meshes = load_with_category(search_dirs=[dataloc.abs_derivative / dir_path / d for d in
                                                               task_config.source_directories_reference_mesh],
                                                  category_regex=task_config.mesh_file_domain_name_regex,
                                                  load_glob="*.vtk",
                                                  loader=sw.Mesh)
            all_reference_meshes.append(reference_meshes)

        # Compute RMS error for predicted to reference and mean to reference
        # --------------------------------------------------------------------------------------------------------------
        meshes_ref_aligned_to_mean = []
        ref_to_predicted_transforms = []
        ref_to_mean_transforms = []
        # Combine all sets of meshes..
        combined_mean_mesh = list(mean_meshes.values())[0].copy()
        for mesh in list(mean_meshes.values())[1:]:
            combined_mean_mesh += mesh

        for predicted_meshes, reference_meshes in zip(all_predicted_meshes, all_reference_meshes):
            # Copy so we don't destroy meshes[0] for later!
            combined_predicted_mesh = list(predicted_meshes.values())[0].copy()
            for mesh in list(predicted_meshes.values())[1:]:
                combined_predicted_mesh += mesh

            combined_reference_mesh = list(reference_meshes.values())[0].copy()
            for mesh in list(reference_meshes.values())[1:]:
                combined_reference_mesh += mesh

            # Find rigid transform for reference to predicted
            ref_to_predicted = combined_reference_mesh.createTransform(combined_predicted_mesh,
                                                                       sw.Mesh.AlignmentType.Rigid,
                                                                       task_config.params.alignment_iterations)
            ref_to_predicted_transforms.append(ref_to_predicted)

            # a = combined_reference_mesh.copy()
            # a = a.applyTransform(ref_to_predicted)
            # p = pv.Plotter()
            # p.add_mesh(sw.sw2vtkMesh(a).extract_all_edges(), color="red")
            # p.add_mesh(sw.sw2vtkMesh(combined_predicted_mesh).extract_all_edges(), color="blue")
            # p.show()

            # Find rigid transform for reference to mean
            ref_to_mean = combined_reference_mesh.createTransform(combined_mean_mesh, sw.Mesh.AlignmentType.Rigid,
                                                                  task_config.params.alignment_iterations)
            ref_to_mean_transforms.append(ref_to_mean)

            ref_aligned_to_mean = combined_reference_mesh.copy()
            ref_aligned_to_mean = ref_aligned_to_mean.applyTransform(ref_to_mean)
            meshes_ref_aligned_to_mean.append(ref_aligned_to_mean)

        # Calculate errors per domain
        all_domain_ref_to_pred_errors = []
        all_domain_ref_to_mean_errors = []
        combined_ref_to_pred_errors = []
        combined_ref_to_mean_errors = []
        for i, (domain_reference_meshes, domain_predicted_meshes, ref_to_predicted, ref_to_mean) in \
                enumerate(zip(all_reference_meshes, all_predicted_meshes, ref_to_predicted_transforms,
                              ref_to_mean_transforms)):

            logger.debug(f"Finding rms error for subject {i + 1} of {len(all_predicted_meshes)}")

            ref_to_pred_errors = {}
            ref_to_mean_errors = {}

            weighted_squares_pred = []
            weighted_squares_mean = []
            total_points = 0
            for k in domain_predicted_meshes.keys():
                logger.debug(f"{k}")
                pred_mesh = domain_predicted_meshes[k]
                mean_mesh = mean_meshes[k]

                ref_mesh = domain_reference_meshes[k].copy()
                mesh_ref_to_pred = ref_mesh.applyTransform(ref_to_predicted)

                # p = pv.Plotter()
                # p.add_mesh(sw.sw2vtkMesh(mesh_ref_to_pred).extract_all_edges(), color="red")
                # p.add_mesh(sw.sw2vtkMesh(pred_mesh).extract_all_edges(), color="blue")
                # p.show()

                err_ref_to_pred = mesh_rms_error(sw.sw2vtkMesh(mesh_ref_to_pred), sw.sw2vtkMesh(pred_mesh),
                                                 show_progress=True)

                ref_mesh = domain_reference_meshes[k].copy()
                mesh_ref_to_mean = ref_mesh.applyTransform(ref_to_mean)
                err_ref_to_mean = mesh_rms_error(sw.sw2vtkMesh(mesh_ref_to_mean), sw.sw2vtkMesh(mean_mesh),
                                                 show_progress=True)

                ref_to_pred_errors[k] = err_ref_to_pred
                ref_to_mean_errors[k] = err_ref_to_mean

                total_points += ref_mesh.numPoints()
                weighted_squares_pred.append(err_ref_to_pred ** 2 * ref_mesh.numPoints())
                weighted_squares_mean.append(err_ref_to_mean ** 2 * ref_mesh.numPoints())

            all_domain_ref_to_pred_errors.append(ref_to_pred_errors)
            all_domain_ref_to_mean_errors.append(ref_to_mean_errors)
            combined_ref_to_pred_errors.append(np.sqrt(np.sum(weighted_squares_pred) / total_points))
            combined_ref_to_mean_errors.append(np.sqrt(np.sum(weighted_squares_mean) / total_points))

        # Save predicted meshes in subject folders
        # --------------------------------------------------------------------------------------------------------------
        for domain_meshes, (dirpath, _, _) in zip(all_predicted_meshes, dirs_list):
            if not os.path.exists(dataloc.abs_derivative / dirpath / task_config.results_directory):
                os.makedirs(dataloc.abs_derivative / dirpath / task_config.results_directory)

            for name, mesh in domain_meshes.items():
                domain_mesh_filename = dataloc.abs_derivative / dirpath / task_config.results_directory \
                                       / f"{name}-predicted.vtk"
                sw.sw2vtkMesh(mesh).save(domain_mesh_filename)

        # Save results per domain
        rms_errors_by_domain_file = output_directory / "rms_errors_by_domain.csv"

        error_by_domain_df = pd.DataFrame(
            columns=[f"{k} error_reference_to_predicted" for k in all_predicted_meshes[0].keys()] + [
                "combined error_reference_to_predicted"] +
                    [f"{k} error_reference_to_mean" for k in all_predicted_meshes[0].keys()] + [
                        "combined error_reference_to_mean"],
            data=np.hstack((np.array([list(d.values()) for d in all_domain_ref_to_pred_errors]),
                            np.array([combined_ref_to_pred_errors]).T,
                            np.array([list(d.values()) for d in all_domain_ref_to_mean_errors]),
                            np.array([combined_ref_to_mean_errors]).T)))

        error_by_domain_df.to_csv(rms_errors_by_domain_file)

        # Save ref_aligned_to_mean meshes in subject folders
        # --------------------------------------------------------------------------------------------------------------
        for domain_meshes, mesh_transform, (dirpath, _, _) in zip(all_reference_meshes, ref_to_mean_transforms,
                                                                  dirs_list):
            if not os.path.exists(dataloc.abs_derivative / dirpath / task_config.results_directory):
                os.makedirs(dataloc.abs_derivative / dirpath / task_config.results_directory)

            for name, mesh in domain_meshes.items():
                mesh_aligned_to_mean = mesh.applyTransform(mesh_transform)

                domain_mesh_filename = dataloc.abs_derivative / dirpath / task_config.results_directory \
                                       / f"{name}-reference_aligned_to_mean.vtk"
                sw.sw2vtkMesh(mesh_aligned_to_mean).save(domain_mesh_filename)

        return []


class GenerateMeshesWithSubjectDataSW(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Use subject data regression model and PCA model to generate meshes based on subject data.

        # TODO: finish

        Parameters
        ----------
        dataloc
            Dataset Locator
        dirs_list
            List of relative paths to the source directories
        output_directory
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Dataset config
        task_config
            **source_directory**
                source directory of Todo:finish this
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **params**
                None implemented

        """

        if task_config.source_directory_parent == "primary":
            pca_directory = dataloc.abs_pooled_primary / task_config.source_directory
        else:
            pca_directory = dataloc.abs_pooled_derivative / task_config.source_directory


all_tasks = [ExtractLungLobesSW, CreateMeshesSW, ExtractWholeLungsSW, ReferenceSelectionMeshSW, MeshTransformSW,
             OptimizeMeshesSW, ComputePCASW, SubjectDataPCACorrelationSW, GenerateMeshesMatchingSubjectsSW]
