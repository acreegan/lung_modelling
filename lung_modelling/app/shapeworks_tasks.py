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
from lung_modelling import find_connected_faces, flatten, voxel_to_mesh, fix_mesh
from lung_modelling.shapeworks_libs import PCA_Embbeder
import medpy.io
import pandas as pd
from DataAugmentationUtils import Utils, Embedder
from fnmatch import fnmatch


class SmoothLungLobesSW(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
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
            mesh = sw.Mesh(mesh.points, pyvista_faces_to_2d(mesh.faces))

            if params.decimate:
                mesh = sw.sw2vtkMesh(mesh)
                mesh = mesh.clean()

                t_faces = params.decimate_target_faces / (4 ** params.subdivide_passes)
                target_reduction = 1 - (t_faces / mesh.n_faces)
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
                if pv_mesh.n_faces == 0:
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


class SmoothWholeLungsSW(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
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

                domain_reference_filename = output_directory / f"{domain_reference_name}_reference_mesh.vtk"
                domain_reference_mesh.write(str(domain_reference_filename))
                domain_reference_filenames.append(domain_reference_filename)

        ref_mesh_combined_filename = output_directory / f"combined_reference_mesh.vtk"
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
        optimize_cmd = ("shapeworks optimize --progress --name " + str(spreadsheet_file)).split()
        subprocess.check_call(optimize_cmd)
        os.chdir(wd)


class ComputePCASW(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Generate PCA model from the optimized shapeworks particle system

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
            **params**
                None implemented

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

        # Load reference meshes and apply transform
        ref_meshes_transformed = []
        for relative_filename in subjects[ref_subject_index].get_groomed_filenames():
            # File paths stored relative to project directory
            ref_mesh_filename = os.path.normpath(os.path.join(project_directory, relative_filename))
            ref_mesh = sw.Mesh(ref_mesh_filename)
            ref_mesh_transformed = ref_mesh.applyTransform(subject_transforms[ref_subject_index])
            ref_meshes_transformed.append(ref_mesh_transformed)

        # Do PCA
        # point_embedder = PCA_Embbeder(all_points_transformed, num_dim=len(subjects)-1)
        point_embedder = PCA_Embbeder(all_points_transformed, percent_variability=1)

        # Project mean points
        mean_points = point_embedder.project(np.zeros(len(subjects)-1))

        # Break points back into domains
        mean_points_split = np.split(mean_points, np.cumsum(domain_n_points))
        ref_points_split = np.split(all_points_transformed[ref_subject_index], np.cumsum(domain_n_points))

        # Do warping
        mean_meshes = []
        for ref_mesh, ref_points, m_points in zip(ref_meshes_transformed, ref_points_split, mean_points_split):
            warper = sw.MeshWarper()
            warper.generateWarp(ref_mesh, ref_points)
            warped_mesh = warper.buildMesh(m_points)
            mean_meshes.append(warped_mesh)

        point_embedder.write_PCA(output_directory, score_option="stdev")


        # Write meshes with domain name and PCA domain index
        # In CreateMeshesSW, the convention is to place the domain name before a dash ("-"). We follow this convention
        # to find the domain names again.
        domain_names = [Path(f).stem.split("-")[0] for f in subjects[0].get_local_particle_filenames()]



        pass


class SubjectDataPCACorrelationSW(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Find correlations between subject data and PCA components

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
                source directory of the PCA model
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **params**
                None implemented

        """

        # Load PCA from directory
        embedder = PCA_Embbeder.from_directory(task_config.source_directory)

        pass 


all_tasks = [SmoothLungLobesSW, CreateMeshesSW, SmoothWholeLungsSW, ReferenceSelectionMeshSW, MeshTransformSW,
             OptimizeMeshesSW, ComputePCASW, SubjectDataPCACorrelationSW]
