import math

from lung_modelling.workflow_manager import EachItemTask, DatasetLocator, AllItemsTask
from pathlib import Path, PurePosixPath
from omegaconf import DictConfig
import os
from glob import glob
from lung_modelling import extract_section, voxel_to_mesh, refine_mesh, parse_discrete
import medpy.io
import SimpleITK as sitk
import numpy as np
import pyacvd
import cc3d
from scipy import ndimage
import pyvista as pv
import pandas as pd
import csv
from medpy.io import load


class SmoothLungLobes(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Pre-process lung lobe segmentation by extracting lobe labels and applying antialiasing.

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
                    Parameters to apply to SimpleITK.AntiAliasBinary

        Returns
        -------
        smoothed_files
            list of Path objects representing the files created.
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file = glob(str(source_directory_primary / dataset_config.lung_image_glob))[0]
        image_data, header = medpy.io.load(image_file)

        suffix = Path(image_file).suffix

        smoothed_files = []
        for lobe, name in task_config.output_filenames.items():
            lobe_array = extract_section(image_data, dataset_config.lobe_mapping[lobe])

            im = sitk.GetImageFromArray(lobe_array)
            im_aa = sitk.AntiAliasBinary(im, maximumRMSError=task_config.params.maximumRMSError,
                                         numberOfIterations=task_config.params.numberOfIterations)
            im_rs = sitk.Resample(im_aa)
            im_bin = sitk.BinaryThreshold(im_rs)
            lobe_array = np.array(sitk.GetArrayFromImage(im_bin))

            output_filename = f"{str(output_directory / task_config.output_filenames[lobe])}{suffix}"
            medpy.io.save(lobe_array, output_filename, hdr=header, use_compression=True)
            smoothed_files.append(Path(output_filename))

        return smoothed_files


class SmoothWholeLungs(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Pre-process lung segmentation by extracting whole lung and applying antialiasing.

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
                **maximumRMSError**, **umberOfIterations**:
                    Parameters to apply to SimpleITK.AntiAliasBinary

        Returns
        -------
        smoothed_files
            list of Path objects representing the files created.
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file = glob(str(source_directory_primary / dataset_config.lung_image_glob))[0]
        image_data, header = medpy.io.load(image_file)

        suffix = Path(image_file).suffix

        smoothed_files = []
        for name, lobes in task_config.output_filenames.items():
            lobe_arrays = [extract_section(image_data, dataset_config.lobe_mapping[lobe]) for lobe in lobes]
            merged = lobe_arrays[0]
            for array in lobe_arrays:
                merged = np.logical_or(merged, array).astype(int)

            im = sitk.GetImageFromArray(merged)
            im_aa = sitk.AntiAliasBinary(im, maximumRMSError=task_config.params.maximumRMSError,
                                         numberOfIterations=task_config.params.numberOfIterations)
            im_rs = sitk.Resample(im_aa)
            im_bin = sitk.BinaryThreshold(im_rs)
            lobe_array = np.array(sitk.GetArrayFromImage(im_bin))

            output_filename = f"{str(output_directory / name)}{suffix}"
            medpy.io.save(lobe_array, output_filename, hdr=header, use_compression=True)
            smoothed_files.append(Path(output_filename))

        return smoothed_files


class CreateMeshes(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Convert medical image files to meshes and apply smoothing.

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
                **n_iter**, **feature_smoothing**, **edge_angle**, **feature_angle**, **relaxation_factor**:
                    Params for pyvista smooth

                **target_reduction**, **volume_preservation**:
                    Params for pyvista decimate

                **hole_size**:
                    Param for pyvista fill_holes

                **fix_mesh**:
                    Option to fix mesh


        Returns
        -------
        mesh_files
            list of Path objects representing the files created.
        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_files = glob(str(source_directory_derivative / task_config.source_directory / "*"))

        if len(image_files) == 0:
            raise RuntimeError("No files found")

        mesh_files = []
        for image_file in image_files:
            image_data, header = medpy.io.load(image_file)
            # id = cc3d.dust(image_data, threshold=100)
            mesh = voxel_to_mesh(image_data, spacing=header.spacing, direction=header.direction, offset=header.offset)

            refined_mesh = refine_mesh(mesh, params=task_config.params)

            output_filename = str(output_directory / Path(image_file).stem) + '.stl'
            refined_mesh.save(output_filename)
            mesh_files.append(Path(output_filename))

        return mesh_files


class ReferenceSelectionMesh(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: Path, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        A task to load all meshes at once so the shape closest to the mean can be found and selected as the reference


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
            Task specific config


        Returns
        -------
        reference
            Mesh selected as the reference

        """
        raise NotImplementedError("Reference selection mesh not yet implemented for non shapeworks libs")


class ExtractTorso(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Extract a torso image from a set of dicom files. Segmented torso is saved in a .nii file

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
            **source_directory**: subdirectory for dicom files

            **results_directory**: subdirectory for results

            **output_filename**: filename for torso image (not including extension)

            **params**: (Dict)
                **threshold**
                    threshold for segmenting torso from dicom image

        Returns
        -------
        [output_filename]
            single item list containing the output filename of the torso image
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image, header = medpy.io.load(str(source_directory_primary / task_config.source_directory))

        binary_image = np.array(image > task_config.params.threshold, dtype=np.int8)

        max_connectivity = cc3d.largest_k(binary_image, k=1)
        image_filled_holes = max_connectivity.copy()
        for i in range(max_connectivity.shape[-1]):
            image_filled_holes[:, :, i] = ndimage.binary_fill_holes(max_connectivity[:, :, i])

        output_filename = f"{str(output_directory / task_config.output_filename)}.nii"
        medpy.io.save(image_filled_holes, output_filename, hdr=header, use_compression=True)

        return [Path(output_filename)]


class MeshLandmarksCoarse(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Calculate points spaced roughly equidistantly around the surface of a mesh. This should help get
        good coverage in optimization without needing to lower initial relative weighting as much.

        Points are placed by tracing a ray from the mesh center of mass to the vertices of a cube. This results in
        8 landmarks. The cube is oriented such that two of its vertices point straight up and down.

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

            **params**: (Dict): No params currently used for this task

        initialize_result
            Return dict from the initialize function

        Returns
        -------
        List of mesh landmark filenames. These take the same form as shapeworks particles files

        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        mesh_files = glob(str(source_directory_derivative / task_config.source_directory / "*"))

        if len(mesh_files) == 0:
            raise RuntimeError("No files found")

        meshes = []
        for mesh_file in mesh_files:
            meshes.append(pv.read(mesh_file))

        mesh_landmark_filenames = []
        for mesh, file in zip(meshes, mesh_files):
            com = mesh.center_of_mass()
            bounds = np.array(mesh.bounds).reshape(3, -1)

            longest_side = np.max(bounds[:, 1] - bounds[:, 0])

            cube = pv.Box(quads=True).scale([longest_side, longest_side, longest_side])
            cube = cube.rotate_x(45).rotate_y(35.264)
            cube = cube.translate(com - cube.center_of_mass())

            mesh_landmarks = []
            for point in cube.points:
                landmark = mesh.ray_trace(com, point)[0][-1]
                mesh_landmarks.append(landmark)

            mesh_landmark_filename = output_directory / f"{str(Path(file).stem)}_landmarks.particles"
            np.savetxt(str(mesh_landmark_filename), mesh_landmarks, delimiter=" ")
            mesh_landmark_filenames.append(mesh_landmark_filename)

        return mesh_landmark_filenames


class ParseCOPDGeneSubjectGroups(AllItemsTask):
    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        A task to parse subject data from the COPDGene dataset so that it can be added to a shapeworks style config file.

        This is an AllItemsTask even though it involves parsing each individual subject separately because we don't
        want pass the large subject data array to lots of duplicate threads. Also the parsing should not take long.


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
            **source_directory**
                subdirectory within derivative source folder to find source files
            **results_directory**
                subdirectory for results
            **subject_data_filename**
                filename for COPDGene subject data file
            **subject_data_dict_filename**
                filename for COPDGene subject data dict file
            **groups**
                list of group labels to parse
            **params**: (Dict): No params currently used for this task


        Returns
        -------
        parsed_data
            File containing parsed groups

        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        data_file = dataloc.abs_pooled_primary / task_config.source_directory / task_config.subject_data_filename
        data = pd.read_csv(data_file, sep="\t")

        data_dict_file = dataloc.abs_pooled_primary / task_config.source_directory / task_config.subject_data_dict_filename
        data_dict = pd.read_excel(data_dict_file)
        data_dict = data_dict.set_index("VariableName")

        group_mappings = {group: parse_discrete(data_dict["CodedValues"][group]) for group in task_config.groups}

        group_data_header = ["dir", *task_config.groups]
        group_values = []
        for dir, _, _ in dirs_list:
            sid = dir.stem.split("_")[0]
            subject_group_values = []
            subject_group_values.append(str(PurePosixPath(dir)))
            for group in task_config.groups:
                raw_value = data.loc[data.sid == sid][group].values[0]
                mapped_value = group_mappings[group][raw_value]
                subject_group_values.append(mapped_value)

            group_values.append(subject_group_values)

        group_data_filename = output_directory / "group_data.csv"
        with open(str(group_data_filename), "w") as ref_dir_file:
            writer = csv.writer(ref_dir_file)
            writer.writerow(group_data_header)

            for row in group_values:
                writer.writerow(row)

        return [Path(group_data_filename)]


class InspectSegmentations(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Display groups of segmentations. Performs a simple marching cubes on them to make them esaier to view

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
            **source_directory**
                subdirectory within derivative source folder to find source files
            **params**: (Dict): No params currently used for this task


        Returns
        -------
        """

        n_per_window = 9
        n_windows = math.ceil(len(dirs_list)/n_per_window)
        selected_dirs = dict()
        for window_index, i in enumerate(range(0, len(dirs_list), n_per_window)):
            shape = (3, 3)
            p = pv.Plotter(shape=shape)

            for j in range(n_per_window):
                if i + j >= len(dirs_list):
                    break

                dir = dirs_list[i + j][0]
                p.subplot(*np.unravel_index(j, shape))
                file = glob(str(dataloc.abs_derivative / dir / task_config.source_directory / "*"))[0]
                image_data, header = load(file)

                mesh = voxel_to_mesh(image_data, spacing=header.spacing, direction=header.direction,
                                     offset=header.offset, step_size=3)

                if i == 0:
                    ref_mesh = mesh
                else:
                    mesh = mesh.align(ref_mesh)

                class update_selected:
                    def __init__(self, dir):
                        self.dir = dir

                    def __call__(self, state):
                        selected_dirs[self.dir] = state
                        if state:
                            print(f"Selected: {self.dir}")
                        else:
                            print(f"Deselected: {self.dir}")

                p.add_mesh(mesh)
                p.add_title(Path(dir).stem, font_size=8)
                p.add_checkbox_button_widget(update_selected(dir))

            p.link_views()
            p.show(title=f"Window {window_index+1}/{n_windows}")

        print(f"Selected dirs:")
        for dir, value in selected_dirs.items():
            if value:
                print(str(dir))





all_tasks = [SmoothLungLobes, CreateMeshes, SmoothWholeLungs, ReferenceSelectionMesh, ExtractTorso, MeshLandmarksCoarse,
             ParseCOPDGeneSubjectGroups, InspectSegmentations]
