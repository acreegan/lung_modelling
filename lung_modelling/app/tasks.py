from lung_modelling.workflow_manager import EachItemTask, DatasetLocator, AllItemsTask
from pathlib import Path
from omegaconf import DictConfig
import os
from glob import glob
from lung_modelling import extract_section, voxel_to_mesh, refine_mesh
import medpy.io
import SimpleITK as sitk
import numpy as np
import pyacvd
import cc3d


class SmoothLungLobes(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @property
    def name(self):
        return "smooth_lung_lobes"

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

    @property
    def name(self):
        return "smooth_whole_lungs"

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

    @property
    def name(self):
        return "create_meshes"

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
    @property
    def name(self):
        return "reference_selection_mesh"

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


all_tasks = [SmoothLungLobes, CreateMeshes, SmoothWholeLungs, ReferenceSelectionMesh]
