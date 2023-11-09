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
from scipy import ndimage
import pyvista as pv


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


all_tasks = [SmoothLungLobes, CreateMeshes, SmoothWholeLungs, ReferenceSelectionMesh, ExtractTorso, MeshLandmarksCoarse]
