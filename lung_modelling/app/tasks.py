from lung_modelling.workflow_manager import EachItemTask, DatasetLocator
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

    @property
    def name(self):
        return "smooth_lung_lobes"

    @staticmethod
    def work(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig,
             source_directory: Path) -> list:
        """
        smooth_lung_lobes

        Parameters
        ----------
        dataloc
            DatasetLocator
        dataset_config
            Dataset config
        task_config
            Task config
        source_directory
            Source directory

        Returns
        -------
        relative_files
            List of generated filenames relative to the dataset root

        """
        output_directory = dataloc.abs_derivative / source_directory / task_config.task_name
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file = glob(str(dataloc.abs_primary / source_directory / dataset_config.lung_image_glob))[0]
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
            smoothed_files.append(output_filename)

        relative_files = [str(dataloc.to_relative(Path(file))) for file in smoothed_files]

        return relative_files


class CreateMeshes(EachItemTask):

    @property
    def name(self):
        return "create_meshes"

    @staticmethod
    def work(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig,
             source_directory: Path) -> list:
        """
        create meshes

        Parameters
        ----------
        dataloc
            DatasetLocator
        dataset_config
            Dataset config
        task_config
            Task config
        source_directory
            Source directory

        Returns
        -------
        relative_files
            List of generated filenames relative to the dataset root

        """
        output_directory = dataloc.abs_derivative / source_directory / task_config.task_name
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_files = glob(str(dataloc.abs_derivative / source_directory / task_config.source_directory / "*"))

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
            mesh_files.append(output_filename)

        relative_files = [str(dataloc.to_relative(Path(file))) for file in mesh_files]

        return relative_files


all_tasks = [SmoothLungLobes, CreateMeshes]
