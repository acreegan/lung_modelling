from lung_modelling.workflow_manager import EachItemTask, DatasetLocator
from pathlib import Path
from omegaconf import DictConfig
import os
from glob import glob
from lung_modelling import extract_section
import medpy.io
import SimpleITK as sitk
import numpy as np


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

        # Todo: groom files here
        # - Load images
        # - Antialias, resample (Maybe save this)
        # - Convert to mesh (so we can extract shared boundary and find landmarks)
        # ------------------------------------------------
        # - Extract shared boundaries (Needs to be in an AllItemsTask) (No it doesn't. It's all lobes, not all subjects,
        #  AllItemsTasks will be things like finding a reference mesh to register all others against

        relative_files = [str(dataloc.to_relative(Path(file))) for file in smoothed_files]

        return relative_files


all_tasks = [SmoothLungLobes]
