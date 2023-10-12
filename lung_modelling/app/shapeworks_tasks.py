from lung_modelling.workflow_manager import EachItemTask, DatasetLocator
from pathlib import Path
from omegaconf import DictConfig
import os
import shapeworks as sw
from glob import glob
import pyvista as pv


class SmoothLungLobesSW(EachItemTask):

    @property
    def name(self):
        return "smooth_lung_lobes_sw"

    @staticmethod
    def work(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig,
             source_directory: Path) -> list:
        """
        smooth_lung_lobes_sw

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
        shape_seg = sw.Image(image_file)

        suffix = Path(image_file).suffix

        smoothed_lobes = []
        for lobe, name in task_config.output_filenames.items():
            s = shape_seg.copy()  # Copy because extractLabel is destructive
            lobe_image = s.extractLabel(dataset_config.lobe_mapping[lobe])
            iso_spacing = [1, 1, 1]
            lobe_image.antialias(task_config.params.numberOfIterations, task_config.params.maximumRMSError).resample(
                iso_spacing, sw.InterpolationType.Linear).binarize()

            filename = f"{str(output_directory / name)}{suffix}"
            lobe_image.write(filename)

            smoothed_lobes.append(filename)

        relative_files = [str(dataloc.to_relative(Path(file))) for file in smoothed_lobes]

        return relative_files


all_tasks = [SmoothLungLobesSW]
