from lung_modelling.workflow_manager import EachItemTask, DatasetLocator
from pathlib import Path
from omegaconf import DictConfig
import os


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

        # Todo: groom files here. Use shapeworks!
        # - Load images
        # - Antialias, resample (Maybe save this)
        # - Convert to mesh (so we can extract shared boundary and find landmarks)
        # ------------------------------------------------
        # - Extract shared boundaries (Needs to be in an AllItemsTask) (No it doesn't. It's all lobes, not all subjects,
        #  AllItemsTasks will be things like finding a reference mesh to register all others against
        groomed_files = []

        relative_files = [str(dataloc.to_relative(Path(file))) for file in groomed_files]

        return relative_files


all_tasks = [SmoothLungLobesSW]
