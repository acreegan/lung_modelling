from lung_modelling.workflow_manager import EachItemTask, DatasetLocator
from pathlib import Path
from omegaconf import DictConfig
import os


class GroomLungsDataset(EachItemTask):

    @property
    def name(self):
        return "groom_lungs"

    @staticmethod
    def work(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig,
             source_directory: Path) -> list:
        """
        Groom lungs

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

        # Todo: groom files here
        # - Load images
        # - Antialias, resample (Maybe save this)
        # - Convert to mesh (so we can extract shared boundary and find landmarks)
        # ------------------------------------------------
        # - Extract shared boundaries (Needs to be in an AllItemsTask)
        groomed_files = []

        relative_files = [str(dataloc.to_relative(Path(file))) for file in groomed_files]

        return relative_files
