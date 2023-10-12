import tkinter as tk
from tkinter.filedialog import askdirectory
from pathlib import Path
from multiprocess.pool import Pool
from lung_modelling.workflow_manager import WorkflowManager
from loguru import logger
import hydra
from hydra.core.hydra_config import HydraConfig
import hydra.utils
from omegaconf import DictConfig
import os
import shutil

"""
A script to run a workflow
"""


@hydra.main(version_base=None, config_path="default_configs", config_name="primary_config")
def run_cli(primary_config: DictConfig):
    """
    Run command line interface
    """
    logger.enable("lung_modelling")
    if primary_config.initialize_user_config:
        initialize_user_config()
        exit()

    hydra_cfg = HydraConfig.get()
    if "user_config" in primary_config:
        # Found a user config
        cfg = primary_config.user_config
        config_choice = hydra_cfg.runtime.choices.get("user_config")
        logger.info(f"Running {Path(__file__).name}, config choice: user_config/{config_choice}")
    else:
        # Using the default config
        cfg = primary_config.default_config
        config_choice = hydra_cfg.runtime.choices.get("default_config")
        logger.info(f"Running {Path(__file__).name}, config choice: default_config/{config_choice}")

    if cfg.dataset_root is None:
        # Todo could use compose API to add this. Then hydra logging wouldn't be ruined.
        tk.Tk().withdraw()
        if not (dataset_root := askdirectory(title="Select dataset root directory")):
            exit()
        cfg.dataset_root = Path(dataset_root)
    else:
        cfg.dataset_root = Path(cfg.dataset_root)

    if cfg.use_multiprocessing:
        mpool = Pool()
    else:
        mpool = None

    workflow_manager = WorkflowManager(cfg.dataset_root, cfg, mpool, show_progress=True)

    # Register tasks
    if config_choice == "standard":
        from lung_modelling.app.tasks import all_tasks
        for task in all_tasks:
            workflow_manager.register_task(task)

    if config_choice == "shapeworks":
        from lung_modelling.app.shapeworks_tasks import all_tasks as all_sw_tasks
        for task in all_sw_tasks:
            workflow_manager.register_task(task)

    workflow_manager.run_workflow(cfg.run_tasks)


def initialize_user_config():
    """
    Initialize user config by copying default config files to a folder in the current working directory
    """
    logger.info("Initializing user config")
    original_cwd = hydra.utils.get_original_cwd()
    config_dirname = "user_config"
    output_dir = Path(original_cwd) / config_dirname
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    default_config_path = Path(__file__).parent / "default_configs" / "default_config"
    for file in os.listdir(default_config_path):
        shutil.copy(default_config_path / file, output_dir)


if __name__ == "__main__":
    run_cli()
