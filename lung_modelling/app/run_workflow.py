import tkinter as tk
from tkinter.filedialog import askdirectory
from pathlib import Path
from multiprocess.pool import Pool
from lung_modelling.workflow_manager import WorkflowManager
from lung_modelling.app.tasks import GroomLungsDataset
from loguru import logger
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

"""
A script to run a workflow
"""


@hydra.main(version_base=None, config_path="hydra_config", config_name="primary_config")
def run_cli(hydra_cfg: DictConfig):
    """
    Run command line interface
    """
    config_choice = "local config" if HydraConfig.get().runtime.choices.get(
        "config") == "local_config" else "default config"
    logger.enable("lung_annotations")
    logger.info(f"Running {__file__}, config choice: {config_choice}")
    cfg = hydra_cfg.config

    if cfg.dataset_root is None:
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
    for task in [GroomLungsDataset]:
        workflow_manager.register_task(task)
    workflow_manager.run_workflow(cfg.run_tasks)


if __name__ == "__main__":
    run_cli()
