import fnmatch
import functools
import glob
import json
import os
import platform
import sys
import time
from ast import literal_eval
from pathlib import Path
from typing import Callable
from abc import ABCMeta, abstractmethod
import pandas as pd
import importlib.metadata
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from typing import Tuple, Type
from multiprocess.pool import Pool


class DatasetLocator:
    def __init__(self, root: Path, rel_primary: Path, rel_derivative: Path):
        self.root = Path(root)
        self.rel_primary = Path(rel_primary)
        self.rel_derivative = Path(rel_derivative)

    @property
    def abs_primary(self):
        return self.root / self.rel_primary

    @property
    def abs_derivative(self):
        return self.root / self.rel_derivative

    def to_relative(self, path: Path) -> Path:
        for parent in path.parents:
            if parent.name == self.rel_primary.name:
                return self.rel_primary / path.relative_to(self.abs_primary)
            elif parent.name == self.rel_derivative.name:
                return self.rel_derivative / path.relative_to(self.abs_derivative)

        raise ValueError("Neither primary nor derivative found in input path")


class EachItemTask:
    __metaclass___ = ABCMeta

    @property
    @abstractmethod
    def name(self):
        pass

    @staticmethod
    @abstractmethod
    def work(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig,
             source_directory: Path) -> list:
        pass


def initialize(dataset_root: Path, task_config: DictConfig, show_progress=True) -> Tuple[
    DatasetLocator, DictConfig, list]:
    """
    Initialization task for processing a dataset directory structure. Loads the dataset config and runs
    gather_directories.

    Parameters
    ----------
    dataset_root
        Root directory of the dataset
    task_config
        Configuration dict for this task
    show_progress
        Option to show progress

    Returns
    -------
    dataset_config
        DatasetConfig object
    dirs_list
        Record of data folders with list of files

    """
    dataset_config_file = dataset_root / task_config["dataset_config_filename"]

    with open(dataset_config_file, "r") as f:
        dataset_config = DictConfig(json.load(f))

    dataloc = DatasetLocator(dataset_root, dataset_config.primary_directory, dataset_config.derivative_directory)

    index_list = None
    if task_config.use_directory_index:
        if len(glob_result := glob.glob(str(dataset_root / dataset_config.directory_index_glob))) > 0:
            index_list = pd.read_csv(glob_result[0], converters={"files": literal_eval}).values.tolist()

    dirs_list = gather_directories(dataloc.abs_primary, dataset_config.data_folder_depth, task_config.skip_dirs,
                                   task_config.select_dirs, index_list, show_progress)

    return dataloc, dataset_config, dirs_list


def gather_directories(primary_root: Path, data_folder_depth, skip_dirs=None, select_dirs=None, index_list=None,
                       show_progress=True) -> list[tuple[Path, list, list]]:
    """
    Gather directories with files to process. Walks through the dataset directory finding data folders at a specified
    depth.

    Parameters
    ----------
    primary_root
        Root of the source directory
    data_folder_depth
        depth of data folder in source directory
    skip_dirs
        List of globs indicating directories to skip
    select_dirs
        List of globs indicating directories to select. Only directories that match the data folder depth are selected.
        If this is none, all directories are selected.
    index_list
        Optional pre-constructed index of the file system
    show_progress
        Option to show progress

    Returns
    -------
    dirs_list
        Record of data folders with list of files

    """
    dirs_list = []

    if index_list is not None:
        iterator = [[primary_root / item[0], item[1], item[2]] for item in index_list]
    else:
        iterator = os.walk(primary_root)

    for dirpath, dirnames, files in tqdm(iterator, desc="Gathering source directories", disable=not show_progress):
        relpath = Path(dirpath).relative_to(primary_root)

        # Skip dirs take precedence
        skip_dir = False
        if skip_dirs:
            for dir_pattern in skip_dirs:
                if fnmatch.filter([str(relpath)], dir_pattern):
                    skip_dir = True

        if skip_dir:
            continue

        if relpath.parents and len(relpath.parents) == (
                data_folder_depth - 1):  # We are in a data folder.

            # Select dirs only works in data folders. If no select dirs are specified, select all
            if select_dirs:
                for dir_pattern in select_dirs:
                    if fnmatch.filter([str(relpath)], dir_pattern):
                        dirs_list.append((relpath, dirnames, files))
            else:
                dirs_list.append((relpath, dirnames, files))

    return dirs_list


def exception_monitor(func, callback, logger):
    """
    Decorator to catch exceptions from multiprocessing tasks

    Parameters
    ----------
    func
        Function to wrap
    callback
        Callback to run if exception occurs
    logger
        Loguru logger reference needed for logging from multiprocessing

    Returns
    -------

    """

    def exception_monitor_wrapper(dataloc, dataset_config, task_config, source_directory):
        try:
            result = (str(source_directory), func(dataloc, dataset_config, task_config, source_directory))
            error = None
        except Exception as e:
            error = (str(source_directory), e)
            result = None
            callback(source_directory, e, logger)

        return result, error

    functools.update_wrapper(exception_monitor_wrapper, func)
    return exception_monitor_wrapper


def args_unpacker(func):
    """
    Decorator to unpack tuple of arguments from multiprocessing methods such as imap

    Parameters
    ----------
    func
        Function to wrap
    """

    def unpack(args):
        return func(*args)

    functools.update_wrapper(unpack, func)
    return unpack


def apply_to_dataset(func: Callable) -> Callable:
    """
    Decorator to apply a function to each subject in a dataset, with optional multiprocessing.

    Parameters
    ----------
    func
        Function to wrap
    """

    @functools.wraps(func)
    def wrapper_apply_to_dataset(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig,
                                 dirs_list, mpool=None, show_progress=False):

        def exception_warning(source_dir, e, logger, task=task_config.task_name):
            logger.warning(f"Task: {task} resulted in exception for {source_dir}, message: {e}")

        if mpool is None:
            results = []
            errors = []
            for source_directory, _, _ in tqdm(dirs_list, desc=f"Running: {task_config.task_name}",
                                               disable=not show_progress):
                try:
                    result = func(dataloc, dataset_config, task_config, source_directory)
                    results.append((str(source_directory), result))
                except Exception as e:
                    exception_warning(source_directory, e, logger)
                    errors.append((str(source_directory), e))
        else:
            args = ((dataloc, dataset_config, task_config, source_directory) for source_directory, _, _ in
                    dirs_list)
            monitored = exception_monitor(func, exception_warning, logger)
            unpacked = args_unpacker(monitored)
            imap_results = list(
                tqdm(mpool.imap(unpacked, args), desc=f"Running: {task_config.task_name}", disable=not show_progress,
                     total=len(dirs_list)))

            results, errors = list(zip(*imap_results))
            results = [r for r in results if r is not None]
            errors = [e for e in errors if e is not None]

        combined = {"results": results, "errors": errors}
        return combined

    return wrapper_apply_to_dataset


def log_workflow(dataloc: DatasetLocator, cfg: DictConfig, task_config: DictConfig, results):
    """
    Workflow task for logging the result of a workflow.

    Parameters
    ----------
    dataloc
        Dataset locator
    cfg
        DatasetConfig object
    task_config
        Configuration dict for this task
    results
        Results of workflow tasks to log

    """
    output_directory = dataloc.root / task_config.task_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_time = time.strftime('%Y-%m-%dT%H_%M')
    log_path = output_directory / f"workflow_log_{log_time}.json"
    log_path = get_unique_file_name(log_path)

    log_platform = platform.platform()
    log_python_version = sys.version

    distributions = list(importlib.metadata.distributions())
    dependencies = [{"name": dist.name, "version": dist.version} for dist in distributions]

    log = {"log_time": log_time, "platform": log_platform, "python_version": log_python_version,
           "config": OmegaConf.to_container(cfg), "dependencies": dependencies, "results": results}

    with open(log_path, "w") as f:
        try:
            json.dump(log, f, indent=4, default=str)
        except Exception as e:
            logger.error(f"Could not write full log: {e}")
            try:
                log = {"log_time": log_time, "platform": log_platform, "python_version": log_python_version,
                       "config": "Error logging config", "dependencies": dependencies,
                       "results": "Error logging results"}

                json.dump(log, f, indent=4, default=str)
            except Exception as e:
                logger.error(f"Could not write log: {e}")


def get_unique_file_name(path: Path):
    """
    Get a unique filename by adding an addition to the end

    Parameters
    ----------
    path
        Original file name

    Returns
    -------
    path
        New unique file name

    """
    i = 1
    while os.path.exists(path):
        addition = "_" + str(i)
        path = Path(path.parent / f"{path.stem}{addition}{path.suffix}")
        i += 1
        if i > 1000000:
            raise ValueError("Could not find unique filename")

    return path


def print_results(results: dict):
    for task_name, result in results.items():
        if "errors" in result and result["errors"]:
            logger.info(f"{task_name} errors {result['errors']}")
        else:
            logger.info(f"{task_name} completed successfully")


class WorkflowManager:
    """
    A class to manage worflows consisting of a list of tasks to run on a collection of data. This class is designed to
    run on data organized in the SPARC Data Structure (https://doi.org/10.1101/2021.02.10.430563). Units of data to
    operate on are organized in a directory tree.

    initialize() loads the dataset specific configuration from the dataset root directory and gathers a list of directories
    representing units of data to operate on.

    register_task() adds tasks implementing the interface EachItemTask (or other supporeted interfaces) to the list of
    tasks that can be run

    run_workflow() executes the work function of each task on each of the data directories

    log_workflow() writes the results to file along with provenance information (most importantly, the traceable version
    of the code that ran in the workflow)

    """

    def __init__(self, dataset_root: Path, cfg: DictConfig, mpool: Pool = None, show_progress=False):
        self.cfg = cfg
        self.show_progress = show_progress
        self.mpool = mpool
        self.dataloc, self.dataset_config, self.dirs_list = initialize(dataset_root, cfg.tasks.initialize,
                                                                       show_progress)
        self.registered_tasks = {}
        self.results = {}

    def register_task(self, task: Type[EachItemTask]):
        """
        Register a task that can be run by the workflow manager. Registered tasks cannot be duplicates or "initialize"
        or "logging". The task name defines which config dict will be passed to it when it is run.

        Parameters
        ----------
        task
            EachItemTask
        """

        if task().name in [*self.registered_tasks.keys(), "initialize", "logging"]:
            # Because there would be confusion about what is the correct configuration
            raise ValueError("Cannot register a task with the same name as a task already registered")

        if task().name not in self.cfg.tasks.keys():
            # Because we don't know if a task can handle it
            raise ValueError("Cannot register a task with no matching task configuration")

        self.registered_tasks[task().name] = task

    def run_workflow(self, task_list: list[str] = None):
        """
        Run all desired tasks, then log the results. If task list is provided, tasks will be run in that order.
        Otherwise, all tasks will be run in the order they were registered in.

        Parameters
        ----------
        task_list
            Optional list of tasks to run
        """
        # If task list contains duplicates or missing tasks, raise an exception
        self.results = {}

        if len(set(task_list)) != len(task_list):
            # Because the results dict would be overwritten
            raise ValueError("Cannot use a task more than once in a workflow run")

        if any([task not in self.registered_tasks.keys() for task in task_list]):
            raise ValueError("Cannot run a task that is not registered")

        for task in task_list:
            self.results[task] = apply_to_dataset(self.registered_tasks[task].work)(self.dataloc, self.dataset_config,
                                                                                    self.cfg.tasks[task],
                                                                                    self.dirs_list, self.mpool,
                                                                                    self.show_progress)

        log_workflow(self.dataloc, self.cfg, self.cfg.tasks.logging, results=self.results)

        print_results(self.results)
