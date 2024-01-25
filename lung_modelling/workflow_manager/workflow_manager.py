from __future__ import annotations

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
    """
    A helper class to convert paths for SPARC datasets between absolute and relative, and to store primary and
    derivative directories

    """

    def __init__(self, root: Path, rel_primary: Path, rel_derivative: Path, rel_pooled_primary:Path, rel_pooled_derivative: Path):
        self.root = Path(root)
        self.rel_primary = Path(rel_primary)
        self.rel_derivative = Path(rel_derivative)
        self.rel_pooled_primary = Path(rel_pooled_primary)
        self.rel_pooled_derivative = Path(rel_pooled_derivative)

    @property
    def abs_primary(self):
        return self.root / self.rel_primary

    @property
    def abs_derivative(self):
        return self.root / self.rel_derivative

    @property
    def abs_pooled_derivative(self):
        return self.root / self.rel_pooled_derivative

    @property
    def abs_pooled_primary(self):
        return self.root / self.rel_pooled_primary

    def to_relative(self, path: Path) -> Path:
        for parent in path.parents:
            if parent.name == self.rel_primary.name:
                return self.rel_primary / path.relative_to(self.abs_primary)
            elif parent.name == self.rel_derivative.name:
                return self.rel_derivative / path.relative_to(self.abs_derivative)
            elif parent.name == self.rel_pooled_derivative.name:
                return self.rel_pooled_derivative / path.relative_to(self.abs_pooled_derivative)
            elif parent.name == self.rel_pooled_primary.name:
                return self.rel_pooled_primary / path.relative_to(self.abs_pooled_primary)

        raise ValueError("Neither primary nor derivative found in input path")


class Task:
    """
    A class providing an interface for types of workflow task
    """

    __metaclass__ = ABCMeta

    def __init__(self, name: str, config: DictConfig):
        self.name = name
        self.config = config


class EachItemTask(Task):
    """
    A class providing an interface for tasks that should be applied to each item in a list of sources.

    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        """
        A method to be called before multiprocessing to load data required by all items. This should only be used to
        load data to avoid multiple simultaneous file access by the work function. It should not create or modify any files.
        """
        pass

    @staticmethod
    @abstractmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        The work function of the implementation of this class. Defines work to be done on a set of sources specified by
        the source directory parameters. This function should not attempt to access files outside these directories
        because this function will be run in parallel for all workflow sources.

        Results of the work should be saved in the output directory. The return type should be a list of Path objects
        representing the files created.

        Parameters
        ----------
        source_directory_primary
            Absolute path of the source directory in the primary folder of the dataset
        source_directory_derivative
            Absolute path of the source directory in the derivative folder of the dataset
        output_directory
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Config relating to the entire dataset
        task_config
            Task specific config
        initialize_result

        Returns
        -------
        list of Path objects representing the files created.
        """
        pass

    def apply_to_dataset(self, dataloc: DatasetLocator, dataset_config: DictConfig, dirs_list,
                         initialize_result=None, mpool=None, show_progress=False) -> dict:
        """
        Apply work function to each subject in a dataset, with optional multiprocessing.

        Parameters
        ----------
        dataloc
        dataset_config
        dirs_list
        mpool
        show_progress

        Returns
        -------

        """

        def exception_warning(source_dir, e, logger, task=self.name):
            logger.warning(f"Task: {task} resulted in exception for {source_dir}, message: {e}")

        if mpool is None:
            results = []
            errors = []
            for source_directory, _, _ in tqdm(dirs_list, desc=f"Running: {self.name}",
                                               disable=not show_progress):
                try:
                    logger.debug(f"Working on {source_directory}")
                    output_directory = dataloc.abs_derivative / source_directory / self.config.results_directory
                    result = self.work(dataloc.abs_primary / source_directory,
                                       dataloc.abs_derivative / source_directory, output_directory, dataset_config,
                                       self.config, initialize_result)
                    results.append((str(source_directory), result))
                except Exception as e:
                    exception_warning(source_directory, e, logger)
                    errors.append((str(source_directory), e))
        else:
            args = (
                (source_directory, dataloc.abs_primary / source_directory, dataloc.abs_derivative / source_directory,
                 dataloc.abs_derivative / source_directory / self.config.results_directory, dataset_config,
                 self.config, initialize_result) for source_directory, _, _ in dirs_list)
            monitored = exception_monitor(self.work, exception_warning, logger)
            unpacked = args_unpacker(monitored)
            imap_results = list(
                tqdm(mpool.imap(unpacked, args), desc=f"Running: {self.name}", disable=not show_progress,
                     total=len(dirs_list)))

            results, errors = list(zip(*imap_results))
            results = [r for r in results if r is not None]
            errors = [e for e in errors if e is not None]

        relative_files = [str(dataloc.to_relative(Path(result))) if isinstance(result, Path) else result for result in
                          results]
        combined = {"results": relative_files, "errors": errors}
        return combined

    def apply_initialize(self, dataloc: DatasetLocator, dataset_config: DictConfig) -> dict:
        """

        Parameters
        ----------
        dataloc
        dataset_config

        Returns
        -------

        """

        def exception_warning(e, logger, task=self.name):
            logger.warning(f"Task: {task} resulted in exception, message: {e}")

        result = {}
        errors = []
        try:
            result = self.initialize(dataloc, dataset_config, self.config)
        except Exception as e:
            exception_warning(e, logger)
            errors.append((str(self.name), e))

        return {"result": result, "errors": errors}


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

    def exception_monitor_wrapper(source_directory, source_directory_primary: Path, source_directory_derivative: Path,
                                  output_directory: Path, dataset_config: DictConfig, task_config: DictConfig,
                                  initialize_result):
        try:
            result = (str(source_directory),
                      func(source_directory_primary, source_directory_derivative, output_directory, dataset_config,
                           task_config, initialize_result))
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


class AllItemsTask(Task):
    """
    A class providing an interface for tasks that should be applied to all items in a list of sources at once.

    """
    __metaclass___ = ABCMeta

    @staticmethod
    @abstractmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        The work function fo the implementation of this class. Defines work to be done on a set of sources specified by
        the source directory parameters.

        Results of the work should be saved in the output directory. The return type should be a list of Path objects
        representing the files created.

        Parameters
        ----------
        dataloc
            Dataset locator for the dataset
        dirs_list
            List of relative paths to the source directories
        output_directory
            Absolute path of the directory in which to save results of the work function
        dataset_config
            Config relating to the entire dataset
        task_config
            Task specific config

        Returns
        -------
        list of Path objects representing the files created.
        """
        pass

    def apply_to_dataset(self, dataloc: DatasetLocator, dataset_config: DictConfig, dirs_list) -> dict:
        """
        Apply work function to all items in a dataset at once.

        Parameters
        ----------
        dataloc
        dataset_config
        dirs_list

        Returns
        -------

        """

        def exception_warning(e, logger, task=self.name):
            logger.warning(f"Task: {task} resulted in exception, message: {e}")

        results = []
        errors = []
        try:
            output_directory = dataloc.abs_pooled_derivative / self.config.results_directory
            result = self.work(dataloc, dirs_list, output_directory, dataset_config, self.config)
            results.append((str(self.name), result))
        except Exception as e:
            exception_warning(e, logger)
            errors.append((str(self.name), e))

        relative_files = [str(dataloc.to_relative(Path(result))) if isinstance(result, Path) else result for result in
                          results]
        return {"results": relative_files, "errors": errors}


def initialize(dataset_root: Path, task_config: DictConfig, show_progress=True) -> Tuple[
    DatasetLocator, DictConfig, list]:
    """
    Initialization task for processing a dataset directory structure. This is run first when
    WorkflowManager.run_workflow is called. Loads the dataset config and runs gather_directories, which gathers a list
    of source directories which tasks will act on.

    Parameters
    ----------
    dataset_root
        Root directory of the dataset
    task_config
        Configuration dict for this task

        **params**
            **dataset_config_filename**
                Filename for the dataset configuration file. This should be directly inside the dataset_root directory.
            **use_directory_index**
                Option to use pre-build index of the source directory instead of iterating through with os.walk.
            **skip_dirs**:
                List of glob strings to match directories to skip. The whole path relative to the dataset_root is tested,
                so slashes can be included to specify depth to match. This takes precedence over select_dirs
            **select_dirs**:
                List of glob strings to match directories to select. If empty, all valid source directories are selected.
                If not, only valid source directories that match one of these are selected.
    show_progress
        Option to show progress

    Returns
    -------
    dataloc
        DatasetLocator object
    dataset_config
        DatasetConfig object
    dirs_list
        Record of data folders with list of files

    """
    dataset_config_file = dataset_root / task_config["dataset_config_filename"]

    with open(dataset_config_file, "r") as f:
        dataset_config = DictConfig(json.load(f))

    dataloc = DatasetLocator(dataset_root, dataset_config.primary_directory, dataset_config.derivative_directory,
                             dataset_config.pooled_primary_directory, dataset_config.pooled_derivative_directory)

    index_list = None
    if task_config.use_directory_index:
        if len(glob_result := glob.glob(str(dataset_root / dataset_config.directory_index_glob))) > 0:
            index_list = pd.read_csv(glob_result[0], converters={"files": literal_eval}).values.tolist()

    dirs_list = gather_directories(dataloc.abs_primary, dataset_config.data_folder_depth, task_config.skip_dirs,
                                   task_config.select_dirs, index_list, show_progress)
    dirs_list = sorted(dirs_list, key=lambda d: str(d[0]))
    logger.debug("dirs_list:\n" + "".join([f"{dir};\n" for dir in dirs_list]))
    if len(dirs_list) == 0:
        raise ValueError("No valid source directories found")

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


def log_workflow(dataloc: DatasetLocator, cfg: DictConfig, task_config: DictConfig, task_name, results):
    """
    Workflow task for logging the result of a workflow. Runs after all tasks are complete in
    WorkflowManager.run_workflow.

    Parameters
    ----------
    dataloc
        Dataset locator
    cfg
        DatasetConfig object
    task_config
        Configuration dict for this task
        No parameters are currently in use for this task_config
    task_name
        name of this task
    results
        Results of workflow tasks to log

    """
    output_directory = dataloc.root / task_name
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
    """
    Print the results dict to the logger.

    Parameters
    ----------
    results
    """
    for task_name, result in results.items():
        if result is None:
            logger.info(f"No result received from {task_name}")
        if "errors" in result and result["errors"]:
            logger.info(f"{task_name} errors {result['errors']}")
        else:
            logger.info(f"{task_name} completed successfully")


class WorkflowManager:
    """
    A class to manage worflows consisting of a list of tasks to run on a collection of data. This class is designed to
    run on data organized in the SPARC Data Structure (https://doi.org/10.1101/2021.02.10.430563). Units of data to
    operate on are organized in a directory tree.
    """

    def __init__(self, dataset_root: Path, cfg: DictConfig, mpool: Pool = None, show_progress=False):
        self.cfg = cfg
        self.show_progress = show_progress
        self.mpool = mpool
        self.dataloc, self.dataset_config, self.dirs_list = initialize(dataset_root, cfg.tasks.initialize,
                                                                       show_progress)
        self.registered_tasks = {}
        self.results = {}

    def register_task(self, task: EachItemTask | AllItemsTask):
        """
        Register a task that can be run by the workflow manager. Registered tasks cannot be duplicates or "initialize"
        or "logging". The task name defines which config dict will be passed to it when it is run.

        Parameters
        ----------
        task
            EachItemTask
        """

        if task.name in [*self.registered_tasks.keys(), "initialize", "logging"]:
            # Because there would be confusion about what is the correct configuration
            raise ValueError(
                f"Error registering {task.name}. Cannot register a task with the same name as a task already registered")

        if task.name not in self.cfg.tasks.keys():
            # Because we don't know if a task can handle it
            raise ValueError(
                f"Error registering {task.name}. Cannot register a task with no matching task configuration")

        # Todo
        # Also check if the output directory is the same as an existing one

        self.registered_tasks[task.name] = task

    def run_workflow(self):
        """
        Run all tasks, then log the results. tasks will be run in the order they were registered in.

        """
        self.results = {}

        for task in self.registered_tasks.values():
            logger.info(f"Running {task.name}")
            if isinstance(task, EachItemTask):

                initialize_result = task.apply_initialize(self.dataloc, self.dataset_config)

                work_result = task.apply_to_dataset(self.dataloc, self.dataset_config, self.dirs_list,
                                                    initialize_result["result"], self.mpool, self.show_progress)

                self.results[task.name] = {"results": work_result["results"], "errors": [*initialize_result["errors"],
                                                                                         *work_result["errors"]]}

            elif isinstance(task, AllItemsTask):
                self.results[task.name] = task.apply_to_dataset(self.dataloc, self.dataset_config, self.dirs_list)
            else:
                raise ValueError(f"Error executing {task.name}. Unrecognized task type")

        log_workflow(self.dataloc, self.cfg, self.cfg.tasks.logging, "logging", results=self.results)

        print_results(self.results)
