from multiprocess import pool
from loguru import logger
from omegaconf import DictConfig
from lung_modelling.workflow_manager import apply_to_dataset


def flatten(l):
    return [item for sublist in l for item in sublist]


def t_apply_to_dataset():
    """
    This test doesn't work with pytest due to dill.....
    Run manually to check it works
    """
    logger.enable("lung_annotations")

    def mytask(dataset_root, dataset_config, task_config, source_directory):
        if source_directory == "world":
            raise ValueError("Valueerror, saw world")
        else:
            return f"Saw {source_directory}"

    dataset_root = "my_dataset_root"
    dataset_config = "my_dataset_config"
    task_config = DictConfig({"task_name": "my_task"})
    dirs_list = [("hello", "", ""), ("world", "", ""), ("foo", "", "")]
    show_progress = True

    correct_results = [('hello', 'Saw hello'), ('foo', 'Saw foo')]
    correct_errors = [('world', ValueError('Valueerror, saw world'))]

    mytask_dataset = apply_to_dataset(mytask)

    combined_results = mytask_dataset(dataset_root, dataset_config, task_config, dirs_list, None,
                                      show_progress)

    results = combined_results["results"]
    errors = combined_results["errors"]

    for r1, r2 in zip(flatten(correct_results), flatten(results)):
        assert r1 == r2

    for e1, e2 in zip(flatten(correct_errors), flatten(errors)):
        assert str(e1) == str(e2)

    with pool.Pool() as mpool:
        pcombined_results = mytask_dataset(dataset_root, dataset_config, task_config, dirs_list, mpool,
                                           show_progress)

    presults = pcombined_results["results"]
    perrors = pcombined_results["errors"]

    for r1, r2 in zip(flatten(correct_results), flatten(presults)):
        assert r1 == r2

    for e1, e2 in zip(flatten(correct_errors), flatten(perrors)):
        assert str(e1) == str(e2)


if __name__ == "__main__":
    t_apply_to_dataset()
