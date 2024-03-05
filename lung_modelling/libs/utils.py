from glob import glob
from pathlib import Path
import re


def load_with_category(search_dirs: list, category_regex: str, load_glob: str, loader: callable) -> dict:
    """
    Load files found in specified search dirs using specified loader function, and assign them to categories based on
    a regex match with their file name.

    Note: If multiple files are found the same category identifier, only the last will be saved.


    Parameters
    ----------
    search_dirs
    category_regex
    load_glob
    loader

    Returns
    -------
    category dict:
        Dict with keys equal to the regex match of the category identifier on the filename. Values equal to the output
        of the loader function.

    """
    all_files = []
    for directory in search_dirs:
        all_files.extend(glob(str(Path(directory) / load_glob)))

    category_dict = {}
    for file in all_files:
        if (cat := re.search(category_regex, str(Path(file).stem))) is not None:
            category_dict[cat.group()] = loader(file)

    return category_dict
