from lung_modelling import load_with_category
from pathlib import Path
import numpy as np

parent_dir = Path(__file__).parent


def test_load_with_category():
    test_data_dir = parent_dir / "test_data" / "test_load_with_category"

    search_dirs = [test_data_dir / "dir_a", test_data_dir / "dir_b"]

    loaded = load_with_category(search_dirs=search_dirs, category_regex=".+?(?=-)", load_glob="*.txt",
                                loader=lambda t: np.loadtxt(t, dtype=str))

    assert np.all([key in list(loaded.keys()) for key in ["cat1", "cat2", "cat3"]]) # Not checking order
    assert str(loaded["cat1"]) == "hello"
