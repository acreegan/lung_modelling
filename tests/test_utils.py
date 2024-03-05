from lung_modelling import load_with_category
from pathlib import Path
import numpy as np

parent_dir = Path(__file__).parent


def test_load_with_category():
    test_data_dir = parent_dir / "test_data" / "test_load_with_category"

    search_dirs = [test_data_dir / "dir_a", test_data_dir / "dir_b"]

    loaded = load_with_category(search_dirs=search_dirs, category_regex=".+?(?=-)", load_glob="*.txt",
                                loader=lambda t: np.loadtxt(t, dtype=str))

    assert list(loaded.keys()) == ["cat1", "cat2", "cat3"]
    assert str(loaded["cat1"]) == "hello"
