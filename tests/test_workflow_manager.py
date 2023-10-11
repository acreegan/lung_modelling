from lung_modelling.workflow_manager import gather_directories, DatasetLocator
from pathlib import Path

parent_dir = Path(__file__).parent


def test_gather_directories():
    dirs1 = gather_directories(parent_dir / "test_data" / "test_dir_structure", data_folder_depth=3, skip_dirs=None,
                               select_dirs=None, index_list=None, show_progress=False)

    correct_1 = [Path(r"a\c"), Path(r"a\d"), Path(r"b\c"), Path(r"b\d")]
    assert len(correct_1) == len(dirs1)
    for c1, d1 in zip(correct_1, [dir[0] for dir in dirs1]):
        assert c1 == d1

    dirs2 = gather_directories(parent_dir / "test_data" / "test_dir_structure", data_folder_depth=3, skip_dirs=["*b*"],
                               select_dirs=None, index_list=None, show_progress=False)

    correct_2 = [Path(r"a\c"), Path(r"a\d")]
    assert len(correct_2) == len(dirs2)
    for c1, d1 in zip(correct_2, [dir[0] for dir in dirs2]):
        assert c1 == d1

    dirs3 = gather_directories(parent_dir / "test_data" / "test_dir_structure", data_folder_depth=3, skip_dirs=None,
                               select_dirs=["*b*"], index_list=None, show_progress=False)

    correct_3 = [Path(r"b\c"), Path(r"b\d")]
    assert len(correct_3) == len(dirs3)
    for c1, d1 in zip(correct_3, [dir[0] for dir in dirs3]):
        assert c1 == d1

    dirs4 = gather_directories(parent_dir / "test_data" / "test_dir_structure", data_folder_depth=3, skip_dirs=["*b*"],
                               select_dirs=["*c*"], index_list=None, show_progress=False)

    correct_4 = [Path(r"a\c")]
    assert len(correct_4) == len(dirs4)
    for c1, d1 in zip(correct_4, [dir[0] for dir in dirs4]):
        assert c1 == d1


def test_dataloc():
    dataloc = DatasetLocator(Path("MyDatasetRoot"), Path("MyPrimary"), Path("MyDerivative"))

    assert dataloc.abs_primary == Path("MyDatasetRoot") / "MyPrimary"
    assert dataloc.abs_derivative == Path("MyDatasetRoot") / "MyDerivative"

    long_abs1 = Path("MyDatasetRoot/MyDerivative/a/b/c")
    long_abs2 = Path("MyDatasetRoot/MyPrimary/a/b/c")

    assert dataloc.to_relative(long_abs1) == Path("MyDerivative/a/b/c")
    assert dataloc.to_relative(long_abs2) == Path("MyPrimary/a/b/c")
