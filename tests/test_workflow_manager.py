from lung_modelling.workflow_manager import gather_directories, DatasetLocator
from pathlib import Path

parent_dir = Path(__file__).parent


def test_gather_directories():
    dirs1 = gather_directories(parent_dir / "test_data" / "test_dir_structure", data_folder_depth=3, skip_dirs=None,
                               select_dirs=None, index_list=None, show_progress=False)

    dirs1 = [dir[0] for dir in dirs1]

    correct_1 = [Path(r"a/c"), Path(r"a/d"), Path(r"b/c"), Path(r"b/d")]
    assert len(correct_1) == len(dirs1)
    for c1 in correct_1:
        assert c1 in dirs1

    dirs2 = gather_directories(parent_dir / "test_data" / "test_dir_structure", data_folder_depth=3, skip_dirs=["*b*"],
                               select_dirs=None, index_list=None, show_progress=False)
    dirs2 = [dir[0] for dir in dirs2]

    correct_2 = [Path(r"a/c"), Path(r"a/d")]
    assert len(correct_2) == len(dirs2)
    for c2 in correct_2:
        assert c2 in dirs2

    dirs3 = gather_directories(parent_dir / "test_data" / "test_dir_structure", data_folder_depth=3, skip_dirs=None,
                               select_dirs=["*b*"], index_list=None, show_progress=False)
    dirs3 = [dir[0] for dir in dirs3]

    correct_3 = [Path(r"b/c"), Path(r"b/d")]
    assert len(correct_3) == len(dirs3)
    for c3 in correct_3:
        assert c3 in dirs3

    dirs4 = gather_directories(parent_dir / "test_data" / "test_dir_structure", data_folder_depth=3, skip_dirs=["*b*"],
                               select_dirs=["*c*"], index_list=None, show_progress=False)
    dirs4 = [dir[0] for dir in dirs4]

    correct_4 = [Path(r"a/c")]
    assert len(correct_4) == len(dirs4)
    for c4 in correct_4:
        assert c4 in dirs4


def test_dataloc():
    dataloc = DatasetLocator(Path("MyDatasetRoot"), Path("MyPrimary"), Path("MyDerivative"), Path("PooledPrimary"), Path("PooledDerivative"))

    assert dataloc.abs_primary == Path("MyDatasetRoot") / "MyPrimary"
    assert dataloc.abs_derivative == Path("MyDatasetRoot") / "MyDerivative"

    long_abs1 = Path("MyDatasetRoot/MyDerivative/a/b/c")
    long_abs2 = Path("MyDatasetRoot/MyPrimary/a/b/c")

    assert dataloc.to_relative(long_abs1) == Path("MyDerivative/a/b/c")
    assert dataloc.to_relative(long_abs2) == Path("MyPrimary/a/b/c")
