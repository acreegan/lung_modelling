from pathlib import Path
from lung_modelling.app.shapeworks_tasks import ExtractWholeLungsSW
from lung_modelling.workflow_manager import DatasetLocator
import pyvista as pv
import tempfile
from omegaconf import DictConfig
import shapeworks as sw
import numpy as np
from glob import glob

parent_dir = Path(__file__).parent


def test_smooth_whole_lungs_sw():
    voxel_matrix = np.array([
        [[1, 1, 2, 2, ],
         [1, 1, 2, 2, ],
         [1, 1, 2, 2, ],
         [1, 1, 2, 2, ], ],

        [[1, 1, 2, 2, ],
         [1, 1, 2, 2, ],
         [1, 1, 2, 2, ],
         [1, 1, 2, 2, ], ],
    ])


    voxel_matrix = np.pad(voxel_matrix, 1)
    voxel_matrix = np.kron(voxel_matrix, np.ones((2, 2, 2)))


    # Create SPARC data structure in temp files
    with tempfile.TemporaryDirectory() as root, \
            tempfile.TemporaryDirectory(dir=root) as derivative, \
            tempfile.TemporaryDirectory(dir=derivative) as data_dir:
        dataloc = DatasetLocator(Path(root), rel_primary=Path(derivative).relative_to(root),
                                 rel_derivative=Path(derivative).relative_to(root), rel_pooled_derivative="")

        image = sw.Image(voxel_matrix.astype(np.float32))
        # Save meshes in directory structure and create configs
        image.write(str(dataloc.abs_derivative / Path(data_dir).stem / "image.nii"))

        dir = (Path(data_dir).stem, "", "")
        output_directory = "."
        task_config = DictConfig({"output_filenames": {"left_lung": ["lul","lll"]},
                                  "params": {"maximumRMSError": 0.009999999776482582, "numberOfIterations": 30}
                                  })

        datset_config = DictConfig({
            "lung_image_glob": "*.nii",
            "lobe_mapping": {"lul": 1, "lll": 2}
        })

        results = ExtractWholeLungsSW.work(source_directory_primary=dataloc.abs_primary / dir[0],
                                          source_directory_derivative=dataloc.abs_derivative / dir[0],
                                          output_directory=dataloc.abs_derivative / dir[0] / output_directory,
                                          dataset_config=datset_config,
                                          task_config=task_config,
                                          initialize_result=None)

        # Load results and apply transform to original meshes
        loaded_results = []
        for result in results:
            result = sw.Image(glob(f"{result}*")[0])
            loaded_results.append(result)

        n_pixels_original = sum(np.logical_and(voxel_matrix, voxel_matrix).astype(int).ravel())
        n_pixels_smoothed = sum(loaded_results[0].toArray(copy=True).ravel())

        # If lobes are incorrectly merged, there will be a big hole. But resampling can decrease the pixels by quite a
        # lot. So just check it's over 2/3 of the original
        assert n_pixels_smoothed > n_pixels_original * 0.66
