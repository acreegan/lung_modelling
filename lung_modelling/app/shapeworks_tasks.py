from lung_modelling.workflow_manager import EachItemTask, DatasetLocator
from pathlib import Path
from omegaconf import DictConfig
import os
import shapeworks as sw
from glob import glob
import pyvista as pv


class SmoothLungLobesSW(EachItemTask):

    @property
    def name(self):
        return "smooth_lung_lobes_sw"

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig) -> list[Path]:
        """
        Pre-process lung lobe images by applying antialiasing using Shapeworks libraries

        Parameters
        ----------
        source_directory_primary
            Absolute path of the source directory in the primary folder of the dataset
        source_directory_derivative
            Absolute path of the source directory in the derivative folder of the dataset
        output_directory
            Directory in which to save results of the work
        dataset_config
            Config relating to the entire dataset
        task_config
            **results_directory**: subdirectory for results

            **output_filenames**: dict providing a mapping from lobe mapping (in dataset config) to output filenames

            **params**: (Dict)
                **maximumRMSError**, **numberOfIterations**:
                    Parameters to apply to SimpleITK.AntiAliasBinary

        Returns
        -------
        smoothed_lobes
            list of Path objects representing the files created.
        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file = glob(str(source_directory_primary / dataset_config.lung_image_glob))[0]
        shape_seg = sw.Image(image_file)

        suffix = Path(image_file).suffix

        smoothed_lobes = []
        for lobe, name in task_config.output_filenames.items():
            s = shape_seg.copy()  # Copy because extractLabel is destructive
            lobe_image = s.extractLabel(dataset_config.lobe_mapping[lobe])
            iso_spacing = [1, 1, 1]
            lobe_image.antialias(task_config.params.numberOfIterations, task_config.params.maximumRMSError).resample(
                iso_spacing, sw.InterpolationType.Linear).binarize()

            filename = f"{str(output_directory / name)}{suffix}"
            lobe_image.write(filename)
            smoothed_lobes.append(Path(filename))

        return smoothed_lobes


class CreateMeshesSW(EachItemTask):

    @property
    def name(self):
        return "create_meshes_sw"

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig) -> list[Path]:
        """
        Convert medical image files to meshes and apply smoothing using Shapeworks libraries.

        Parameters
        ----------
        source_directory_primary
            Absolute path of the source directory in the primary folder of the dataset
        source_directory_derivative
            Absolute path of the source directory in the derivative folder of the dataset
        output_directory
            Directory in which to save results of the work
        dataset_config
            Config relating to the entire dataset
        task_config
            **source_directory**: subdirectory within derivative source folder to find source files

            **results_directory**: subdirectory for results

            **params**: (Dict)

                **remesh**, **remesh_percentage**, **adaptivity**:
                    Option to remesh and parameters for shapeworks remesh
                **smooth**, **smooth_iterations**, **relaxation**:
                    Option to smooth and parameters for shapeworks smooth
                **fill_holes**, **hole_size**:
                    Option to fill holes nad parameters for shapeworks fill_holes}

        Returns
        -------
        mesh_files
            list of Path objects representing the files created.
        """
        params = task_config.params

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_files = glob(str(source_directory_derivative / task_config.source_directory / "*"))

        if len(image_files) == 0:
            raise RuntimeError("No files found")

        mesh_files = []
        for image_file in image_files:
            image_data = sw.Image(image_file)
            mesh = image_data.toMesh(1)  # Seems to get the mesh in the right orientation without manual realignment

            # Shapeworks remeshing uses ACVD (vtkIsotropicDiscreteRemeshing). Should be the same as pyACVD with pyvista
            mesh_rm = mesh.remeshPercent(percentage=params.remesh_percentage,
                                         adaptivity=params.adaptivity) if params.remesh else mesh
            # Shapeworks smooth uses vtkSmothPolyData. Should be the same as pyVista
            mesh_sm = mesh_rm.smooth(iterations=params.smooth_iterations,
                                     relaxation=params.relaxation) if params.smooth else mesh_rm
            # Shapeworks fillHoles uses vtkFillHolesFilter. Should be the same as pyVista
            mesh_fh = mesh_sm.fillHoles(hole_size=params.hole_size) if params.fill_holes else mesh_sm

            output_filename = str(output_directory / Path(image_file).stem) + '.vtk'
            mesh_fh.write(output_filename)
            mesh_files.append(Path(output_filename))

        return mesh_files


all_tasks = [SmoothLungLobesSW, CreateMeshesSW]
