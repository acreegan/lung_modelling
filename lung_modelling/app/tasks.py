import math

from lung_modelling.workflow_manager import EachItemTask, DatasetLocator, AllItemsTask
from pathlib import Path, PurePosixPath
from omegaconf import DictConfig
import os
from glob import glob
from lung_modelling import extract_section, voxel_to_mesh, refine_mesh, parse_discrete, load_with_category
import medpy.io
import SimpleITK as sitk
import numpy as np
import pyacvd
import cc3d
from scipy import ndimage
import pyvista as pv
import pandas as pd
import csv
from medpy.io import load
import fnmatch
from tetrahedralizer.mesh_lib import preprocess_and_tetrahedralize
import re
from matplotlib import colormaps
from pyeit.mesh.wrapper import PyEITMesh
from pyeit.mesh.external import place_electrodes_3d
import pyeit.eit.protocol as protocol
from pyeit.eit.jac import JAC
from pyeit.eit.fem import EITForward
from pyeit.visual.plot import create_3d_plot_with_slice
from vtkmodules.util.vtkConstants import VTK_TETRA
import datetime

class ExtractLungLobes(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Pre-process lung lobe segmentation by extracting lobe labels and applying antialiasing.

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
            **results_directory**: Name of the results folder (Stem of output_directory)

            **output_filenames**: dict providing a mapping from lobe mapping (in dataset config) to output filenames

            **params**: (Dict)
                **maximumRMSError**, **numberOfIterations**:
                    Parameters to apply to SimpleITK.AntiAliasBinary

        Returns
        -------
        smoothed_files
            list of Path objects representing the files created.
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file = glob(str(source_directory_primary / dataset_config.lung_image_glob))[0]
        image_data, header = medpy.io.load(image_file)

        suffix = Path(image_file).suffix

        smoothed_files = []
        for lobe, name in task_config.output_filenames.items():
            lobe_array = extract_section(image_data, dataset_config.lobe_mapping[lobe])

            im = sitk.GetImageFromArray(lobe_array)
            im_aa = sitk.AntiAliasBinary(im, maximumRMSError=task_config.params.maximumRMSError,
                                         numberOfIterations=task_config.params.numberOfIterations)
            im_rs = sitk.Resample(im_aa)
            im_bin = sitk.BinaryThreshold(im_rs)
            lobe_array = np.array(sitk.GetArrayFromImage(im_bin))

            output_filename = f"{str(output_directory / task_config.output_filenames[lobe])}{suffix}"
            medpy.io.save(lobe_array, output_filename, hdr=header, use_compression=True)
            smoothed_files.append(Path(output_filename))

        return smoothed_files


class ExtractWholeLungs(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Pre-process lung segmentation by extracting whole lung and applying antialiasing.

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
            **results_directory**: Name of the results folder (Stem of output_directory)

            **output_filenames**: dict providing a mapping from lobe mapping (in dataset config) to output filenames

            **params**: (Dict)
                **maximumRMSError**, **umberOfIterations**:
                    Parameters to apply to SimpleITK.AntiAliasBinary

        Returns
        -------
        smoothed_files
            list of Path objects representing the files created.
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_file = glob(str(source_directory_primary / dataset_config.lung_image_glob))[0]
        image_data, header = medpy.io.load(image_file)

        suffix = Path(image_file).suffix

        smoothed_files = []
        for name, lobes in task_config.output_filenames.items():
            lobe_arrays = [extract_section(image_data, dataset_config.lobe_mapping[lobe]) for lobe in lobes]
            merged = lobe_arrays[0]
            for array in lobe_arrays:
                merged = np.logical_or(merged, array).astype(int)

            im = sitk.GetImageFromArray(merged)
            im_aa = sitk.AntiAliasBinary(im, maximumRMSError=task_config.params.maximumRMSError,
                                         numberOfIterations=task_config.params.numberOfIterations)
            im_rs = sitk.Resample(im_aa)
            im_bin = sitk.BinaryThreshold(im_rs)
            lobe_array = np.array(sitk.GetArrayFromImage(im_bin))

            output_filename = f"{str(output_directory / name)}{suffix}"
            medpy.io.save(lobe_array, output_filename, hdr=header, use_compression=True)
            smoothed_files.append(Path(output_filename))

        return smoothed_files


class CreateMeshes(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Convert medical image files to meshes and apply smoothing.

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
            **source_directory**: subdirectory within derivative source folder to find source files

            **results_directory**: Name of the results folder (Stem of output_directory)

            **params**: (Dict)
                **n_iter**, **feature_smoothing**, **edge_angle**, **feature_angle**, **relaxation_factor**:
                    Params for pyvista smooth

                **target_reduction**, **volume_preservation**:
                    Params for pyvista decimate

                **hole_size**:
                    Param for pyvista fill_holes

                **fix_mesh**:
                    Option to fix mesh


        Returns
        -------
        mesh_files
            list of Path objects representing the files created.
        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image_files = glob(str(source_directory_derivative / task_config.source_directory / "*"))

        if len(image_files) == 0:
            raise RuntimeError("No files found")

        mesh_files = []
        for image_file in image_files:
            image_data, header = medpy.io.load(image_file)
            # id = cc3d.dust(image_data, threshold=100)
            mesh = voxel_to_mesh(image_data, spacing=header.spacing, direction=header.direction, offset=header.offset)

            refined_mesh = refine_mesh(mesh, params=task_config.params)

            output_filename = str(output_directory / Path(image_file).stem) + '.stl'
            refined_mesh.save(output_filename)
            mesh_files.append(Path(output_filename))

        return mesh_files


class ReferenceSelectionMesh(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: Path, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        A task to load all meshes at once so the shape closest to the mean can be found and selected as the reference


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
        reference
            Mesh selected as the reference

        """
        raise NotImplementedError("Reference selection mesh not yet implemented for non shapeworks libs")


class ExtractTorso(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Extract a torso image from a set of dicom files. Segmented torso is saved in a .nii file

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
            **source_directory**: subdirectory for dicom files

            **results_directory**: Name of the results folder (Stem of output_directory)

            **output_filename**: filename for torso image (not including extension)

            **params**: (Dict)
                **threshold**
                    threshold for segmenting torso from dicom image

        Returns
        -------
        [output_filename]
            single item list containing the output filename of the torso image
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        image, header = medpy.io.load(str(source_directory_primary / task_config.source_directory))

        binary_image = np.array(image > task_config.params.threshold, dtype=np.int8)

        max_connectivity = cc3d.largest_k(binary_image, k=1)
        image_filled_holes = max_connectivity.copy()
        for i in range(max_connectivity.shape[-1]):
            image_filled_holes[:, :, i] = ndimage.binary_fill_holes(max_connectivity[:, :, i])

        output_filename = f"{str(output_directory / task_config.output_filename)}.nii"
        medpy.io.save(image_filled_holes, output_filename, hdr=header, use_compression=True)

        return [Path(output_filename)]


class MeshLandmarksCoarse(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        pass

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Calculate points spaced roughly equidistantly around the surface of a mesh. This should help get
        good coverage in optimization without needing to lower initial relative weighting as much.

        Points are placed by tracing a ray from the mesh center of mass to the vertices of a cube. This results in
        8 landmarks. The cube is oriented such that two of its vertices point straight up and down.

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
            **source_directory**: subdirectory within derivative source folder to find source files

            **results_directory**: Name of the results folder (Stem of output_directory)

            **params**: (Dict): No params currently used for this task

        initialize_result
            Return dict from the initialize function

        Returns
        -------
        List of mesh landmark filenames. These take the same form as shapeworks particles files

        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        mesh_files = glob(str(source_directory_derivative / task_config.source_directory / "*"))

        if len(mesh_files) == 0:
            raise RuntimeError("No files found")

        meshes = []
        for mesh_file in mesh_files:
            meshes.append(pv.read(mesh_file))

        mesh_landmark_filenames = []
        for mesh, file in zip(meshes, mesh_files):
            com = mesh.center_of_mass()
            bounds = np.array(mesh.bounds).reshape(3, -1)

            longest_side = np.max(bounds[:, 1] - bounds[:, 0])

            cube = pv.Box(quads=True).scale([longest_side, longest_side, longest_side])
            cube = cube.rotate_x(45).rotate_y(35.264)
            cube = cube.translate(com - cube.center_of_mass())

            mesh_landmarks = []
            for point in cube.points:
                landmark = mesh.ray_trace(com, point)[0][-1]
                mesh_landmarks.append(landmark)

            mesh_landmark_filename = output_directory / f"{str(Path(file).stem)}_landmarks.particles"
            np.savetxt(str(mesh_landmark_filename), mesh_landmarks, delimiter=" ")
            mesh_landmark_filenames.append(mesh_landmark_filename)

        return mesh_landmark_filenames


class ParseCOPDGeneSubjectGroups(AllItemsTask):
    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        A task to parse subject data from the COPDGene dataset so that it can be added to a shapeworks style config file.

        This is an AllItemsTask even though it involves parsing each individual subject separately because we don't
        want pass the large subject data array to lots of duplicate threads. Also the parsing should not take long.


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
            **source_directory**
                subdirectory within derivative source folder to find source files
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **subject_data_filename**
                filename for COPDGene subject data file
            **subject_data_dict_filename**
                filename for COPDGene subject data dict file
            **groups**
                list of group labels to parse
            **params**: (Dict):
                No params currently used for this task


        Returns
        -------
        parsed_data
            File containing parsed groups

        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        data_file = dataloc.abs_pooled_primary / task_config.source_directory / task_config.subject_data_filename
        data = pd.read_csv(data_file, sep="\t")

        data_dict_file = dataloc.abs_pooled_primary / task_config.source_directory / task_config.subject_data_dict_filename
        data_dict = pd.read_excel(data_dict_file)
        data_dict = data_dict.set_index("VariableName")

        group_mappings = {group: parse_discrete(data_dict["CodedValues"][group]) for group in task_config.groups}

        group_data_header = ["dir", *task_config.groups]
        group_values = []
        for dir, _, _ in dirs_list:
            sid = dir.parts[
                dataset_config.subject_id_folder_depth - 2]  # -1 because dirs start after primary/derivative, -1 again to zero index
            subject_group_values = []
            subject_group_values.append(str(PurePosixPath(dir)))
            for group in task_config.groups:
                raw_value = data.loc[data.sid == sid][group].values[0]
                mapped_value = group_mappings[group][raw_value]
                subject_group_values.append(mapped_value)

            group_values.append(subject_group_values)

        group_data_filename = output_directory / "group_data.csv"
        with open(str(group_data_filename), "w") as ref_dir_file:
            writer = csv.writer(ref_dir_file)
            writer.writerow(group_data_header)

            for row in group_values:
                writer.writerow(row)

        return [Path(group_data_filename)]


class SelectCOPDGeneSubjectsByValue(AllItemsTask):
    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Selects subjects from the COPDGene dataset by value using a dict of key value pairs representing columns and
        values in the subject data file. If multiple columns are specified, only subjects which match all values are
        selected.

        Selections are also filtered by whether CT data is present, and which ventilation state is desired (
        End inspiratory (INSP), or end expiratory (EXP))

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
            **source_directory**
                subdirectory within derivative source folder to find source files
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **subject_data_filename**
                filename for COPDGene subject data file
            **subject_data_dict_filename**
                filename for COPDGene subject data dict file
            **search_values**
                Dict of key value pairs
            **insp_exp**:
                ventilation state to select. options: "INSP", or "EXP"
            **params**: (Dict):
                No params currently used for this task


        Returns
        -------
        List containing the output file path
            Format: csv
            Columns: sid, dirpath
        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        data_file = dataloc.abs_pooled_primary / task_config.source_directory / task_config.subject_data_filename
        data = pd.read_csv(data_file, sep="\t")

        dirpaths = [item[0] for item in dirs_list]
        dirpaths = pd.DataFrame(
            data=np.array([[path.parts[dataset_config.subject_id_folder_depth - 2] for path in dirpaths], dirpaths]).T,
            columns=["sid", "dirpath"])

        # Filter for desired values in the subject data
        # -------------------------------------------------------------------------------------------------------------
        # Filter desired study phase
        selected = data.copy()
        selected = selected.loc[selected["Phase_study"] == task_config.study_phase]

        # Filter desired column values
        for data_column, value in dict(task_config.search_values).items():
            selected = selected.loc[selected[data_column] == value]

        # Filter for desired existing data
        for column in task_config.data_columns_exist:
            selected = selected.loc[selected[column].notna()]

        # Filter selected subjects checking that data exists and is labelled as good
        # -------------------------------------------------------------------------------------------------------------
        # Make sure we are only using subjects that are present in the filesystem
        selected = selected.loc[selected.sid.isin(dirpaths.sid)]
        # Only use subjects that have good CT data
        selected = selected[selected.CTMissing_Reason == 0]

        # Select directories based on selected subjects and filter selected directories by kernel as specified in
        # subject data (This should be the one with pre-segmented lobes and airways
        # --------------------------------------------------------------------------------------------------------------
        # Map kernel column of subject data to the folder naming convention in the dataset
        selected["kernel_label"] = ["STD" if l == "STANDARD" else l for l in selected.kernel]
        selected = selected.set_index("sid")
        dirpaths_selected = dirpaths.loc[dirpaths.sid.isin(selected.index)]
        dirpaths_selected = dirpaths_selected[[len(fnmatch.filter([i], f"*_{selected.loc[j].kernel_label}_*")) > 0
                                               for i, j in zip(dirpaths_selected.dirpath, dirpaths_selected.sid)]]

        # Final check that desired data is present in the file system
        # --------------------------------------------------------------------------------------------------------------
        # All directories that match the kernel label of a subject should have Lobes.mhd file
        dirpaths_selected = dirpaths_selected[[len(glob(str(Path(dataloc.abs_primary / i) / "*Lobes.mhd"))) > 0
                                               for i in dirpaths_selected.dirpath]]

        # Filter selected directories using desired ventilation state
        # --------------------------------------------------------------------------------------------------------------
        dirpaths_selected = dirpaths_selected[
            [len(fnmatch.filter([i], f"*_{task_config.insp_exp}_*")) > 0 for i in dirpaths_selected.dirpath]]

        # Remove "LD" (Low dose?) folders
        # --------------------------------------------------------------------------------------------------------------
        dirpaths_selected = dirpaths_selected[
            [len(fnmatch.filter([i], f"*_LD_*")) == 0 for i in dirpaths_selected.dirpath]]

        # Write Results
        # ---------------------------------------------------------------------------------------------------------------
        selected_subjects_filename = output_directory / "selected_subjects.csv"
        with open(str(selected_subjects_filename), "w") as ref_dir_file:
            writer = csv.writer(ref_dir_file)
            writer.writerow(["sid", "dirpath"])

            for row in range(len(dirpaths_selected)):
                writer.writerow([dirpaths_selected.iloc[row].sid, PurePosixPath(dirpaths_selected.iloc[row].dirpath)])

        return [Path(selected_subjects_filename)]


class FormatSubjects(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Format a list of subjects. Input data is filtered by key value pairs as in SelectCOPDGeneSubjectsByValue.
        Output is a csv file with rows consisting of elements from the input data column formatted with a prefix and
        suffix. If multiple formats are specified, each is an additional column in the output file, using the same input
        cell. Delimiter and newline characters are configurable.

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
            **source_directory**
                subdirectory within derivative source folder to find source files
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **input_file_glob**:
                glob pattern for input file
            **column**
                column to use as subject ids to format
            **search_values**:
                Dict of key value pairs
            **formats**
                list of dicts specifying prefix and suffix
            **delimiter**
                delimiter to use between formats
            **newline**
                newline to separate formatted items
            **params**: (Dict):
                No params currently used for this task


        Returns
        -------
        List containing the output file path
        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        data_file = \
            glob(str(dataloc.abs_pooled_derivative / task_config.source_directory / task_config.input_file_glob))[0]
        data = pd.read_csv(data_file)

        match = data.copy()
        for key, value in dict(task_config.search_values).items():
            match = match.loc[match[key] == value]

        formatted_subjects_filename = output_directory / "formatted_subjects.csv"
        with open(str(formatted_subjects_filename), "w", newline=task_config.newline) as f:
            writer = csv.writer(f, delimiter=task_config.delimiter)

            for row in match[task_config.column]:
                writer.writerow(["{}{}{}".format(fmt.prefix, row, fmt.suffix) for fmt in task_config.formats])

        return [Path(formatted_subjects_filename)]


class InspectMeshes(AllItemsTask):

    @staticmethod
    def work(dataloc: DatasetLocator, dirs_list: list, output_directory: Path, dataset_config: DictConfig,
             task_config: DictConfig) -> list[Path]:
        """
        Display groups of meshes, allowing the user to select from them. The selected and non-selected directories are
        recorded in an output file. This allows manual inspection for exclusion, for example of torso scans that are
        incomplete.

        If the target output file already exists, it is used to set the initial selection state of any mesh that is in
        the current list. Note: using the output file as input is normally dangerous and not recommended because it
        breaks idempotency. It is used in this case since this is a user input task (we can't rely on the user to behave
        the same way multiple times anyway). This allows the user to review past selections.

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
            **source_directory**
                subdirectory within derivative source folder to find source files
            **results_directory**:
                Name of the results folder (Stem of output_directory)
            **params**: (Dict):
                No params currently used for this task


        Returns
        -------
        List containing the output file path
            Format: csv
            Columns: dirpath, selected
        """

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        df_dirs = pd.DataFrame(data=np.array([[d[0] for d in dirs_list], [False] * len(dirs_list)]).T,
                               columns=["dirpath", "selected"])

        inspected_dirs_filename = output_directory / "inspected_dirs.csv"
        if len(glob(str(inspected_dirs_filename))) > 0:
            inspected_dirs_previous = pd.read_csv(inspected_dirs_filename)
            previous_selected = inspected_dirs_previous[inspected_dirs_previous.selected == True]
            df_dirs.selected[df_dirs.dirpath.isin(previous_selected.dirpath.apply(Path))] = True

        n_per_window = 9
        n_windows = math.ceil(len(dirs_list) / n_per_window)
        for window_index, i in enumerate(range(0, len(df_dirs), n_per_window)):
            shape = (3, 3)
            p = pv.Plotter(shape=shape)

            for j in range(n_per_window):
                if i + j >= len(df_dirs):
                    break

                dir = df_dirs.dirpath.iloc[i + j]
                p.subplot(*np.unravel_index(j, shape))
                file = glob(str(dataloc.abs_derivative / dir / task_config.source_directory / "*"))[0]

                mesh = pv.read(file)

                if i == 0:
                    ref_mesh = mesh
                else:
                    mesh = mesh.align(ref_mesh)

                class update_selected:
                    def __init__(self, dir):
                        self.dir = dir

                    def __call__(self, state):
                        df_dirs.selected[df_dirs.dirpath == Path(self.dir)] = state
                        if state:
                            print(f"Selected: {self.dir}")
                        else:
                            print(f"Deselected: {self.dir}")

                p.add_mesh(mesh)
                p.add_title(Path(dir).stem, font_size=8)
                p.add_checkbox_button_widget(update_selected(dir), value=df_dirs.selected[df_dirs.dirpath == Path(dir)])

            p.link_views()
            p.show(title=f"Window {window_index + 1}/{n_windows}")

        df_dirs.dirpath = df_dirs.dirpath.apply(PurePosixPath)
        df_dirs.to_csv(inspected_dirs_filename, index=False)

        return [Path(inspected_dirs_filename)]


class TetrahedralizeMeshes(EachItemTask):

    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:

        if not os.path.exists(dataloc.abs_pooled_derivative / task_config.results_directory):
            os.makedirs(dataloc.abs_pooled_derivative / task_config.results_directory)

        mean_mesh_dict = load_with_category(search_dirs=[dataloc.abs_pooled_derivative /
                                                         task_config.source_directory_mean_mesh],
                                            category_regex=task_config.mesh_file_domain_name_regex,
                                            load_glob="*.vtk",
                                            loader=pv.read)

        # TODO: We should just change the params in create meshes, not do this here
        if "remesh" in task_config.params:
            for k, v in mean_mesh_dict.items():
                clus = pyacvd.Clustering(v)
                clus.cluster(task_config.params["remesh"])
                remesh = clus.create_mesh()

                clus2 = pyacvd.Clustering(remesh)
                clus2.subdivide(3)
                clus2.cluster(task_config.params["remesh"])
                remesh2 = clus2.create_mesh()

                mean_mesh_dict[k] = remesh2

        outer_mesh_label = task_config.outer_mesh_domain_name
        inner_mesh_labels = [name for name in mean_mesh_dict.keys() if name != task_config.outer_mesh_domain_name]

        mean_tet = preprocess_and_tetrahedralize(outer_mesh=mean_mesh_dict[task_config.outer_mesh_domain_name],
                                                 inner_meshes=[mean_mesh_dict[key] for key in inner_mesh_labels],
                                                 mesh_repair_kwargs=task_config.params.mesh_repair_kwargs,
                                                 gmsh_options=task_config.params.gmsh_options,
                                                 outer_mesh_element_label=outer_mesh_label,
                                                 inner_mesh_element_labels=inner_mesh_labels)

        mean_tet_file = dataloc.abs_pooled_derivative / task_config.results_directory / "mean_meshes_tetrahedralized.vtu"
        mean_tet.save(mean_tet_file)

        return mean_mesh_dict

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Convert surface meshes to 3D finite element meshes suitable for EIT using tetrahedralizer.

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
            **source_directory**: subdirectory within derivative source folder to find source files

            **results_directory**: Name of the results folder (Stem of output_directory)

            **params**: (Dict): No params currently used for this task

        initialize_result
            Return dict from the initialize function

        Returns
        -------


        """

        # For each set of input folders (e.g., From CT with lobes, From PCA estimated)
        # Outer and inner surfaces should be specified
        # From CT  is the reference for simulation
        # From PCA, specify a set with and without lungs (for a priori forward, and reconstruction respectively
        # Run tetrahedralizer on them.

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        reference_mesh_dict = load_with_category(search_dirs=[source_directory_derivative / d for d in
                                                              task_config.source_directories_reference_mesh],
                                                 category_regex=task_config.mesh_file_domain_name_regex,
                                                 load_glob="*.vtk",
                                                 loader=pv.read)

        predicted_mesh_dict = load_with_category(search_dirs=[source_directory_derivative /
                                                              task_config.source_directory_predicted_mesh],
                                                 category_regex=task_config.mesh_file_domain_name_regex,
                                                 load_glob="*.vtk",
                                                 loader=pv.read)

        mean_mesh_dict = initialize_result

        # TODO: We should just change the params in create meshes, not do this here
        if "remesh" in task_config.params:
            for d in [reference_mesh_dict, predicted_mesh_dict]:
                for k, v in d.items():
                    clus = pyacvd.Clustering(v)
                    clus.cluster(task_config.params["remesh"])
                    remesh = clus.create_mesh()

                    clus2 = pyacvd.Clustering(remesh)
                    clus2.subdivide(3)
                    clus2.cluster(task_config.params["remesh"])
                    remesh2 = clus2.create_mesh()

                    d[k] = remesh2

        # cmap = colormaps["Set1"]
        # p = pv.Plotter(shape=(1, 3))
        # titles = ["Reference Meshes", "Predicted Meshes", "Mean Meshes"]
        # for i, d in enumerate([reference_mesh_dict, predicted_mesh_dict, mean_mesh_dict]):
        #     p.subplot(0, i)
        #     for j, k in enumerate(reference_mesh_dict.keys()):
        #         p.add_mesh(d[k].extract_all_edges(), color=cmap(j), label=k)
        #     p.add_legend()
        #     p.add_text(titles[i])
        #
        # p.link_views()
        # p.show()

        outer_mesh_label = task_config.outer_mesh_domain_name
        inner_mesh_labels = [name for name in reference_mesh_dict.keys() if name != task_config.outer_mesh_domain_name]
        ref_tet = preprocess_and_tetrahedralize(outer_mesh=reference_mesh_dict[task_config.outer_mesh_domain_name],
                                                inner_meshes=[reference_mesh_dict[key] for key in inner_mesh_labels],
                                                mesh_repair_kwargs=task_config.params.mesh_repair_kwargs,
                                                gmsh_options=task_config.params.gmsh_options,
                                                outer_mesh_element_label=outer_mesh_label,
                                                inner_mesh_element_labels=inner_mesh_labels)

        pred_tet = preprocess_and_tetrahedralize(outer_mesh=predicted_mesh_dict[task_config.outer_mesh_domain_name],
                                                 inner_meshes=[predicted_mesh_dict[key] for key in inner_mesh_labels],
                                                 mesh_repair_kwargs=task_config.params.mesh_repair_kwargs,
                                                 gmsh_options=task_config.params.gmsh_options,
                                                 outer_mesh_element_label=outer_mesh_label,
                                                 inner_mesh_element_labels=inner_mesh_labels)

        ref_tet_file = output_directory / "reference_meshes_tetrahedralized.vtu"
        pred_tet_file = output_directory / "predicted_meshes_tetrahedralized.vtu"

        ref_tet.save(ref_tet_file)
        pred_tet.save(pred_tet_file)


# Todo
# Tetrahedralize mean separately.. as it only needs doing once


class EITSimulation(EachItemTask):
    @staticmethod
    def initialize(dataloc: DatasetLocator, dataset_config: DictConfig, task_config: DictConfig) -> dict:
        mean_mesh = pv.read(glob(str(dataloc.abs_pooled_derivative / task_config.source_directory / "*mean*.vtu"))[0])

        return {"mean_mesh": mean_mesh}

    @staticmethod
    def work(source_directory_primary: Path, source_directory_derivative: Path, output_directory: Path,
             dataset_config: DictConfig, task_config: DictConfig, initialize_result=None) -> list[Path]:
        """
        Run an EIT simulation in pyEIT for meshes in each subject directory

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
            **source_directory**: subdirectory within derivative source folder to find source files

            **results_directory**: Name of the results folder (Stem of output_directory)

            **params**: (Dict): No params currently used for this task

        initialize_result
            Return dict from the initialize function

        Returns
        -------


        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        lung_deflated = task_config.params.r_lung_deflated
        lung_inflated = task_config.params.r_lung_inflated
        surrounding_tissue = task_config.params.r_surrounding_tissue
        n_el = task_config.params.n_electrodes
        lung_slice_ratio = task_config.params.lung_slice_ratio
        lamb = task_config.params["lambda"]

        pv_reference_mesh = pv.read(
            glob(str(source_directory_derivative / task_config.source_directory / "*reference*.vtu"))[0])
        pv_predicted_mesh = pv.read(
            glob(str(source_directory_derivative / task_config.source_directory / "*predicted*.vtu"))[0])
        pv_mean_mesh = initialize_result["mean_mesh"]

        # Simulate
        # --------------------------------------------------------------------------------------------------------------
        reference_mesh = PyEITMesh(node=pv_reference_mesh.points, element=pv_reference_mesh.cells_dict[VTK_TETRA])

        perm_deflated = np.array([lung_deflated if label in ["left_lung", "right_lung"] else surrounding_tissue for label in pv_reference_mesh["Element Label"]])
        perm_inflated = np.array([lung_inflated if label in ["left_lung", "right_lung"] else surrounding_tissue for label in pv_reference_mesh["Element Label"]])

        # Todo actually slice by lung height, not overall height
        electrode_nodes_reference = place_electrodes_3d(pv_reference_mesh, n_el, "z", lung_slice_ratio)
        reference_mesh.el_pos = np.array(electrode_nodes_reference)

        protocol_obj = protocol.create(
            n_el=n_el, dist_exc=3, step_meas=1, parser_meas="std"
        )
        fwd = EITForward(reference_mesh, protocol_obj)
        v0 = fwd.solve_eit(perm=perm_deflated)
        v1 = fwd.solve_eit(perm=perm_inflated)

        # Reconstruct with predicted mesh
        # --------------------------------------------------------------------------------------------------------------
        predicted_mesh = PyEITMesh(node=pv_predicted_mesh.points, element=pv_predicted_mesh.cells_dict[VTK_TETRA],
                                   perm=np.array([lung_deflated if label in ["left_lung", "right_lung"] else surrounding_tissue for label in pv_predicted_mesh["Element Label"]]))
        electrode_nodes_predicted = place_electrodes_3d(pv_predicted_mesh, n_el, "z", lung_slice_ratio)
        predicted_mesh.el_pos = np.array(electrode_nodes_predicted)

        # Recon
        # Set up eit object
        pyeit_obj = JAC(predicted_mesh, protocol_obj)
        pyeit_obj.setup(p=0.5, lamb=lamb, method="kotre", perm=predicted_mesh.perm)

        # # Dynamic solve simulated data
        ds_sim = pyeit_obj.solve(v1, v0, normalize=False)
        solution = np.real(ds_sim)

        pv_predicted_mesh["Reconstructed Impedance"] = solution

        # Reconstruct with mean mesh
        # --------------------------------------------------------------------------------------------------------------
        mean_mesh = PyEITMesh(node=pv_mean_mesh.points, element=pv_mean_mesh.cells_dict[VTK_TETRA],
                                   perm=np.array(
                                       [lung_deflated if label in ["left_lung", "right_lung"] else surrounding_tissue
                                        for label in pv_mean_mesh["Element Label"]]))
        electrode_nodes_mean = place_electrodes_3d(pv_mean_mesh, n_el, "z", lung_slice_ratio)
        mean_mesh.el_pos = np.array(electrode_nodes_mean)

        # Recon
        # Set up eit object
        pyeit_obj = JAC(mean_mesh, protocol_obj)
        pyeit_obj.setup(p=0.5, lamb=lamb, method="kotre", perm=mean_mesh.perm)

        # # Dynamic solve simulated data
        ds_sim = pyeit_obj.solve(v1, v0, normalize=False)
        solution = np.real(ds_sim)

        pv_mean_mesh["Reconstructed Impedance"] = solution

        # Save
        pv_predicted_mesh_output_file = output_directory / "predicted_mesh_solved.vtu"
        pv_predicted_mesh.save(pv_predicted_mesh_output_file)

        pv_mean_mesh_output_file = output_directory / "mean_mesh_solved.vtu"
        pv_mean_mesh.save(pv_mean_mesh_output_file)

        return [pv_predicted_mesh_output_file, pv_mean_mesh_output_file]


# Todo maybe one more AllItemsTask to generate summary report on EIT simulation results

all_tasks = [ExtractLungLobes, CreateMeshes, ExtractWholeLungs, ReferenceSelectionMesh, ExtractTorso,
             MeshLandmarksCoarse, ParseCOPDGeneSubjectGroups, SelectCOPDGeneSubjectsByValue, FormatSubjects,
             InspectMeshes, TetrahedralizeMeshes, EITSimulation]
