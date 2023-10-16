Workflow App
============

App Usage
---------

Setup
*****
To use the workflow app, first install the Lung Modelling package. See :ref:`installation`
for instructions


Once the package is installed, initialize a user-editable configuration by running:

.. code-block:: shell

    python -m lung_modelling initialize_user_configuration=True

This should create a folder named user_config in your working directory. The user
config folder contains .yaml files will all the editable configuration parameters
for the existing workflow tasks. See the `Hydra documentation <https://hydra.cc/docs/intro/>`_
for detailed information on how this works.

Running Example Workflow
************************
To test the package on example data, first download the `example data file <https://github.com/acreegan/lung_modelling/blob/main/example_data/covid_lung_lobe_segmentation.zip>`_
and unzip it:

.. code-block::

    tar -xf covid_lung_segmentation.zip

Run the batch workflow with the example settings:

.. code-block::

    python -m lung_modelling user_config=example_workflow

This will bring up a file select dialog. Select the folder covid_lung_lobe_segmentation.


Dataset Configuration
---------------------

The worfkflow app requires a specific
directory structure for the dataset. An example of the structure is as follows:

.. code-block::

    dataset_name/
    |--- dataset_config.json
    |--- primary
    |    |--- subject_directory_1
    |         |--- subject_data.nii
    |--- derivative
    |    |--- subject_directory_1
    |         |--- raw_mesh
    |              |--- lobe_1.stl

A top level directory with the dataset name has sub directories primary and
derivative. Initial data is stored in the primary directory, separated into
individual subject directories. Generated data is placed in the derivative
directory, with a structure mirroring that of the primary. Any number of sub
directories can exist before the data files.

The dataset structure along with information about the files contained in the
dataset must be specified in a file named dataset_config.json. The
dataset_config.json for the example dataset is as follows:

.. code-block:: json

    {
      "primary_directory": "primary",
      "derivative_directory": "derivative",
      "pooled_derivative_directory": "pooled_derivative",
      "directory_index_glob": "directory_index*.csv",
      "data_folder_depth": 2,
      "lung_image_glob": "*.nii",
      "lobe_mapping": {"rul": 3, "rml": 4, "rll": 5, "lul": 1,
                        "lll": 2}
    }

**primary_directory** refers to the name of the directory holding the primary
data in the dataset.

**derivative_directory** refers to the name of the directory in which to place
generated data.

**pooled_derivative_directory** refers to the name of the directory in which to place
data generated from two or more samples combined into a single output file.

**directory_index_glob** refers to a glob used to find a pre-built directory
index of the dataset if it exists.

**data_folder_depth** is the number of folders between the top level dataset
folder and the data files

**lung_image_glob** refers to a glob used to find lung image data files

**lobe_mapping** specifies the value used to indicate each lung lobe in the
lung lobe image files.
