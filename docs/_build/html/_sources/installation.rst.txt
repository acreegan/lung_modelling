Installation
============

To use the statistical shape modelling features of this package, you must first install
Shapeworks. See the `Shapeworks website <http://sciinstitute.github.io/ShapeWorks/latest/>`_
for a guide on how to do this.

Once you have an activated conda environment with Shapworks installed, (if not
using the Shapeworks features you can use a fresh conda or venv environment), lung_modelling
can be installed as follows.

Developer Environment
---------------------

To install a developer environment, clone the git repository:

.. code-block:: shell

    git clone https://github.com/acreegan/lung_modelling
    cd lung_modelling

and install with dev requirements in editable mode:

.. code-block:: shell

    pip install -e .[dev]

User Installation
-----------------

If you just want to use the app with existing workflow tasks (parameters can still be
modified), simply install the package:

.. code-block:: shell

    pip install git+https://github.com/acreegan/lung_modelling




