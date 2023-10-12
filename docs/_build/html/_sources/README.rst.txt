Lung Modelling
==============

Lung Modelling is a python package that provides tools for running statistical shape analysis
workflows on lung imaging datasets. The core statistical shape analysis functionality
is provided by `Shapeworks <http://sciinstitute.github.io/ShapeWorks/latest/>`_.

Full documentation can be found on `Github Pages <https://acreegan.github.io/lung_modelling/>`_

Source code can be found on `Github <https://github.com/acreegan/lung_modelling>`_

Features
--------
- Command line app to run an automated statistical shape analysis workflow.
- Configuration management powered by `Hydra <https://hydra.cc/docs/intro/>`_
- Workflow logging for data provenance
- Simple workflow orchestration tool WorkflowManager which provides an interface
  for adding new workflow tasks
- Library of image/mesh preprocessing tools
