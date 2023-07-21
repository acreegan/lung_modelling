from tkinter import Tk
import pyvista as pv
from tkinter.filedialog import askopenfilenames
import os
from cmlibs.argon.argondocument import ArgonDocument
from cmlibs.zinc.field import Field
from cmlibs.zinc.graphics import Graphics
from cmlibs.exporter.stl import ArgonSceneExporter as STLExporter
from pathlib import Path
import tempfile

"""
App to convert user selected exnode and exelem files to STL
"""


def main():
    output_directory = "output"
    # Select File
    Tk().withdraw()
    filenames = askopenfilenames(title="Select .exnode and .exelem files to convert")
    if not filenames:
        return
    Tk().destroy()

    # Create an argon document
    document = ArgonDocument()
    document.initialiseVisualisationContents()
    context = document.getZincContext()
    region = context.getDefaultRegion()

    node_file_names = []
    for filename in filenames:
        if Path(filename).suffix == ".exnode":  # Make sure to read nodes first
            region.readFile(filename)
            node_file_names.append(Path(filename).stem)

    for filename in filenames:
        if Path(filename).suffix == ".exelem":
            region.readFile(filename)

    # Create a scene
    scene = region.getScene()
    graphics = scene.createGraphics(Graphics.TYPE_SURFACES)
    coordinate_field = get_default_coordinate_field(scene)
    if coordinate_field is not None:
        graphics.setCoordinateField(coordinate_field)

    # Export scene
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    prefix = "_".join(node_file_names)
    exporter = STLExporter(output_target=output_directory, output_prefix=prefix)
    exporter.set_document(document)
    exporter.export()

    with tempfile.TemporaryDirectory() as tmpdirname:
        exporter.export(output_target=tmpdirname)
        for filename in os.listdir(tmpdirname):
            mesh = pv.read(Path(tmpdirname) / filename)

    pv.plot(mesh)


def get_default_coordinate_field(scene):
    """
    Get the first coordinate field from the current scene
    """

    fielditer = scene.getRegion().getFieldmodule().createFielditerator()
    field = fielditer.next()
    while field.isValid():
        if field.isTypeCoordinate() and (field.getValueType() == Field.VALUE_TYPE_REAL) and \
                (field.getNumberOfComponents() <= 3) and field.castFiniteElement().isValid():
            return field
        field = fielditer.next()
    return None


if __name__ == "__main__":
    main()
