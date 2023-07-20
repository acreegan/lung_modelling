from cmlibs.argon.argondocument import ArgonDocument
from cmlibs.zinc.field import Field
from cmlibs.zinc.graphics import Graphics
from cmlibs.exporter.stl import ArgonSceneExporter as STLExporter
import os
from pathlib import Path

"""
Minimal example for converting exnode/exelem mesh to stl using cmlibs 
"""


def main():
    data_directory = r"data\\"
    output_directory = r"output\\"

    # Create an argon document
    document = ArgonDocument()
    document.initialiseVisualisationContents()
    context = document.getZincContext()
    region = context.getDefaultRegion()

    # Read all mesh files in the data directory
    pathlist_node = Path(data_directory).rglob("*.exnode")  # Make sure to read nodes first
    for path in pathlist_node:
        region.readFile(str(path))

    pathlist_elem = Path(data_directory).rglob("*.exelem")
    for path in pathlist_elem:
        region.readFile(str(path))

    # Create a scene
    scene = region.getScene()
    scene.beginChange()
    graphics = scene.createGraphics(Graphics.TYPE_SURFACES)
    coordinate_field = getDefaultCoordinateField(scene)
    if coordinate_field is not None:
        graphics.setCoordinateField(coordinate_field)
    scene.endChange()

    # Export scene
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    exporter = STLExporter(output_directory)
    exporter.set_document(document)
    exporter.export()


def getDefaultCoordinateField(scene):
    """
    Get the first coordinate field from the current scene
    """
    if scene:
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
