import pyvista as pv
from pathlib import Path
from glob import glob
import matplotlib
import tkinter as tk
from tkinter.filedialog import askdirectory

if __name__ == "__main__":
    tk.Tk().withdraw()
    if not (directory := askdirectory(title="Select directory")):
        exit()
    directory = Path(directory)
    filename_glob = "*.stl"

    files = glob(str(directory / filename_glob))

    if not files:
        print("No .stl files found. Did you select the right directory?")
        exit()

    meshes = []
    for file in files:
        mesh = pv.read(file)
        meshes.append(mesh)

    p = pv.Plotter()
    c = matplotlib.colormaps["hsv"]
    for i, mesh in enumerate(meshes):
        p.add_mesh(mesh, color=c((i + 1) / (len(meshes))), label=str(Path(files[i]).stem))

    p.add_legend()
    p.show()
