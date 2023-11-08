import pyvista as pv
from pathlib import Path
import tkinter as tk
from tkinter.filedialog import askdirectory
from medpy.io import load
from glob import glob

if __name__ == "__main__":
    tk.Tk().withdraw()
    if not (directory := askdirectory(title="Select dicom directory")):
        exit()
    directory = Path(directory)

    filename_glob = "*.dcm"
    files = glob(str(directory / filename_glob))
    if not files:
        print("No dicom files found. Did you select the right directory?")
        exit()

    image, header = load(str(directory))

    p = pv.Plotter()
    v = p.add_volume(image)
    p.add_volume_clip_plane(v, normal="z", invert=True)

    p.add_axes()
    p.show()
