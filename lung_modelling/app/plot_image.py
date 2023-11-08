import pyvista as pv
from pathlib import Path
import matplotlib
import tkinter as tk
from tkinter.filedialog import askopenfilename
from medpy.io import load


if __name__ == "__main__":
    tk.Tk().withdraw()
    if not (filename := askopenfilename(title="Select medical image file")):
        exit()

    image_data, header = load(filename)
    n_values = len(set(image_data.ravel()))

    p = pv.Plotter()
    c = matplotlib.colormaps["Set1"]

    p.add_volume(image_data, opacity="linear", cmap=c, show_scalar_bar=False)
    p.add_scalar_bar(n_labels=n_values, use_opacity=False,)
    p.show()
