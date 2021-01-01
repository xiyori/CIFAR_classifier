# import torch
import numpy as np

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def mat_to_img(mat: np.array) -> np.array:
    fig = Figure()
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    ax = fig.add_subplot(111)
    ax.matshow(mat)

    canvas.draw()       # draw the canvas, cache the renderer

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image.shape = (h, w, 3)
    image = np.moveaxis(np.moveaxis(image, 1, -1), 1, 0)
    return image
