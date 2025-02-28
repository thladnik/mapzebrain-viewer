from __future__ import annotations

import os
from typing import Dict, TYPE_CHECKING, Tuple, Union

import numpy as np
import stl
from pyqtgraph.Qt import QtGui

if TYPE_CHECKING:
    from main import Window


default_marker_name = 'jf5Tg'
debug: bool = False
window: Union[Window, None] = None

marker_image: Union[np.ndarray, None] = None

regions: Dict[str, Tuple[np.ndarray, Union[None, stl.Mesh]]] = {}
region_colors: Dict[str, QtGui.QColor] = {}

roi_sets: Dict[str, np.ndarray] = {}
roi_set_colors: Dict[str, QtGui.QColor] = {}


def marker_path():
    path = os.path.join(os.getenv('LOCALAPPDATA'), 'mapzebview', 'markers')

    if not os.path.exists(path):
        os.makedirs(path)

    return path


def region_path() -> str:
    path = os.path.join(os.getenv('LOCALAPPDATA'), 'mapzebview', 'regions')

    if not os.path.exists(path):
        os.makedirs(path)

    return path
