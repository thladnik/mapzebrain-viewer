from __future__ import annotations
from abc import abstractmethod
from typing import Dict, List, Union

import colorcet as cc
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import tifffile

debug = False

regions_structure = {
    'prosencephalon (forebrain)': {
        'telencephalon': {
            'olfactory bulb': 2.0,
            'pallium (dorsal telencephalon)': 2.0,
            'subpallium (ventral telencephalon)': 2.0
        },
        'hypothalamus': {
            'rostral hypothalamus': 2.0,
            'intermediate hypothalamus (entire)': {
                'intermediate hypothalamus (remaining)': 2.0,
                'diffuse nucleus of the inferior lobe': 2.0
            },
            'caudal hypothalamus': 2.0,
            'pituitary (hypophysis)': 2.0
        },
        'preoptic area': {
            'retinal arborization field 1': 2.0,
            'retinal arborization field 2': 2.0
        },
        'posterior tuberculum, anterior part (basal prosomere 3, ventral posterior tuberculum)': 2.0,
        'retina': 2.0,
        'eminentia thalami': {
            'ventral entopeduncular nucleus': 2.0,
            'eminentia thalami (remaining)': 2.0
        },
        'prethalamus (alar prosomere 3, ventral thalamus)': {
            'retinal arborization field 3': 2.0
        },
        'thalamus (alar prosomere 2, dorsal thalamus)': {
            'pineal complex (epiphysis)': 2.0,
            'habenula': {
                'dorsal habenula': 2.0,
                'ventral habenula': 2.0
            },
            'thalamus proper': {
                'retinal arborization field 4': 2.0
            }
        },
        'pretectum (alar prosomere 1)': {
            'retinal arborization field 5': 2.0,
            'retinal arborization field 6': 2.0,
            'retinal arborization field 7': 2.0,
            'retinal arborization field 8': 2.0,
            'retinal arborization field 9': 2.0
        },
        'posterior tuberculum, posterior part (basal prosomere 2, dorsal posterior tuberculum)': 2.0,
        'region of the nucleus of the medial longitudinal fascicle (basal prosomere 1)': {
            'nucleus of the medial longitudinal fascicle': 2.0
        }
    },
    'mesencephalon (midbrain)': {
        'tegmentum (midbrain tegmentum)': {
            'medial tegmentum (entire)': {
                'medial tegmentum (remaining)': 2.0,
                'oculomotor nucleus': 2.0
            },
            'lateral tegmentum': 2.0
        },
        'tectum & tori': {
            'torus longitudinalis': 2.0,
            'tectum': {
                'tectum neuropil': {
                    'boundary zone between SFGS and SGC': 2.0,
                    'stratum opticum (SO)': 2.0,
                    'stratum fibrosum et griseum superficiale (SFGS)': 2.0,
                    'stratum marginale (SM)': 2.0,
                    'boundary zone between SAC and periventricular layer': 2.0,
                    'stratum album centrale (SAC)': 2.0,
                    'stratum griseum centrale (SGC)': 2.0
                },
                'periventricular layer': 2.0
            },
            'torus semicircularis': 2.0
        }
    },
    'rhombencephalon (hindbrain)': {
        'cerebellum': 2.0,
        'medulla oblongata': {
            'superior medulla oblongata': {
                'superior dorsal medulla oblongata': {
                    'superior dorsal medulla oblongata stripe 1 (entire)': {
                        'trochlear motor nucleus': 2.0,
                        'superior dorsal medulla oblongata stripe 1 (remaining)': 2.0
                    },
                    'superior dorsal medulla oblongata stripe 2&3': 2.0,
                    'superior dorsal medulla oblongata stripe 4': 2.0,
                    'superior dorsal medulla oblongata stripe 5': 2.0,
                    'medial octavolateralis nucleus': 2.0
                },
                'superior ventral medulla oblongata (entire)': {
                    'anterior (dorsal) trigeminal motor nucleus': 2.0,
                    'superior raphe': 2.0,
                    'interpeduncular nucleus': 2.0,
                    'locus coeruleus': 2.0,
                    'posterior (ventral) trigeminal motor nucleus': 2.0,
                    'superior ventral medulla oblongata (remaining)': 2.0
                }
            }
        }
    },
    'peripheral nervous system': {
        'olfactory epithelium': 2.0,
        'anterior lateral line ganglion': 2.0,
        'trigeminal ganglion': 2.0,
        'posterior lateral line ganglion': 2.0,
        'octaval ganglion': 2.0,
        'glossopharyngeal ganglion': 2.0
    }
}


class SecionView(pg.ImageView):

    direction_label: str = None

    sig_index_changed = QtCore.Signal(int)

    last_idx: int = -1

    def __init__(self, window: Window):
        level_mode = 'rgba' if debug else 'mono'
        pg.ImageView.__init__(self, discreteTimeLine=True, levelMode=level_mode)
        self.window = window

        # Disable unnecessary UI
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        # Add scatter plot item for ROI display
        self.sct = pg.ScatterPlotItem(symbol='o', size=6)
        self.sct.setPen(pg.mkPen({'color': 'black', 'width': 3}))
        self.sct.setBrush(pg.mkBrush(color='white'))
        self.view.addItem(self.sct)

        # Add region image item
        self.region_image_item = pg.ImageItem()
        self.view.addItem(self.region_image_item)

        # Add lines
        self.vline = VerticalLine(self)
        self.view.addItem(self.vline)

        self.hline = HorizontalLine(self)
        self.view.addItem(self.hline)

        # Add anatomical directions
        # self.direction_text_item = pg.TextItem(self.direction_label)
        # self.view.addItem(self.direction_text_item)
        # self.direction_text_item.setPos(30, 20)
        # self.label_image_item = pg.ImageItem(tifffile.imread('icon_rostral_dorsal.tiff'))
        # self.label_image_item.setScale(0.05)
        # self.view.addItem(self.label_image_item)

        # Connect
        self.sigTimeChanged.connect(self.time_changed)
        self.sig_index_changed.connect(self.update_regions)
        self.sig_index_changed.connect(self.update_scatter)

    @abstractmethod
    def get_region_slice(self, region: np.ndarray):
        pass

    @abstractmethod
    def get_coordinate_slice(self) -> np.ndarray:
        pass

    @abstractmethod
    def update_marker_image(self):
        pass

    @abstractmethod
    def ymax(self) -> int:
        pass

    def time_changed(self):
        """Check currentIndex against last_idx. This is done to prevent unnecessary update calls, since
        sigTimeChanged is also emitted on fractional changes of the timeline
        """

        cur_idx = self.currentIndex
        if self.last_idx == cur_idx:
            return

        self.last_idx = cur_idx

        self.sig_index_changed.emit(cur_idx)

    def update_regions(self):

        if len(self.window.regions) == 0:
            return

        images = []
        for i, (name, region) in enumerate(self.window.regions.items()):
            region_slice = self.get_region_slice(region)
            color = cc.m_glasbey_dark(i)

            slice_im = np.repeat(region_slice[:, :, None], 4, axis=2) * color
            # slice_im[:, :, 3] = 50
            slice_im[region_slice == 0, 3] = 0

            images.append(slice_im)

        im = np.array(images).sum(axis=0)
        im[im[:, :, 3] > 0, 3] = 50
        # im = np.clip(im, 0, 255)

        self.region_image_item.setImage(im.astype(np.uint8))

    def update_scatter(self):

        # Skip if no points are set
        if self.window.points is None:
            return

        data_slice = self.get_coordinate_slice()

        # print('>', len(data_slice))
        if len(data_slice) > 0:
            self.sct.setData(*data_slice.T)
        else:
            self.sct.setData([], [])

    def update_vline(self, idx: int):
        self.vline.blockSignals(True)
        self.vline.setPos(idx)
        self.vline.blockSignals(False)

    def update_hline(self, idx: int):
        self.hline.blockSignals(True)
        self.hline.setPos(self.ymax() - idx)
        self.hline.blockSignals(False)

    def update_index(self, idx: int):
        print(f'Set index {idx}')
        self.timeLine.setPos(idx)


class SaggitalView(SecionView):

    def update_marker_image(self):
        self.setImage(self.window.marker_image[:, :, ::-1], axes={'t': 0, 'x': 1, 'y': 2, 'c': 3})
        self.timeLine.setPos(self.window.marker_image.shape[0] // 2)

    def get_region_slice(self, region: np.ndarray):
        return np.swapaxes(region[self.currentIndex, :, :], 0, 1)[::-1, :]

    def get_coordinate_slice(self) -> np.ndarray:

        data_slice = self.window.points[self.window.points[:, 0].astype(int) == self.currentIndex, 1:]
        data_slice[:, 1] = self.window.marker_image.shape[2] - data_slice[:, 1]

        return data_slice

    def ymax(self):
        return self.window.marker_image.shape[2]


class CoronalView(SecionView):

    def update_marker_image(self):
        self.setImage(self.window.marker_image[::-1, :, :], axes={'t': 2, 'x': 1, 'y': 0, 'c': 3})
        self.timeLine.setPos(self.window.marker_image.shape[2] // 2)

    def get_region_slice(self, region: np.ndarray):
        return region[::-1, :, self.currentIndex]

    def get_coordinate_slice(self) -> np.ndarray:
        data_slice = self.window.points[self.window.points[:, 2].astype(int) == self.currentIndex, :2][:, ::-1]
        data_slice[:, 1] = self.ymax() - data_slice[:, 1]

        return data_slice

    def ymax(self):
        return self.window.marker_image.shape[0]


class TransversalView(SecionView):

    def update_marker_image(self):
        self.setImage(self.window.marker_image[:, :, ::-1], axes={'t': 1, 'x': 0, 'y': 2, 'c': 3})
        self.timeLine.setPos(self.window.marker_image.shape[1] // 2)

    def get_region_slice(self, region: np.ndarray):
        return np.swapaxes(region[:, self.currentIndex, :], 0, 1)[::-1, :]

    def get_coordinate_slice(self) -> np.ndarray:
        data_slice = self.window.points[self.window.points[:, 1].astype(int) == self.currentIndex, ::2]
        data_slice[:, 1] = self.window.marker_image.shape[2] - data_slice[:, 1]

        return data_slice

    def ymax(self):
        return self.window.marker_image.shape[2]


class MovableLine(pg.InfiniteLine):

    _angle: int = None
    sig_position_changed = QtCore.Signal(int)

    def __init__(self, section_view: SecionView):
        pg.InfiniteLine.__init__(self, angle=self._angle, movable=True)
        self.section_view = section_view

        self.setPen(pg.mkPen(color=(200, 200, 100), width=3))
        self.setHoverPen(pg.mkPen(color=(255, 0, 0), width=self.pen.width()))

        self.sigPositionChanged.connect(self.position_changed)

    @abstractmethod
    def position_changed(self, line: pg.InfiniteLine):
        pass


class HorizontalLine(MovableLine):

    _angle = 0

    def position_changed(self, line: pg.InfiniteLine):
        self.sig_position_changed.emit(self.section_view.ymax() - int(line.getPos()[1]))


class VerticalLine(MovableLine):

    _angle = 90

    def position_changed(self, line: pg.InfiniteLine):
        self.sig_position_changed.emit(int(line.getPos()[0]))


class SearchSelectTreeWidget(QtWidgets.QWidget):

    sig_item_selected = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_removed = QtCore.Signal(QtWidgets.QTreeWidgetItem)

    def __init__(self, *args):
        QtWidgets.QWidget.__init__(self, *args)

        self.setLayout(QtWidgets.QVBoxLayout())

        # Add regions tree widget
        self.search_field = QtWidgets.QLineEdit('')
        self.search_field.setPlaceholderText('Type to search for region')
        self.search_field.textChanged.connect(self.search_tree)
        self.layout().addWidget(self.search_field)
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.headerItem().setText(0, 'Regions')
        self.tree_widget.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree_widget.headerItem().setText(1, '')
        self.tree_widget.header().resizeSection(1, 40)
        self.tree_widget.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.tree_widget.header().setStretchLastSection(False)

        self.selected_items = []

    def build_tree(self, data: dict):

        self.selected_items.clear()
        self.tree_widget.clear()

        # Recursive build function
        def _build_tree(name: str, data: dict, parent_item = None):
            parent_item = parent_item if parent_item is not None else self.tree_widget

            # Add tree item
            tree_item = QtWidgets.QTreeWidgetItem(parent_item)

            # Add label
            label = QtWidgets.QLabel(name, self.tree_widget)
            self.tree_widget.setItemWidget(tree_item, 0, label)
            tree_item.setData(0, 1, name)

            # Add button
            btn = QtWidgets.QPushButton('select')
            btn.setContentsMargins(0, 0, 0, 0)
            btn.clicked.connect(lambda: self.toggle_item(tree_item))
            self.tree_widget.setItemWidget(tree_item, 1, btn)

            if isinstance(data, dict):
                for n, d in data.items():
                    tree_item.addChild(_build_tree(n, d, tree_item))

            return tree_item

        for tl_name, tl_data in data.items():
            # print(tl_name, tl_data)
            item = _build_tree(tl_name, tl_data)
            self.tree_widget.addTopLevelItem(item)
        self.layout().addWidget(self.tree_widget)

    def search_tree(self, search_text: str):

        def _find_text_in_item(item: QtWidgets.QTreeWidgetItem) -> bool:
            match = False
            for i in range(item.childCount()):
                child = item.child(i)
                match = match | _find_text_in_item(child)

            item_text = item.data(0, 1)
            print(item_text)

            selected = item in self.selected_items
            found = (len(search_text) > 0 and search_text in item_text)
            match |= found | selected

            item.setHidden(not match)
            if len(search_text) == 0:
                item.setHidden(False)
            item.setExpanded(match)

            lbl = self.tree_widget.itemWidget(item, 0)
            if match:
                lbl.setText(item_text.replace(search_text, f'<span style="color: red">{search_text}</span>'))
            else:
                lbl.setText(item_text)

            return match

        # Run search
        for i in range(self.tree_widget.topLevelItemCount()):
            _find_text_in_item(self.tree_widget.topLevelItem(i))

    def toggle_item(self, tree_item: QtWidgets.QTreeWidgetItem):
        print(f'Selected {tree_item}')

        # Add item
        if tree_item not in self.selected_items:

            self.selected_items.append(tree_item)

            label = self.tree_widget.itemWidget(tree_item, 0)
            label.setStyleSheet('background-color: green;')
            btn = self.tree_widget.itemWidget(tree_item, 1)
            btn.setText('remove')

            self.sig_item_selected.emit(tree_item)

        # Remove item
        else:

            self.selected_items.remove(tree_item)

            label = self.tree_widget.itemWidget(tree_item, 0)
            label.setStyleSheet('')
            btn = self.tree_widget.itemWidget(tree_item, 1)
            btn.setText('select')

            self.sig_item_removed.emit(tree_item)


class ControlPanel(QtWidgets.QWidget):

    def __init__(self, window: Window):
        QtWidgets.QWidget.__init__(self, parent=window)
        self.window = window

        self.setMinimumWidth(300)
        self.setMaximumWidth(400)
        self.setLayout(QtWidgets.QVBoxLayout())

        self.region_tree = SearchSelectTreeWidget()
        self.region_tree.build_tree(regions_structure)
        self.layout().addWidget(self.region_tree)

        # TEMP:
        btn1 = QtWidgets.QPushButton('periventricular_layer')
        btn1.clicked.connect(lambda: self.window.add_region('periventricular_layer'))
        self.layout().addWidget(btn1)

        btn2 = QtWidgets.QPushButton('pretectum')
        btn2.clicked.connect(lambda: self.window.add_region('pretectum'))
        self.layout().addWidget(btn2)


class Window(QtWidgets.QMainWindow):

    points: Union[List[float], np.ndarray] = None
    marker_image: np.ndarray = None
    regions: Dict[str, np.ndarray] = None

    sig_regions_updated = QtCore.Signal()
    sig_marker_image_updated = QtCore.Signal()

    def __init__(self, points: Union[List[float], np.ndarray] = None,
                 marker: str = None,
                 regions: List[str] = None):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(1400, 800)
        self.show()

        self.points = points
        self.regions = {}

        self.wdgt = QtWidgets.QWidget()
        self.setCentralWidget(self.wdgt)
        self.wdgt.setLayout(QtWidgets.QGridLayout())

        # Add panel
        self.panel = ControlPanel(self)
        self.wdgt.layout().addWidget(self.panel, 0, 0, 2, 1)

        self.sag_view = SaggitalView(self)
        self.sig_marker_image_updated.connect(self.sag_view.update_marker_image)
        self.sig_regions_updated.connect(self.sag_view.update_regions)
        self.wdgt.layout().addWidget(self.sag_view, 0, 1, 1, 2)

        self.cor_view = CoronalView(self)
        self.sig_marker_image_updated.connect(self.cor_view.update_marker_image)
        self.sig_regions_updated.connect(self.cor_view.update_regions)
        self.wdgt.layout().addWidget(self.cor_view, 1, 1, 1, 2)

        self.trans_view = TransversalView(self)
        self.sig_marker_image_updated.connect(self.trans_view.update_marker_image)
        self.sig_regions_updated.connect(self.trans_view.update_regions)
        self.wdgt.layout().addWidget(self.trans_view, 1, 3)

        # Connect line updates
        self.sag_view.sig_index_changed.connect(self.cor_view.update_hline)
        self.sag_view.sig_index_changed.connect(self.trans_view.update_vline)
        self.sag_view.hline.sig_position_changed.connect(self.cor_view.update_index)
        self.sag_view.vline.sig_position_changed.connect(self.trans_view.update_index)

        self.cor_view.sig_index_changed.connect(self.sag_view.update_hline)
        self.cor_view.sig_index_changed.connect(self.trans_view.update_hline)
        self.cor_view.hline.sig_position_changed.connect(self.sag_view.update_index)
        self.cor_view.vline.sig_position_changed.connect(self.trans_view.update_index)

        self.trans_view.sig_index_changed.connect(self.sag_view.update_vline)
        self.trans_view.sig_index_changed.connect(self.cor_view.update_vline)
        self.trans_view.hline.sig_position_changed.connect(self.cor_view.update_index)
        self.trans_view.vline.sig_position_changed.connect(self.sag_view.update_index)

        if marker is not None:
            self.set_marker(marker)

        if regions is not None:
            for r in regions:
                self.add_region(r)

    def set_marker(self, name: str):

        if not debug:
            print(f'Set marker to {name}')
            path = f'../../ants_registration/ants_registration/mapzebrain/{name}.tif'
            self.marker_image = np.swapaxes(np.moveaxis(tifffile.imread(path), 0, 2), 0, 1)[:,:,:,None]

        else:
            # Debug image:
            im = np.zeros((300, 700, 200, 3)).astype(np.uint8)
            im[:150, :, :, 0] += 80
            im[:, :350, :, 1] += 80
            im[:, :, :100, 2] += 80

            self.marker_image = im

        self.sig_marker_image_updated.emit()

    def add_region(self, name: str):

        path = f'../../ants_registration/ants_registration/mapzebrain/regiondata/{name}.tif'

        print(f'Load region {path}')
        self.regions[name] = np.swapaxes(np.moveaxis(tifffile.imread(path), 0, 2), 0, 1)

        self.sig_regions_updated.emit()


def run(points: Union[List, np.ndarray] = None, marker: str = None, regions: List[str] = None):

    print('Open window')

    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')

    app = pg.mkQApp()

    win = Window(points=points, marker=marker, regions=regions)

    pg.exec()


if __name__ == '__main__':

    run()



