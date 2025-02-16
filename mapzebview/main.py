from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from abc import abstractmethod
from typing import Dict, List, Tuple, Union

import colorcet as cc
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import stl
from pyqtgraph import Vector
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import tifffile

import mapzebview
from mapzebview import regions

debug = False


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
        self.region_image_items = {}

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

        # Hide all
        for image_item in self.region_image_items.values():
            image_item.hide()

        # Update all
        for i, (name, (region, _)) in enumerate(self.window.regions.items()):

            image_item = self.region_image_items.get(name)

            # Create
            if image_item is None:
                image_item = pg.ImageItem()
                image_item.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_ColorDodge)
                self.view.addItem(image_item)
                self.region_image_items[name] = image_item

            # Set colormap
            color = self.window.region_colors[name]
            cmap = pg.ColorMap(pos=[0., 1.], color=[[0, 0, 0], color], linearize=True)
            image_item.setColorMap(cmap)
            # image_item.setColorMap('CET-L14')

            region_slice = self.get_region_slice(region)
            image_item.setImage(region_slice)
            image_item.show()

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


class VolumeView(gl.GLViewWidget):

    mesh_items: Dict[str, gl.GLMeshItem] = {}
    sag_plane_mesh_item: gl.GLMeshItem = None

    sag_plane_verts = np.array([[0.5, 0, 0], [0.5, 1, 0], [0.5, 1, 1], [0.5, 0, 1]], dtype=np.float32)
    cor_plane_verts = np.array([[0, 0, 0.0], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]], dtype=np.float32)
    trans_plane_verts = np.array([[0, 0.5, 0], [1, 0.5, 0], [1, 0.5, 1], [0, 0.5, 1]], dtype=np.float32)

    volume_bounds: np.ndarray = None

    def __init__(self, window: Window):
        gl.GLViewWidget.__init__(self)
        self.window = window
        self.opts['fov'] = 1.

        self.scatter_item = gl.GLScatterPlotItem(size=5, color=(1., 1., 1., 1.0), pxMode=False)
        self.scatter_item .setGLOptions('additive')
        self.addItem(self.scatter_item)

        sag_plane_data = gl.MeshData(vertexes=self.sag_plane_verts, faces=np.array([[0, 1, 2], [2, 3, 0]]))
        self.saggital_plane = gl.GLMeshItem(meshdata=sag_plane_data, smooth=True, shader='balloon')
        self.saggital_plane.setColor((200 / 255, 200 / 255, 100 / 255, 0.05))
        self.saggital_plane.setGLOptions('additive')
        self.addItem(self.saggital_plane)

        cor_plane_data = gl.MeshData(vertexes=self.cor_plane_verts, faces=np.array([[0, 1, 2], [2, 3, 0]]))
        self.coronal_plane = gl.GLMeshItem(meshdata=cor_plane_data, smooth=True, shader='balloon')
        self.coronal_plane.setColor((200 / 255, 200 / 255, 100 / 255, 0.05))
        self.coronal_plane.setGLOptions('additive')
        self.addItem(self.coronal_plane)

        trans_plane_data = gl.MeshData(vertexes=self.trans_plane_verts, faces=np.array([[0, 1, 2], [2, 3, 0]]))
        self.transverse_plane = gl.GLMeshItem(meshdata=trans_plane_data, smooth=True, shader='balloon')
        self.transverse_plane.setColor((200 / 255, 200 / 255, 100 / 255, 0.05))
        self.transverse_plane.setGLOptions('additive')
        self.addItem(self.transverse_plane)

    def marker_image_updated(self):

        self.volume_bounds = np.array(self.window.marker_image.shape[:3], dtype=np.float32)

        # Set plane extents and position
        self.set_saggital_data(self.window.marker_image.shape[0] // 2)
        self.set_coronal_data(self.window.marker_image.shape[2] // 2)
        self.set_transverse_data(self.window.marker_image.shape[1] // 2)

        # Set camera
        self.setCameraPosition(pos=Vector(*[int(i // 2) for i in self.window.marker_image.shape[:3]]), distance=60000)

    def set_saggital_data(self, current_idx: int):

        if self.volume_bounds is None:
            return

        vertices = self.sag_plane_verts * self.volume_bounds[None, :]
        vertices[:, 0] = self.volume_bounds[0] - current_idx
        self.saggital_plane.setMeshData(meshdata=gl.MeshData(vertexes=vertices,
                                                             faces=np.array([[0, 1, 2], [2, 3, 0]])))

    def set_coronal_data(self, current_idx: int):

        if self.volume_bounds is None:
            return

        vertices = self.cor_plane_verts * self.volume_bounds[None, :]
        vertices[:, 2] = current_idx
        self.coronal_plane.setMeshData(meshdata=gl.MeshData(vertexes=vertices,
                                                            faces=np.array([[0, 1, 2], [2, 3, 0]])))

    def set_transverse_data(self, current_idx: int):

        if self.volume_bounds is None:
            return

        vertices = self.trans_plane_verts * self.volume_bounds[None, :]
        vertices[:, 1] = current_idx
        self.transverse_plane.setMeshData(meshdata=gl.MeshData(vertexes=vertices,
                                                               faces=np.array([[0, 1, 2], [2, 3, 0]])))

    def update_scatter(self):
        points = self.window.points
        points[:, 0] = self.volume_bounds[0] - points[:, 0]
        self.scatter_item.setData(pos=points)
        self.scatter_item.setData(color=(1., 1., 1., 0.1))

    def update_regions(self):

        for mesh_item in self.mesh_items.values():
            mesh_item.hide()

        for i, (name, (_, region_mesh)) in enumerate(self.window.regions.items()):

            if name not in self.mesh_items:
                vecs = region_mesh.vectors
                # Invert X for GL view
                vecs[:, :, 0] = self.volume_bounds[0] - vecs[:, :, 0]

                data_item = gl.MeshData(vertexes=vecs)
                mesh_item = gl.GLMeshItem(meshdata=data_item, smooth=True, shader='balloon')
                mesh_item.setGLOptions('additive')

                self.addItem(mesh_item)

                self.mesh_items[name] = mesh_item

            mesh_item = self.mesh_items[name]

            # Show mesh
            mesh_item.show()

            # Set color
            color = self.window.region_colors[name]
            mesh_item.setColor((*[c / 255 for c in color[:3]], 0.2))


class SearchSelectTreeWidget(QtWidgets.QWidget):

    ContinuousIdRole = 40
    UniqueNameRole = 50
    ColorRole = 60

    sig_item_selected = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_removed = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_color_changed = QtCore.Signal(QtWidgets.QTreeWidgetItem)

    item_count = 0

    def __init__(self, *args, colorpicker: bool = False):
        QtWidgets.QWidget.__init__(self, *args)
        self.colorpicker = colorpicker
        self.setLayout(QtWidgets.QVBoxLayout())

        # Add searchfield
        self.search_field = QtWidgets.QLineEdit('')
        self.search_field.setPlaceholderText('Type to search for region')
        self.search_field.textChanged.connect(self.search_tree)
        self.layout().addWidget(self.search_field)

        # Add tree widget
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.headerItem().setText(0, 'Regions')
        self.tree_widget.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree_widget.headerItem().setText(1, '')
        self.tree_widget.header().resizeSection(1, 50)
        self.tree_widget.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        if self.colorpicker:
            self.tree_widget.headerItem().setText(2, '')
            self.tree_widget.header().resizeSection(2, 25)
            self.tree_widget.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.tree_widget.header().setStretchLastSection(False)

        self.sig_item_color_changed.connect(self.update_colorpicker_button)

        self.selected_items = []

    def build_tree(self, data: dict):

        self.selected_items.clear()
        self.tree_widget.clear()

        self.item_count = 0

        # Recursive build function
        def _build_tree(name: str, data: dict, parent_item = None):
            parent_item = parent_item if parent_item is not None else self.tree_widget

            # Add tree item
            tree_item = QtWidgets.QTreeWidgetItem(parent_item)

            # Add label
            label = QtWidgets.QLabel(name, self.tree_widget)
            self.tree_widget.setItemWidget(tree_item, 0, label)

            # Add select button
            select_btn = QtWidgets.QPushButton('show')
            select_btn.setContentsMargins(0, 0, 0, 0)
            select_btn.clicked.connect(lambda: self.toggle_item(tree_item))
            self.tree_widget.setItemWidget(tree_item, 1, select_btn)

            # Add colorpicker button
            color_btn = QtWidgets.QPushButton(f' ')
            color_btn.setContentsMargins(0, 0, 0, 0)
            self.tree_widget.setItemWidget(tree_item, 2, color_btn)
            color_btn = self.tree_widget.itemWidget(tree_item, 2)
            color_btn.setDisabled(True)

            # Set item data
            tree_item.setData(0, SearchSelectTreeWidget.UniqueNameRole, name)
            tree_item.setData(0, SearchSelectTreeWidget.ContinuousIdRole, self.item_count)

            # Increment before children
            self.item_count += 1

            # Go through children
            if isinstance(data, dict):
                for n, d in data.items():
                    tree_item.addChild(_build_tree(n, d, tree_item))

            return tree_item

        for tl_name, tl_data in data.items():
            item = _build_tree(tl_name, tl_data)
            self.tree_widget.addTopLevelItem(item)
        self.layout().addWidget(self.tree_widget)

    def search_tree(self, search_text: str):

        def _find_text_in_item(item: QtWidgets.QTreeWidgetItem, match: bool) -> bool:
            for i in range(item.childCount()):
                child = item.child(i)
                match |= _find_text_in_item(child, match)

            # Get text to search
            item_text = item.data(0, SearchSelectTreeWidget.UniqueNameRole)

            # Check match
            selected = item in self.selected_items
            found = (len(search_text) > 0 and search_text in item_text)
            match |= found | selected

            # Set visibility and expansion state
            item.setHidden(not match)
            if len(search_text) == 0:
                item.setHidden(False)
            item.setExpanded(match)

            # Update label to mark search phrase
            lbl = self.tree_widget.itemWidget(item, 0)
            if match:
                lbl.setText(item_text.replace(search_text, f'<span style="color: red">{search_text}</span>'))
            else:
                lbl.setText(item_text)

            return match

        # Run search
        for i in range(self.tree_widget.topLevelItemCount()):
            _find_text_in_item(self.tree_widget.topLevelItem(i), False)

    def select_exact_match_in_tree(self, search_text: str):

        def _find_item_in_tree(item: QtWidgets.QTreeWidgetItem) -> Union[QtWidgets.QTreeWidgetItem, None]:

            # Get text to search
            item_text = item.data(0, SearchSelectTreeWidget.UniqueNameRole)

            if item_text == search_text:
                return item

            # Go through children
            for i in range(item.childCount()):
                child = item.child(i)
                ret = _find_item_in_tree(child)

                if ret is not None:
                    return ret

            return None

        # Run search
        for i in range(self.tree_widget.topLevelItemCount()):
            final_ret = _find_item_in_tree(self.tree_widget.topLevelItem(i))

            if final_ret is not None:
                self.toggle_item(final_ret)
                # Search tree to cause selected item to expand
                self.search_tree('')
                break

    def toggle_item(self, tree_item: QtWidgets.QTreeWidgetItem):

        # Add item
        if tree_item not in self.selected_items:
            print(f'Selected {tree_item}')

            self.selected_items.append(tree_item)

            # Set select button
            btn = self.tree_widget.itemWidget(tree_item, 1)
            btn.setText('hide')

            # Set color and state on color picker button
            # color = cc.m_glasbey_bw(self.get_item_continuous_id(tree_item))[:3]
            color = cc.glasbey_hv[self.get_item_continuous_id(tree_item)][:3]
            # color = cc.m_glasbey_warm(self.get_item_continuous_id(tree_item))[:3]
            color = [int(255 * c) for c in color]
            color_btn = self.tree_widget.itemWidget(tree_item, 2)
            color_btn.setDisabled(True)

            # Set color to item data
            self.set_item_color(tree_item, color)

            self.sig_item_selected.emit(tree_item)

        # Remove item
        else:
            print(f'Removed {tree_item}')

            # Update select button
            label = self.tree_widget.itemWidget(tree_item, 0)
            label.setStyleSheet('')
            btn = self.tree_widget.itemWidget(tree_item, 1)
            btn.setText('show')

            # Update color picker button
            color_btn = self.tree_widget.itemWidget(tree_item, 2)
            color_btn.setStyleSheet('')
            color_btn.setDisabled(True)

            # Remove item
            self.selected_items.remove(tree_item)
            self.sig_item_removed.emit(tree_item)

    def update_colorpicker_button(self, tree_item: QtWidgets.QTreeWidgetItem):

        color = self.get_item_color(tree_item)
        color_btn = self.tree_widget.itemWidget(tree_item, 2)
        color_btn.setStyleSheet(f'background-color: rgb{(*color,)};')

    def set_item_color(self, tree_item: QtWidgets.QTreeWidgetItem, color: tuple):
        tree_item.setData(0, self.ColorRole, color)

        self.sig_item_color_changed.emit(tree_item)

    def get_item_name(self, tree_item: QtWidgets.QTreeWidgetItem):
        return tree_item.data(0, self.UniqueNameRole)

    def get_item_continuous_id(self, tree_item: QtWidgets.QTreeWidgetItem):
        return tree_item.data(0, self.ContinuousIdRole)

    def get_item_color(self, tree_item: QtWidgets.QTreeWidgetItem):
        return tree_item.data(0, self.ColorRole)


class RoiListWidget(QtWidgets.QGroupBox):

    _drop_text = 'Drop files to load...'

    sig_path_added = QtCore.Signal(str)

    def __init__(self, parent=None):
        QtWidgets.QGroupBox.__init__(self, 'ROIs', parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)
        self.setMaximumHeight(300)
        self.setLayout(QtWidgets.QVBoxLayout())

        self.drop_label = QtWidgets.QLabel(self._drop_text)
        self.layout().addWidget(self.drop_label)

        self.list_widget = QtWidgets.QListWidget()
        self.layout().addWidget(self.list_widget)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            ext = event.mimeData().urls()[0].fileName().split('.')[-1]
            self.list_widget.hide()
            self.setFixedSize(self.size())

            if ext.lower() in ['h5', 'hdf5', 'npy', 'json', 'csv']:
                self.drop_label.setText(f'Drop {ext.upper()} file here to load')
                self.drop_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            else:
                self.drop_label.setText(f'Unable to import file with extension {ext.upper()}')
                self.drop_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            event.accept()

        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.reset_widgets()
        event.accept()

    def dropEvent(self, event):
        self.drop_label.setText('Loading data...')

        QtWidgets.QApplication.instance().processEvents()

        for url in event.mimeData().urls():
            print(f'Load ROIs from {url}')
            ext = url.fileName().split('.')[-1]

            if ext in ['h5', 'hdf5', 'npy', 'json', 'csv']:
                self.sig_path_added.emit(url.path().strip('/'))

        self.reset_widgets()

    def reset_widgets(self):
        self.list_widget.show()
        self.drop_label.setText(self._drop_text)

        self.setMaximumSize(9999, 9999)
        self.setMinimumSize(0, 0)


class ControlPanel(QtWidgets.QGroupBox):

    sig_region_color_changed = QtCore.Signal(str, list)
    sig_roi_data_changed = QtCore.Signal()

    def __init__(self, window: Window):
        QtWidgets.QGroupBox.__init__(self, 'Navigation', parent=window)
        self.window = window

        self.setMinimumWidth(300)
        self.setMaximumWidth(400)
        self.setLayout(QtWidgets.QVBoxLayout())

        self.region_tree = SearchSelectTreeWidget(colorpicker=True)
        self.region_tree.build_tree(regions.structure)
        self.region_tree.sig_item_selected.connect(self.region_selected)
        self.region_tree.sig_item_removed.connect(self.region_removed)
        self.region_tree.sig_item_color_changed.connect(self.region_color_changed)
        self.layout().addWidget(self.region_tree)

        self.roi_list = RoiListWidget()
        self.roi_list.sig_path_added.connect(self.roi_path_added)
        self.layout().addWidget(self.roi_list)

    def roi_path_added(self, path: str):
        ext = path.split('.')[-1]

        if ext in ['h5', 'hdf5']:
            df: pd.DataFrame = pd.read_hdf(path)
            keys = []
            for _n in ['x', 'y', 'z']:
                _keys = [k for k in df.keys() if _n in k]
                if len(_keys) > 0:
                    keys.append(_keys[0])

            if len(keys) == 0:
                raise KeyError('No matching coordinate keys found for x/y/z')

            print(f'Found coordinate keys: {keys}')

            data = df[keys].values

        elif ext == 'npy':
            data = np.load(path)

        else:
            data = None

        self.window.points = data

        self.sig_roi_data_changed.emit()

    def region_selected(self, item: QtWidgets.QTreeWidgetItem):
        name = self.region_tree.get_item_name(item)
        self.window.add_region(name)

    def region_color_changed(self, item: QtWidgets.QTreeWidgetItem):
        name = self.region_tree.get_item_name(item)
        color = self.region_tree.get_item_color(item)

        self.sig_region_color_changed.emit(name, color)

    def region_removed(self, item: QtWidgets.QTreeWidgetItem):
        name = item.data(0, SearchSelectTreeWidget.UniqueNameRole)
        self.window.remove_region(name)


class Window(QtWidgets.QMainWindow):

    points: Union[List[float], np.ndarray] = None
    marker_image: np.ndarray = None
    regions: Dict[str, Tuple[np.ndarray, Union[None, stl.Mesh]]] = {}
    region_colors: Dict[str, list] = {}

    sig_regions_updated = QtCore.Signal()
    sig_marker_image_updated = QtCore.Signal()

    def __init__(self,
                 points: Union[List[float], np.ndarray] = None,
                 marker: str = None,
                 regions: List[str] = None):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(1600, 800)
        self.show()
        self.setWindowTitle('MapZeBrain Viewer')

        # TODO: handle loading of coordinate data uniformly
        if isinstance(points, pd.DataFrame):
            points = points.values

        self.points = points

        self.wdgt = QtWidgets.QWidget()
        self.setCentralWidget(self.wdgt)
        self.wdgt.setLayout(QtWidgets.QHBoxLayout())

        # Add panel
        self.panel = ControlPanel(self)
        self.panel.sig_region_color_changed.connect(self.update_region_color)
        self.wdgt.layout().addWidget(self.panel)

        self.browser = QtWidgets.QGroupBox('Anatomy browser')
        self.browser.setLayout(QtWidgets.QGridLayout())
        self.wdgt.layout().addWidget(self.browser)

        # Saggital view
        self.saggital_view = SaggitalView(self)
        self.sig_marker_image_updated.connect(self.saggital_view.update_marker_image)
        self.sig_regions_updated.connect(self.saggital_view.update_regions)
        self.browser.layout().addWidget(self.saggital_view, 0, 0)

        # Coronal view
        self.coronal_view = CoronalView(self)
        self.sig_marker_image_updated.connect(self.coronal_view.update_marker_image)
        self.sig_regions_updated.connect(self.coronal_view.update_regions)
        self.browser.layout().addWidget(self.coronal_view, 1, 0)

        # Transversal view
        self.transverse_view = TransversalView(self)
        self.sig_marker_image_updated.connect(self.transverse_view.update_marker_image)
        self.sig_regions_updated.connect(self.transverse_view.update_regions)
        self.browser.layout().addWidget(self.transverse_view, 1, 1)

        # Volumetric view
        self.volume_view = VolumeView(self)
        self.sig_marker_image_updated.connect(self.volume_view.marker_image_updated)
        self.sig_regions_updated.connect(self.volume_view.update_regions)
        self.browser.layout().addWidget(self.volume_view, 0, 1)

        # Connect line updates
        self.saggital_view.sig_index_changed.connect(self.coronal_view.update_hline)
        self.saggital_view.sig_index_changed.connect(self.transverse_view.update_vline)
        self.saggital_view.hline.sig_position_changed.connect(self.coronal_view.update_index)
        self.saggital_view.vline.sig_position_changed.connect(self.transverse_view.update_index)
        self.saggital_view.sig_index_changed.connect(self.volume_view.set_saggital_data)

        self.coronal_view.sig_index_changed.connect(self.saggital_view.update_hline)
        self.coronal_view.sig_index_changed.connect(self.transverse_view.update_hline)
        self.coronal_view.hline.sig_position_changed.connect(self.saggital_view.update_index)
        self.coronal_view.vline.sig_position_changed.connect(self.transverse_view.update_index)
        self.coronal_view.sig_index_changed.connect(self.volume_view.set_coronal_data)

        self.transverse_view.sig_index_changed.connect(self.saggital_view.update_vline)
        self.transverse_view.sig_index_changed.connect(self.coronal_view.update_vline)
        self.transverse_view.hline.sig_position_changed.connect(self.coronal_view.update_index)
        self.transverse_view.vline.sig_position_changed.connect(self.saggital_view.update_index)
        self.transverse_view.sig_index_changed.connect(self.volume_view.set_transverse_data)

        # Connect scatter plot updates
        self.panel.sig_roi_data_changed.connect(self.saggital_view.update_scatter)
        self.panel.sig_roi_data_changed.connect(self.coronal_view.update_scatter)
        self.panel.sig_roi_data_changed.connect(self.transverse_view.update_scatter)
        self.panel.sig_roi_data_changed.connect(self.volume_view.update_scatter)

        marker_catalog_path = os.path.join(self.marker_path(), 'markers_catalog.json')
        if not os.path.exists(marker_catalog_path):
            print('Get marker catalog from MapZeBrain')
            urllib.request.urlretrieve('https://api.mapzebrain.org/media/downloads/Lines/markers_catalog.json',
                                       marker_catalog_path)

        print('Load marker catalog')
        markers_catalog = json.load(open(marker_catalog_path, 'r'))
        self.marker_structure = {d['name']: d['stack'] for d in markers_catalog}

        # Make sure marker line is set
        if marker is None:
            marker = mapzebview.default_marker_name
            print(f'Set to use defaul marker line: {marker}')

        # Set
        self.set_marker(marker)

        # If regions are pre-set, add them
        if regions is not None:
            for r in regions:
                self.panel.region_tree.select_exact_match_in_tree(r)

    def marker_path(self):
        path = os.path.join(os.getenv('LOCALAPPDATA'), 'mapzebview', 'markers')

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def set_marker(self, name: str):

        if not debug:
            print(f'Set marker to {name}')
            self.marker_image = self.load_marker(name)
            # path = f'../../ants_registration/ants_registration/mapzebrain/{name}.tif'
            # self.marker_image = np.swapaxes(np.moveaxis(tifffile.imread(path), 0, 2), 0, 1)[:,:,:,None]

        else:
            # Debug image:
            im = np.zeros((300, 700, 200, 3)).astype(np.uint8)
            im[:150, :, :, 0] += 80
            im[:, :350, :, 1] += 80
            im[:, :, :100, 2] += 80

            self.marker_image = im

        self.sig_marker_image_updated.emit()

    def load_marker(self, name: str) -> np.ndarray:

        # Get file name from URL
        file_name = self.marker_structure[name].split('/')[-1].split('.')[0]

        file_path = os.path.join(self.marker_path(), f'{file_name}.tif')
        print(f'Load marker line {name} from {file_path}')
        if not os.path.exists(file_path):
            self.download_marker(name)

        return np.swapaxes(np.moveaxis(tifffile.imread(file_path), 0, 2), 0, 1)[:, :, :, None]

    def download_marker(self, name: str):

        url = self.marker_structure[name]
        zip_path = os.path.join(self.marker_path(), f'{name}.zip')
        print(f'Download marker data for {name} from {url}')
        urllib.request.urlretrieve(url, zip_path)

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.marker_path())

        # Remove ZIP
        os.remove(zip_path)

    def region_path(self) -> str:
        path = os.path.join(os.getenv('LOCALAPPDATA'), 'mapzebview', 'regions')

        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def add_region(self, name: str):
        print(f'Add region {name}')

        self.regions[name] = self.load_region(name)

        print(f'Region {name} added')
        self.sig_regions_updated.emit()

    def remove_region(self, name: str):
        print(f'Remove region {name}')

        if name in self.regions:
            del self.regions[name]

        self.sig_regions_updated.emit()

    def load_region(self, name: str) -> Tuple[np.ndarray, stl.Mesh]:

        name_str = name.replace(' ', '_')
        file_path = os.path.join(self.region_path(), f'{name_str}.tif')
        print(f'Load region {file_path}')

        if not os.path.exists(file_path):
            self.download_region(name_str)

        im = np.swapaxes(np.moveaxis(tifffile.imread(file_path), 0, 2), 0, 1)

        try:
            mesh = stl.Mesh.from_file(os.path.join(self.region_path(), f'{name_str}.stl'))
        except FileNotFoundError as e:
            print(f'WARNING: no mesh data for {name}')
            mesh = None

        return im, mesh

    def download_region(self, name_str: str):

        url = f'https://api.mapzebrain.org/media/Regions/v2.0.1/{name_str}/{name_str}.tif'
        print(f'Download region data for {name_str} from {url}')
        urllib.request.urlretrieve(url, os.path.join(self.region_path(), f'{name_str}.tif'))

        try:
            url_stl = f'https://api.mapzebrain.org/media/Regions/v2.0.1/{name_str}/{name_str}.stl'
            print(f'Download region mesh data for {name_str} from {url_stl}')
            urllib.request.urlretrieve(url_stl, os.path.join(self.region_path(), f'{name_str}.stl'))

        except urllib.request.HTTPError as _:
            print('WARNING: Failed to load region mesh data')

    def update_region_color(self, name: str, color: Union[list, tuple]):
        self.region_colors[name] = [*color,]

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



