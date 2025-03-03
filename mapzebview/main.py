from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from typing import Dict, List, Tuple, Union

import colorcet as cc
import numpy as np
import pandas as pd
import pyqtgraph as pg
import stl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import tifffile

from mapzebview import config
from mapzebview.regions import region_structure
from mapzebview.views import CoronalView, PrettyView, SaggitalView, TransversalView, VolumeView

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    print('Matplotlib not installed. No pretty view export available')


class Window(QtWidgets.QMainWindow):

    sig_regions_updated = QtCore.Signal()
    sig_marker_image_updated = QtCore.Signal()

    def __init__(self,
                 marker: str = None,
                 regions: List[str] = None,
                 rois: Union[np.ndarray, pd.DataFrame, Dict[str, Union[np.ndarray, pd.DataFrame]]] = None):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(1600, 800)
        self.setWindowTitle('MapZeBrain Viewer')
        config.window = self

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
        self.saggital_view.sig_index_changed.connect(self.volume_view.set_saggital_position)

        self.coronal_view.sig_index_changed.connect(self.saggital_view.update_hline)
        self.coronal_view.sig_index_changed.connect(self.transverse_view.update_hline)
        self.coronal_view.hline.sig_position_changed.connect(self.saggital_view.update_index)
        self.coronal_view.vline.sig_position_changed.connect(self.transverse_view.update_index)
        self.coronal_view.sig_index_changed.connect(self.volume_view.set_coronal_position)

        self.transverse_view.sig_index_changed.connect(self.saggital_view.update_vline)
        self.transverse_view.sig_index_changed.connect(self.coronal_view.update_vline)
        self.transverse_view.hline.sig_position_changed.connect(self.coronal_view.update_index)
        self.transverse_view.vline.sig_position_changed.connect(self.saggital_view.update_index)
        self.transverse_view.sig_index_changed.connect(self.volume_view.set_transverse_position)

        # Connect scatter plot updates
        # Saggital view
        self.panel.roi_list.sig_item_added.connect(self.saggital_view.add_scatter)
        self.panel.roi_list.sig_item_shown.connect(self.saggital_view.add_scatter)
        self.panel.roi_list.sig_item_color_changed.connect(self.saggital_view.update_scatter_color)
        self.panel.roi_list.sig_item_removed.connect(self.saggital_view.remove_scatter)
        self.panel.roi_list.sig_item_hidden.connect(self.saggital_view.remove_scatter)

        # Coronal view
        self.panel.roi_list.sig_item_added.connect(self.coronal_view.add_scatter)
        self.panel.roi_list.sig_item_shown.connect(self.coronal_view.add_scatter)
        self.panel.roi_list.sig_item_color_changed.connect(self.coronal_view.update_scatter_color)
        self.panel.roi_list.sig_item_removed.connect(self.coronal_view.remove_scatter)
        self.panel.roi_list.sig_item_hidden.connect(self.coronal_view.remove_scatter)

        # Transverse view
        self.panel.roi_list.sig_item_added.connect(self.transverse_view.add_scatter)
        self.panel.roi_list.sig_item_shown.connect(self.transverse_view.add_scatter)
        self.panel.roi_list.sig_item_color_changed.connect(self.transverse_view.update_scatter_color)
        self.panel.roi_list.sig_item_removed.connect(self.transverse_view.remove_scatter)
        self.panel.roi_list.sig_item_hidden.connect(self.transverse_view.remove_scatter)

        # Volumetric view
        self.panel.roi_list.sig_item_added.connect(self.volume_view.update_scatter)
        self.panel.roi_list.sig_item_shown.connect(self.volume_view.update_scatter)
        self.panel.roi_list.sig_item_color_changed.connect(self.volume_view.update_scatter)
        self.panel.roi_list.sig_item_removed.connect(self.volume_view.update_scatter)
        self.panel.roi_list.sig_item_hidden.connect(self.volume_view.update_scatter)

        marker_catalog_path = os.path.join(config.marker_path(), 'markers_catalog.json')
        if not os.path.exists(marker_catalog_path):
            print('Get marker catalog from MapZeBrain')
            urllib.request.urlretrieve('https://api.mapzebrain.org/media/downloads/Lines/markers_catalog.json',
                                       marker_catalog_path)

        print('Load marker catalog')
        markers_catalog = json.load(open(marker_catalog_path, 'r'))
        self.marker_structure = {d['name']: d['stack'] for d in markers_catalog}

        # Make sure marker line is set
        if marker is None:
            marker = config.default_marker_name
            print(f'Set to use defaul marker line: {marker}')

        # Set
        self.set_marker(marker)

        # If regions are pre-set, add them
        if regions is not None:
            for r in regions:
                self.panel.region_tree.select_exact_match_in_tree(r)

        # If ROIs are pre-set, add those
        if rois is not None:
            if isinstance(rois, (np.ndarray, pd.DataFrame)):
                rois = {None: rois}

            for name, data in rois.items():
                self.panel.roi_list.add_roi_set(data=data, name=name)

        # Show
        self.show()

    def set_marker(self, name: str):

        if not config.debug:
            print(f'Set marker to {name}')
            config.marker_image = self.load_marker(name)

        else:
            # Debug image:
            im = np.zeros((300, 700, 200, 3)).astype(np.uint8)
            im[:150, :, :, 0] += 80
            im[:, :350, :, 1] += 80
            im[:, :, :100, 2] += 80

            config.marker_image = im

        self.sig_marker_image_updated.emit()

    def load_marker(self, name: str) -> np.ndarray:

        # Get file name from URL
        file_name = self.marker_structure[name].split('/')[-1].split('.')[0]

        file_path = os.path.join(config.marker_path(), f'{file_name}.tif')
        print(f'Load marker line {name} from {file_path}')
        if not os.path.exists(file_path):
            self.download_marker(name)

        return np.swapaxes(np.moveaxis(tifffile.imread(file_path), 0, 2), 0, 1)[:, :, :, None]

    def download_marker(self, name: str):

        url = self.marker_structure[name]
        zip_path = os.path.join(config.marker_path(), f'{name}.zip')
        print(f'Download marker data for {name} from {url}')
        urllib.request.urlretrieve(url, zip_path)

        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.marker_path())

        # Remove ZIP
        os.remove(zip_path)

    def add_region(self, name: str):
        print(f'Add region {name}')

        config.regions[name] = self.load_region(name)

        print(f'Region {name} added')
        self.sig_regions_updated.emit()

    def remove_region(self, name: str):
        print(f'Remove region {name}')

        if name in config.regions:
            del config.regions[name]

        self.sig_regions_updated.emit()

    def load_region(self, name: str) -> Tuple[np.ndarray, stl.Mesh]:

        name_str = name.replace(' ', '_')
        file_path = os.path.join(config.region_path(), f'{name_str}.tif')
        print(f'Load region volume {file_path}')

        if not os.path.exists(file_path):
            self.download_region(name_str)

        im = np.swapaxes(np.moveaxis(tifffile.imread(file_path), 0, 2), 0, 1)

        try:
            path_stl = os.path.join(config.region_path(), f'{name_str}.stl')
            print(f'Load region mesh {path_stl}')
            mesh = stl.Mesh.from_file(path_stl)
        except FileNotFoundError as e:
            print(f'WARNING: no mesh data for {name}')
            mesh = None

        return im, mesh

    @staticmethod
    def download_region(name_str: str):

        url = f'https://api.mapzebrain.org/media/Regions/v2.0.1/{name_str}/{name_str}.tif'
        print(f'Download region data for {name_str} from {url}')
        urllib.request.urlretrieve(url, os.path.join(config.region_path(), f'{name_str}.tif'))

        try:
            url_stl = f'https://api.mapzebrain.org/media/Regions/v2.0.1/{name_str}/{name_str}.stl'
            print(f'Download region mesh data for {name_str} from {url_stl}')
            urllib.request.urlretrieve(url_stl, os.path.join(config.region_path(), f'{name_str}.stl'))

        except urllib.request.HTTPError as _:
            print('WARNING: Failed to load region mesh data')

    def update_region_color(self, name: str, color: QtGui.QColor):
        config.region_colors[name] = color

        self.sig_regions_updated.emit()


class ControlPanel(QtWidgets.QGroupBox):

    sig_region_color_changed = QtCore.Signal(str, QtGui.QColor)
    sig_roi_data_changed = QtCore.Signal()

    pretty_view: PrettyView = None

    def __init__(self, parent: Window):
        QtWidgets.QGroupBox.__init__(self, 'Navigation', parent=parent)

        self.setMinimumWidth(300)
        self.setMaximumWidth(400)
        self.setLayout(QtWidgets.QVBoxLayout())

        self.region_tree = RegionTreeWidget()
        self.region_tree.build_tree(region_structure)
        self.region_tree.sig_item_selected.connect(self.region_selected)
        self.region_tree.sig_item_removed.connect(self.region_removed)
        self.region_tree.sig_item_color_changed.connect(self.region_color_changed)
        self.layout().addWidget(self.region_tree)

        self.roi_list = RoiWidget(self)
        self.layout().addWidget(self.roi_list)

        self.export_to_image_btn = QtWidgets.QPushButton('Export view to image')
        self.export_to_image_btn.clicked.connect(self.open_pretty_view)
        self.layout().addWidget(self.export_to_image_btn)

    def open_pretty_view(self):

        if self.pretty_view is not None:
            self.pretty_view.close()

        self.pretty_view = PrettyView(self)

    def region_selected(self, item: QtWidgets.QTreeWidgetItem):
        name = self.region_tree.get_item_name(item)
        config.window.add_region(name)

    def region_color_changed(self, item: QtWidgets.QTreeWidgetItem):
        name = self.region_tree.get_item_name(item)
        color = self.region_tree.get_item_color(item)

        self.sig_region_color_changed.emit(name, color)

    def region_removed(self, item: QtWidgets.QTreeWidgetItem):
        name = item.data(0, RegionTreeWidget.UniqueNameRole)
        config.window.remove_region(name)


class RegionTreeWidget(QtWidgets.QWidget):

    ContinuousIdRole = 40
    UniqueNameRole = 50
    ColorRole = 60

    sig_item_selected = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_removed = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_color_changed = QtCore.Signal(QtWidgets.QTreeWidgetItem)

    item_count = 0

    def __init__(self, *args):
        QtWidgets.QWidget.__init__(self, *args)
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
            color_btn.clicked.connect(lambda: self.open_colorpicker(tree_item))
            self.tree_widget.setItemWidget(tree_item, 2, color_btn)
            color_btn = self.tree_widget.itemWidget(tree_item, 2)
            color_btn.setDisabled(True)

            # Set item data
            tree_item.setData(0, RegionTreeWidget.UniqueNameRole, name)
            tree_item.setData(0, RegionTreeWidget.ContinuousIdRole, self.item_count)

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
            item_text = item.data(0, RegionTreeWidget.UniqueNameRole)

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
            item_text = item.data(0, RegionTreeWidget.UniqueNameRole)

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

            self.selected_items.append(tree_item)

            # Set select button
            select_btn = self.tree_widget.itemWidget(tree_item, 1)
            select_btn.setText('hide')

            # Set state and coloron color picker button
            color = cc.glasbey_hv[self.get_item_continuous_id(tree_item)]
            color = QtGui.QColor.fromRgbF(*color, 1.0)
            self.set_item_color(tree_item, color)

            # Enable color picker button
            color_btn = self.tree_widget.itemWidget(tree_item, 2)
            color_btn.setDisabled(False)

            # Emit signal
            self.sig_item_selected.emit(tree_item)

        # Remove item
        else:

            # Update label
            label = self.tree_widget.itemWidget(tree_item, 0)
            label.setStyleSheet('')

            # Update select button
            select_btn = self.tree_widget.itemWidget(tree_item, 1)
            select_btn.setText('show')

            # Update color picker button
            color_btn = self.tree_widget.itemWidget(tree_item, 2)
            color_btn.setStyleSheet('')
            color_btn.setDisabled(True)

            # Remove item
            self.selected_items.remove(tree_item)

            # Emit signal
            self.sig_item_removed.emit(tree_item)

    def open_colorpicker(self, tree_item: QtWidgets.QTreeWidgetItem):

        current_color = self.get_item_color(tree_item)

        new_color = QtWidgets.QColorDialog().getColor(initial=current_color,
                                                      options=QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)

        if new_color.isValid():
            print(f'New color set for {self.get_item_name(tree_item)}')
            self.set_item_color(tree_item, new_color)

    def update_colorpicker_button(self, tree_item: QtWidgets.QTreeWidgetItem):

        color = self.get_item_color(tree_item)
        color_btn = self.tree_widget.itemWidget(tree_item, 2)
        color_btn.setStyleSheet(f'background-color: rgba{color.getRgb()};')

    def get_item_name(self, tree_item: QtWidgets.QTreeWidgetItem):
        return tree_item.data(0, self.UniqueNameRole)

    def get_item_continuous_id(self, tree_item: QtWidgets.QTreeWidgetItem):
        return tree_item.data(0, self.ContinuousIdRole)

    def get_item_color(self, tree_item: QtWidgets.QTreeWidgetItem) -> QtGui.QColor:
        return tree_item.data(0, self.ColorRole)

    def set_item_color(self, tree_item: QtWidgets.QTreeWidgetItem, color: QtGui.QColor):
        tree_item.setData(0, self.ColorRole, color)

        self.sig_item_color_changed.emit(tree_item)


class RoiWidget(QtWidgets.QGroupBox):

    sig_path_added = QtCore.Signal(str)
    sig_item_shown = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_hidden = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_color_changed = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_added = QtCore.Signal(QtWidgets.QTreeWidgetItem)
    sig_item_removed = QtCore.Signal(QtWidgets.QTreeWidgetItem)

    item_count: int = 0
    selected_items: List[QtWidgets.QTreeWidgetItem] = []

    def __init__(self, parent=None):
        QtWidgets.QGroupBox.__init__(self, 'ROIs', parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(200)
        self.setMaximumHeight(300)
        self.setLayout(QtWidgets.QVBoxLayout())

        # Add drop label
        self.drop_label = QtWidgets.QLabel('Drop files to load...')
        self.drop_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(self.drop_label)

        # Add tree widget
        self.tree_widget = QtWidgets.QTreeWidget(self)
        self.tree_widget.headerItem().setText(0, 'ROI sets')
        self.tree_widget.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree_widget.headerItem().setText(1, '')
        self.tree_widget.header().resizeSection(1, 50)
        self.tree_widget.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.tree_widget.headerItem().setText(2, '')
        self.tree_widget.header().resizeSection(2, 25)
        self.tree_widget.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.tree_widget.header().setStretchLastSection(False)
        self.layout().addWidget(self.tree_widget)

        self.sig_path_added.connect(self.add_roi_set)
        self.sig_item_shown.connect(self.update_color_btn)
        self.sig_item_hidden.connect(self.update_color_btn)
        self.sig_item_color_changed.connect(self.update_color_btn)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            ext = event.mimeData().urls()[0].fileName().split('.')[-1]

            if ext.lower() in ['h5', 'hdf5', 'npy', 'json', 'csv']:
                event.accept()
            else:
                event.ignore()

        else:
            event.ignore()

    def dropEvent(self, event):

        for url in event.mimeData().urls():
            print(f'Load ROIs from {url}')
            ext = url.fileName().split('.')[-1]

            if ext in ['h5', 'hdf5', 'npy', 'json', 'csv']:
                self.add_roi_set(data=url.path().strip('/'))
            else:
                print(f'WARNING: unknown file type {ext}')

    def keyPressEvent(self, event, /):
        if event.key() == QtCore.Qt.Key.Key_Delete:
            current_item = self.tree_widget.currentItem()

            # Check if item is selected
            if current_item is None:
                return

            # Delete item
            print(f'Delete ROI set {current_item.text(0)}')

            # Remove from selection list
            if current_item in self.selected_items:
                self.selected_items.remove(current_item)

            # Remove from tree
            self.tree_widget.invisibleRootItem().removeChild(current_item)

            # Emit item removed signal
            self.sig_item_removed.emit(current_item)

        QtWidgets.QGroupBox.keyPressEvent(self, event)

    def load_roi_data(self, path: Union[str, os.PathLike]):

        ext = path.split('.')[-1]

        if ext in ['h5', 'hdf5']:
            data = pd.read_hdf(path)

        elif ext == 'npy':
            data = np.load(path)

        else:
            return

        return data

    def add_roi_set(self, data: Union[np.ndarray, pd.DataFrame, str, os.PathLike], name: str = None):

        # Load data from path
        if isinstance(data, (str, os.PathLike)):
            if name is None:
                name = data

            data = self.load_roi_data(data)

        if data is None:
            return

        # Unpack DataFrames
        if isinstance(data, pd.DataFrame):

            if len(data.columns) == 3:
                keys = data.columns

            else:
                keys = []
                for _n in ['x', 'y', 'z']:
                    _keys = [k.lower() for k in data.keys() if _n in k]
                    if len(_keys) > 0:
                        keys.append(_keys[0])

                if len(keys) != 3:
                    raise KeyError('No matching coordinate keys found for x/y/z')

            data = data[keys].values
            print(f'Found coordinate keys: {keys} for axes x/y/z respectively')

        # Deal with NaNs
        if np.any(np.isnan(data)):
            print('WARNING: coordinates contain NaN values')
            data = data[~np.any(np.isnan(data), axis=1), :]

        # Set name if none given
        if name is None:
            name = f'ROI set {self.item_count}'
            self.item_count += 1

        if '/' in name:
            name_short = name.split('/')[-1]
        elif '\\' in name:
            name_short = name.split('\\')[-1]
        else:
            name_short = name

        # Add tree item
        tree_item = QtWidgets.QTreeWidgetItem(self.tree_widget)
        tree_item.setToolTip(0, name)
        tree_item.name = name
        tree_item.coordinates = data
        self.tree_widget.addTopLevelItem(tree_item)
        # Start color
        color = cc.glasbey_hv[np.random.randint(len(cc.glasbey_hv))]
        tree_item.color = QtGui.QColor.fromRgbF(*color, 0.5)

        label = QtWidgets.QLabel(name_short, self.tree_widget)
        self.tree_widget.setItemWidget(tree_item, 0, label)

        # Add select button
        select_btn = QtWidgets.QPushButton('show')
        select_btn.setContentsMargins(0, 0, 0, 0)
        select_btn.clicked.connect(lambda: self.toggle_item(tree_item))
        self.tree_widget.setItemWidget(tree_item, 1, select_btn)

        # Add colorpicker button
        color_btn = QtWidgets.QPushButton(f' ')
        color_btn.setContentsMargins(0, 0, 0, 0)
        color_btn.clicked.connect(lambda: self.open_colorpicker(tree_item))
        self.tree_widget.setItemWidget(tree_item, 2, color_btn)

        # Emit item added signal
        self.sig_item_added.emit(tree_item)

        # Toggle on by default
        self.toggle_item(tree_item)

    def toggle_item(self, tree_item: QtWidgets.QTreeWidgetItem):

        # Add item
        if tree_item not in self.selected_items:

            self.selected_items.append(tree_item)
            config.roi_set_items[tree_item.name] = tree_item

            # Set select button
            select_btn = self.tree_widget.itemWidget(tree_item, 1)
            select_btn.setText('hide')

            # Enable color picker button
            color_btn = self.tree_widget.itemWidget(tree_item, 2)
            color_btn.setDisabled(False)

            # Emit signal
            self.sig_item_shown.emit(tree_item)

        # Remove item
        else:

            # Update label
            label = self.tree_widget.itemWidget(tree_item, 0)
            label.setStyleSheet('')

            # Update select button
            select_btn = self.tree_widget.itemWidget(tree_item, 1)
            select_btn.setText('show')

            # Update color picker button
            color_btn = self.tree_widget.itemWidget(tree_item, 2)
            color_btn.setStyleSheet('')
            color_btn.setDisabled(True)

            # Remove item
            self.selected_items.remove(tree_item)
            del config.roi_set_items[tree_item.name]

            # Emit signal
            self.sig_item_hidden.emit(tree_item)

    def open_colorpicker(self, tree_item: QtWidgets.QTreeWidgetItem):

        new_color = QtWidgets.QColorDialog().getColor(initial=tree_item.color,
                                                      options=QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)

        if new_color.isValid():
            tree_item.color = new_color
            self.sig_item_color_changed.emit(tree_item)

    def update_color_btn(self, tree_item: QtWidgets.QTreeWidgetItem):

        color = tree_item.color
        color_btn = self.tree_widget.itemWidget(tree_item, 2)

        if color_btn.isEnabled():
            color_btn.setStyleSheet(f'background-color: rgba{color.getRgb()};')
        else:
            color_btn.setStyleSheet('')


def run(rois: Union[np.ndarray, pd.DataFrame, Dict[str, Union[np.ndarray, pd.DataFrame]]] = None, marker: str = None, regions: List[str] = None):

    print('Open window')

    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')

    app = pg.mkQApp()

    win = Window(rois=rois, marker=marker, regions=regions)

    pg.exec()


if __name__ == '__main__':

    run()



