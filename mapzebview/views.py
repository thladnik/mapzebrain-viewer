from __future__ import annotations

from abc import abstractmethod
from typing import Dict

import numpy as np
import pyqtgraph as pg
from pyqtgraph import Vector, opengl as gl
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from mapzebview import config

try:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:
    pass


class SecionView(pg.ImageView):

    direction_label: str = None

    sig_index_changed = QtCore.Signal(int)

    last_idx: int = -1

    def __init__(self, parent):
        level_mode = 'rgba' if config.debug else 'mono'
        pg.ImageView.__init__(self, parent=parent, discreteTimeLine=True, levelMode=level_mode)

        # Disable unnecessary UI
        self.ui.histogram.hide()
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()

        # Add region image item
        self.region_image_items = {}
        self.scatter_items: Dict[str, pg.ScatterPlotItem] = {}
        self.scatter_coordinates: Dict[str, np.ndarray] = {}

        # Add lines
        self.vline = VerticalLine(self)
        self.view.addItem(self.vline)

        self.hline = HorizontalLine(self)
        self.view.addItem(self.hline)

        # Connect
        self.sigTimeChanged.connect(self.time_changed)
        self.sig_index_changed.connect(self.update_regions)
        self.sig_index_changed.connect(self.update_scatter)

    @abstractmethod
    def get_region_slice(self, region: np.ndarray):
        pass

    @abstractmethod
    def get_coordinate_slice(self, points: np.ndarray) -> np.ndarray:
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
        for i, (name, (region, _)) in enumerate(config.regions.items()):

            image_item = self.region_image_items.get(name)

            # Create
            if image_item is None:
                image_item = pg.ImageItem()
                image_item.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_ColorDodge)
                self.view.addItem(image_item)
                self.region_image_items[name] = image_item

            # Set colormap
            color = config.region_colors[name]
            cmap = pg.ColorMap(pos=[0., 1.], color=[[0, 0, 0], color], linearize=True)
            image_item.setColorMap(cmap)
            # image_item.setColorMap('CET-L14')

            region_slice = self.get_region_slice(region)
            image_item.setImage(region_slice)
            image_item.show()

    def add_scatter(self, tree_item: QtWidgets.QTreeWidgetItem):

        name = tree_item.name

        if name not in self.scatter_items:
            # Add scatter plot item for ROI display
            scatter_item = pg.ScatterPlotItem(symbol='o', size=6)
            scatter_item.setPen(pg.mkPen({'color': 'black', 'width': 1}))
            scatter_item.setBrush(pg.mkBrush(color=tree_item.color))
            self.view.addItem(scatter_item)
            self.scatter_items[name] = scatter_item
            self.scatter_coordinates[name] = tree_item.coordinates

        self.update_scatter()

    def update_scatter(self):

        for name in self.scatter_items:

            scatter_item = self.scatter_items[name]
            scatter_coordinates = self.scatter_coordinates[name]

            data_slice = self.get_coordinate_slice(scatter_coordinates)
            scatter_item.setData(*data_slice.T)

    def update_scatter_color(self, tree_item: QtWidgets.QTreeWidgetItem):

        name = tree_item.name

        if name not in self.scatter_items:
            return

        scatter_item = self.scatter_items[name]
        scatter_item.setBrush(pg.mkBrush(color=tree_item.color))

    def remove_scatter(self, tree_item: QtWidgets.QTreeWidgetItem):
        name = tree_item.name
        if name in self.scatter_items:
            scatter_item = self.scatter_items[name]
            self.view.removeItem(scatter_item)
            del self.scatter_items[name]

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
        self.setImage(config.marker_image[:, :, ::-1], axes={'t': 0, 'x': 1, 'y': 2, 'c': 3})
        self.timeLine.setPos(config.marker_image.shape[0] // 2)

    def get_region_slice(self, region: np.ndarray):
        return np.swapaxes(region[self.currentIndex, :, :], 0, 1)[::-1, :]

    def get_coordinate_slice(self, points: np.ndarray) -> np.ndarray:

        data_slice = points[points[:, 0].astype(int) == self.currentIndex, 1:]
        data_slice[:, 1] = config.marker_image.shape[2] - data_slice[:, 1]

        return data_slice

    def ymax(self):
        return config.marker_image.shape[2]


class CoronalView(SecionView):

    def update_marker_image(self):
        self.setImage(config.marker_image[::-1, :, :], axes={'t': 2, 'x': 1, 'y': 0, 'c': 3})
        self.timeLine.setPos(config.marker_image.shape[2] // 2)

    def get_region_slice(self, region: np.ndarray):
        return region[::-1, :, self.currentIndex]

    def get_coordinate_slice(self, points: np.ndarray) -> np.ndarray:

        data_slice = points[points[:, 2].astype(int) == self.currentIndex, :2][:, ::-1]
        data_slice[:, 1] = self.ymax() - data_slice[:, 1]

        return data_slice

    def ymax(self):
        return config.marker_image.shape[0]


class TransversalView(SecionView):

    def update_marker_image(self):
        self.setImage(config.marker_image[:, :, ::-1], axes={'t': 1, 'x': 0, 'y': 2, 'c': 3})
        self.timeLine.setPos(config.marker_image.shape[1] // 2)

    def get_region_slice(self, region: np.ndarray):
        return np.swapaxes(region[:, self.currentIndex, :], 0, 1)[::-1, :]

    def get_coordinate_slice(self, points: np.ndarray) -> np.ndarray:

        data_slice = points[points[:, 1].astype(int) == self.currentIndex, ::2]
        data_slice[:, 1] = config.marker_image.shape[2] - data_slice[:, 1]

        return data_slice

    def ymax(self):
        return config.marker_image.shape[2]


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
    scatter_items: Dict[str, gl.GLScatterPlotItem] = {}

    volume_bounds: np.ndarray = None
    plane_color = np.array([200, 200, 100, 5])
    plane_thickness = 3

    def __init__(self, parent):
        gl.GLViewWidget.__init__(self, parent=parent, rotationMethod='euler')
        self.opts['fov'] = 1.

        sag_data = np.ones((1, 100, 100, 4)) * self.plane_color[None, None, None, :]
        self.saggital_plane = gl.GLVolumeItem(sag_data)
        self.saggital_plane.setGLOptions('additive')
        self.addItem(self.saggital_plane)

        cor_data = np.ones((100, 100, 1, 4)) * self.plane_color[None, None, None, :]
        self.coronal_plane = gl.GLVolumeItem(cor_data)
        self.coronal_plane.setGLOptions('additive')
        self.addItem(self.coronal_plane)

        trans_data = np.ones((100, 1, 100, 4)) * self.plane_color[None, None, None, :]
        self.transverse_plane = gl.GLVolumeItem(trans_data)
        self.transverse_plane.setGLOptions('additive')
        self.addItem(self.transverse_plane)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):

        accelerate = QtGui.Qt.KeyboardModifier.ShiftModifier in ev.modifiers()

        # Translate
        step_size = 1
        if ev.key() in (QtCore.Qt.Key.Key_W, QtCore.Qt.Key.Key_A,
                        QtCore.Qt.Key.Key_S, QtCore.Qt.Key.Key_D):
            x = y = 0
            if ev.key() == QtCore.Qt.Key.Key_W:
                y -= step_size + accelerate * 5
            if ev.key() == QtCore.Qt.Key.Key_S:
                y += step_size + accelerate * 5
            if ev.key() == QtCore.Qt.Key.Key_A:
                x -= step_size + accelerate * 5
            if ev.key() == QtCore.Qt.Key.Key_D:
                x += step_size + accelerate * 5

            self.pan(x, y, 0, relative='view')

        # Rotate
        rot_size = 1
        if ev.key() in (QtCore.Qt.Key.Key_Q, QtCore.Qt.Key.Key_E,
                        QtCore.Qt.Key.Key_R, QtCore.Qt.Key.Key_F):

            a = e = 0
            if ev.key() == QtCore.Qt.Key.Key_Q:
                a -= rot_size + accelerate * 5
            if ev.key() == QtCore.Qt.Key.Key_E:
                a += rot_size + accelerate * 5
            if ev.key() == QtCore.Qt.Key.Key_R:
                e += rot_size + accelerate * 5
            if ev.key() == QtCore.Qt.Key.Key_F:
                e -= rot_size + accelerate * 5

            self.orbit(a, e)

        gl.GLViewWidget.keyPressEvent(self, ev)

    def marker_image_updated(self):

        volume_shape = config.marker_image.shape[:3]

        self.volume_bounds = np.array(volume_shape, dtype=np.float32)

        # Set plane extents
        sag_data = (np.ones((self.plane_thickness, volume_shape[1], volume_shape[2], 4)) * self.plane_color[None, None, None, :]).astype(np.uint8)
        self.saggital_plane.setData(sag_data)

        cor_data = (np.ones((volume_shape[0], volume_shape[1], self.plane_thickness, 4)) * self.plane_color[None, None, None, :]).astype(np.uint8)
        self.coronal_plane.setData(cor_data)

        trans_data = np.ones((volume_shape[0], self.plane_thickness, volume_shape[2], 4))
        trans_data *= self.plane_color[None, None, None, :]
        self.transverse_plane.setData(trans_data)

        # Translate
        self.set_saggital_position(config.marker_image.shape[0] // 2)
        self.set_coronal_position(config.marker_image.shape[2] // 2)
        self.set_transverse_position(config.marker_image.shape[1] // 2)

        # Set camera
        self.setCameraPosition(pos=Vector(*[int(i // 2) for i in config.marker_image.shape[:3]]), distance=60000)

    def set_saggital_position(self, current_idx: int):

        if self.volume_bounds is None:
            return

        self.saggital_plane.resetTransform()
        self.saggital_plane.translate(self.volume_bounds[0]-current_idx, 0, 0)

    def set_coronal_position(self, current_idx: int):

        if self.volume_bounds is None:
            return

        self.coronal_plane.resetTransform()
        self.coronal_plane.translate(0, 0, current_idx)

    def set_transverse_position(self, current_idx: int):

        if self.volume_bounds is None:
            return

        self.transverse_plane.resetTransform()
        self.transverse_plane.translate(0, current_idx, 0)

    def update_scatter(self):

        # Hide all items
        for scatter_item in self.scatter_items.values():
            scatter_item.hide()

        # Go through all the ones that should be displayed
        for name, roi_tree_item in config.roi_set_items.items():

            if name not in self.scatter_items:
                coordinates = roi_tree_item.coordinates.copy()
                coordinates[:, 0] = self.volume_bounds[0] - coordinates[:, 0]

                scatter_item = gl.GLScatterPlotItem(size=5, color=(1., 1., 1., 1.0), pos=coordinates, pxMode=False)
                scatter_item .setGLOptions('additive')
                self.addItem(scatter_item)
                self.scatter_items[name] = scatter_item

            scatter_item = self.scatter_items[name]

            # Set color
            color = roi_tree_item.color
            scatter_item.setData(color=color.getRgbF())

            # Show
            scatter_item.show()

    def update_regions(self):

        for mesh_item in self.mesh_items.values():
            mesh_item.hide()

        for i, (name, (_, region_mesh)) in enumerate(config.regions.items()):

            if name not in self.mesh_items:
                vecs = region_mesh.vectors.copy()
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
            color = config.region_colors[name].getRgbF()

            # Reduce alpha for volume view
            mesh_item.setColor((*color[:3], color[3] / 10))


class PrettyView(QtWidgets.QWidget):

    ortho_views = {
        'xy': (-90, 90),
        'xz': (-90, 0),
        'yz': (0, 0),
        '-xy': (90, -90),
        '-xz': (90, 0),
        '-yz': (180, 0),
    }

    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.setWindowFlag(QtCore.Qt.WindowType.Window)
        self.setMinimumSize(1000, 600)
        self.setLayout(QtWidgets.QHBoxLayout())

        self.region_items: Dict[str, Poly3DCollection] = {}

        # Add property tuner
        self.property_tuner = QtWidgets.QGroupBox('Properties', self)
        self.property_tuner.setMaximumWidth(250)
        self.property_tuner.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(self.property_tuner)

        self.property_tuner.layout().addWidget(QtWidgets.QLabel('Set orthogonal view'))
        self.view_selector = QtWidgets.QWidget()
        self.view_selector.setLayout(QtWidgets.QHBoxLayout())
        self.property_tuner.layout().addWidget(self.view_selector)
        self.view_btns = []
        for view in ['xy', 'xz', 'yz', '-xy', '-xz', '-yz']:
            btn = QtWidgets.QPushButton(view.upper())
            btn.clicked.connect(self._get_set_view_fun(view))
            self.view_btns.append(btn)
            self.view_selector.layout().addWidget(btn)

        self.property_tuner.layout().addStretch()

        # Save to file button
        self.save_to_file_btn = QtWidgets.QPushButton('Save to file')
        self.save_to_file_btn.clicked.connect(self.save_to_file)
        self.property_tuner.layout().addWidget(self.save_to_file_btn)

        # Add figure canvas
        self.fig_canvas = FigureCanvasQTAgg()
        self.ax = self.fig_canvas.figure.add_subplot(projection='3d')
        self.ax.set_proj_type('ortho')
        self.layout().addWidget(self.fig_canvas)

        # pos_marker_color = (200 / 255, 200 / 255, 100 / 255)
        pos_marker_color = (0 / 255, 0 / 255, 0 / 255)
        pos_marker_linewidth = 1.

        # Add lines
        self.xline, = self.ax.plot(*np.zeros((3, 2)), color=pos_marker_color, linewidth=pos_marker_linewidth)
        self.yline, = self.ax.plot(*np.zeros((3, 2)), color=pos_marker_color, linewidth=pos_marker_linewidth)
        self.zline, = self.ax.plot(*np.zeros((3, 2)), color=pos_marker_color, linewidth=pos_marker_linewidth)

        self.ax.axes.set_axis_off()
        self.fig_canvas.figure.tight_layout()

        # Show
        self.show()

        # Update view
        camera_params = config.window.volume_view.cameraParams()
        self.update_current_position()
        self.set_view(azim=camera_params['azimuth'], elev=camera_params['elevation'])
        self.update_regions()
        self.update_rois()

    def update_regions(self):

        for name, (_, region_mesh) in config.regions.items():
            if name not in self.region_items:
                vecs = region_mesh.vectors.copy()
                # Invert X for GL view
                vecs[:, :, 0] = config.marker_image.shape[0] - vecs[:, :, 0]
                coll = Poly3DCollection(vecs, facecolors='black', edgecolors='none')
                self.ax.add_collection3d(coll)
                self.region_items[name] = coll

            coll = self.region_items[name]
            color = config.region_colors[name].getRgbF()

            # Set to lower alpha value
            coll.set_facecolor((*color[:3], color[3] / 10))

        self.fig_canvas.draw()

    def update_rois(self):

        for name, roi_tree_item in config.roi_set_items.items():
            coordinates = roi_tree_item.coordinates.copy()
            coordinates[:, 0] = config.marker_image.shape[0] - coordinates[:, 0]
            color = roi_tree_item.color

            self.ax.scatter(*coordinates.T, s=3, c=[color.getRgbF()], edgecolor='none')
        self.fig_canvas.draw()

    def update_current_position(self):

        current_position = (config.window.saggital_view.currentIndex,
                            config.window.transverse_view.currentIndex,
                            config.window.coronal_view.currentIndex)

        volume_shape = config.marker_image.shape[:3]

        self.ax.set_box_aspect(volume_shape)

        self.ax.set_xlim(current_position[0] - volume_shape[0] / 2, current_position[0] + volume_shape[0] / 2)
        self.ax.set_ylim(current_position[1] - volume_shape[1] / 2, current_position[1] + volume_shape[1] / 2)
        self.ax.set_zlim(current_position[2] - volume_shape[2] / 2, current_position[2] + volume_shape[2] / 2)

        vol_half_range = np.array(volume_shape) / 2

        xline = np.array([[volume_shape[0] - current_position[0] - vol_half_range[0], current_position[1], current_position[2]],
                          [volume_shape[0] - current_position[0] + vol_half_range[0], current_position[1], current_position[2]]])

        yline = np.array([[volume_shape[0] - current_position[0], current_position[1] - vol_half_range[1], current_position[2]],
                          [volume_shape[0] - current_position[0], current_position[1] + vol_half_range[1], current_position[2]]])

        zline = np.array([[volume_shape[0] - current_position[0], current_position[1], current_position[2] - vol_half_range[2]],
                          [volume_shape[0] - current_position[0], current_position[1], current_position[2] + vol_half_range[2]]])

        self.xline.set_data_3d(*xline.T)
        self.yline.set_data_3d(*yline.T)
        self.zline.set_data_3d(*zline.T)

        self.fig_canvas.draw()

    def _get_set_view_fun(self, view: str):

        def _set_view():
            return self.set_view(view=view)

        return _set_view

    def set_view(self, view: str = None, azim: float = None, elev: float = None):

        if view is not None:
            print(f'Set view to {view}')
            azim = self.ortho_views[view][0]
            elev = self.ortho_views[view][1]

        if azim is None or elev is None:
            raise ValueError('Azimuth and elevation need to be set')

        self.ax.view_init(azim=azim, elev=elev)
        self.fig_canvas.draw()

    def save_to_file(self):

        # TODO: file selection dialog

        self.fig_canvas.figure.savefig('test.png', dpi=300)
