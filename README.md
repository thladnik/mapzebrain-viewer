# mapzebrain-viewer
Graphical user interface for viewing markers and regions of the [MapZeBrain atlas](https://mapzebrain.org/) - for quick and simple visualization of ROI locations against the atlas data in Python. 
This viewer uses the provided MapZeBrain API, but is not affiliated with the project in any way.

![image](https://github.com/user-attachments/assets/9ac8bc19-f066-4376-ba53-1c8b26170aac)

It enables the export of anatomical images for different views, while using `matplotlib` for keeping with the style of other publication figures:
![image](https://github.com/user-attachments/assets/41eb1bc7-f755-4cde-8bc2-4dc650e5557e)



## Installation
Run `pip install git+https://github.com/thladnik/mapzebrain-viewer` in your Python environment.

In order to enable exports of pretty `matplotlib` renderings of the data, either run `pip install matplotlib` or install mapzebview with optional dependencies `pip install git+https://github.com/thladnik/mapzebrain-viewer[pretty]` 


## Usage

### Simple
Either run it from the command line useing the `mapzebview` command or start it from within your Python code:
```Python
import mapzebview

mapzebview.run()
```
By default the interface uses the jf5Tg line. This behavior can be changed by changing the variable `mapzebview.config.default_marker_name: str`

### Advanced

To use a different marker line:
```Python
import mapzebview

mapzebview.run(marker='mpn212Tg')
```

To pre-select regions (can also be toggled in UI):
```Python
import mapzebview

mapzebview.run(regions=['pretectum', 'periventricular layer', 'oculomotor nucleus'])
```

To directly plot ROI locations from within the analysis:
```Python
import mapzebview

# Option 1
import pandas as pd
roi_coordinates = pd.read_hdf('path/to/file.hdf5')

# Option 2
import numpy as np
roi_coordinates = np.load('path/to/file.npy')

mapzebview.run(rois=roi_coordinates)  # pandas.DataFrame with x/y/z columns or numpy.ndarray with shape Nx3
```
To plot ROI locations from a file directly, .hdf5 or .npy files can also be dropped into the interface

To plot multiple sets of ROIs at once:
```Python
import mapzebview

import pandas as pd
import numpy as np

roi_sets = {
    'ROI set 1': pd.read_hdf('path/to/file.hdf5'), 
    'ROI set 2': np.load('path/to/file.npy'), 
    'Some other set': np.load('path/to/other/file.npy')
}

mapzebview.run(rois=roi_sets)  # dictionary containing pairs of name: pandas.DataFrame/numpy.ndarray
```
