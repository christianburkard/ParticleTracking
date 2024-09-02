#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

# Functions
from functions import preprocess_image, predict

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path(Path.cwd().parent, 'data')
model_name = "model-weights_0512.h5"
paths = list(data_path.glob("**/*.jpg"))
idx = 23

# Patches
size = int(model_name[14:18])
overlap = size // 4 # overlap between patches

#%% Predict -------------------------------------------------------------------

# Open & preprocess image
img = io.imread(paths[idx]).astype("float32")
img = np.mean(img, axis=2) # RGB to float
img = preprocess_image(img)

# Predict (shell & cores)
sProbs, cProbs = predict(img, size, overlap, model_name)

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(img, contrast_limits=(0.1, 1), opacity=0.33)
viewer.add_image(sProbs, contrast_limits=(0, 1), blending="additive", colormap="bop blue")
viewer.add_image(cProbs, contrast_limits=(0, 1), blending="additive", colormap="bop orange")