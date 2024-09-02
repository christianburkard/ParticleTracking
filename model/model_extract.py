#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path
from functions import preprocess_image
from bdtools.patch import extract_patches

#%% Inputs --------------------------------------------------------------------

# Path
data_path  = Path(Path.cwd().parent, 'data')
train_path = Path(Path.cwd().parent, 'data', 'train')
img_paths = list(data_path.glob("**/*.jpg"))

# Selection
nPatch = 2 # number of patch(es) extracted per image 
size = 1024 # size of extract patches
overlap = 0 # overlap between patches

#%% Extract -------------------------------------------------------------------

for path in img_paths:
    
    # Open & preprocess image
    img = io.imread(path)
    img = np.mean(img, axis=2) # RGB to float
    img = preprocess_image(img)
    
    # Extract patches
    patches = extract_patches(img, size, overlap)
    
    # Select & save patches
    np.random.seed(42)
    idxs = np.random.choice(range(0, len(patches)), size=nPatch, replace=False)
    for idx in idxs:
        patch = patches[idx]
        name = path.name.replace(".jpg", f"_p{idx:02d}.tif")
        io.imsave(
            Path(train_path, name),
            patch.astype("float32"),
            check_contrast=False
            )
        