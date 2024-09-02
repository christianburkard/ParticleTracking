#%% Imports -------------------------------------------------------------------

import cv2
import time
import numpy as np
import pandas as pd
from skimage import io
from pathlib import Path
import segmentation_models as sm
from bdtools.patch import extract_patches, merge_patches

# Skimage
from skimage.measure import label
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.measure import regionprops
from skimage.exposure import adjust_gamma
from skimage.segmentation import watershed, clear_border
from skimage.morphology import disk, binary_erosion, remove_small_objects

# Scipy
from scipy.ndimage import distance_transform_edt

#%% Functions (preprocess) ----------------------------------------------------

def normalize_gcn(img):
    img = img - np.mean(img)
    img = img / np.std(img)     
    return img

def normalize_pct(img, min_pct, max_pct):
    pMin = np.percentile(img, min_pct); # print(pMin)
    pMax = np.percentile(img, max_pct); # print(pMax)
    if pMax == 0:
        pMax = np.max(img) # Debug
    np.clip(img, pMin, pMax, out=img)
    img -= pMin
    img /= (pMax - pMin)   
    return img

def preprocess_image(img):
    img = gaussian(img, sigma=2, preserve_range=True)
    img = normalize_gcn(img)
    img = normalize_pct(img, 0.01, 99.99)  
    return img

def preprocess_mask(msk, gamma=1.0):
    labels = np.unique(msk)[1:]
    edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
    for l, lab in enumerate(labels):
        tmp = msk == lab
        tmp = distance_transform_edt(tmp)
        if gamma != 1.0:
            tmp = adjust_gamma(tmp, gamma=gamma, gain=1)
        tmp = normalize_pct(tmp, 0.01, 99.99)
        edm[l,...] = tmp
    edm = np.max(edm, axis=0).astype("float32")  
    return edm

#%% Functions (predict) -------------------------------------------------------

def predict(img, size, overlap, model_name):
    
    # Define model
    model = sm.Unet(
        'resnet18', # ResNet 18, 34, 50, 101 or 152
        input_shape=(None, None, 1), 
        classes=1, 
        activation='sigmoid', 
        encoder_weights=None,
        )
    
    # Extract patches
    patches = extract_patches(img, size, overlap)
    patches = np.stack(patches)
    
    # Load weights & predict (cores)
    model_path = Path(Path.cwd(), model_name.replace("weights_", "weights_cores_"))
    model.load_weights(model_path)
    cProbs = model.predict(patches).squeeze()
    cProbs = merge_patches(cProbs, img.shape, overlap)
    
    # Load weights & predict (shell)
    model_path = Path(Path.cwd(), model_name.replace("weights_", "weights_shell_"))
    model.load_weights(model_path)
    sProbs = model.predict(patches).squeeze()
    sProbs = merge_patches(sProbs, img.shape, overlap)
    
    return sProbs, cProbs

#%% Functions (objects) -------------------------------------------------------

def label_objects(probs, thresh1=0.5, thresh2=0.2, rf=1):
    
    if rf != 1: 
        probs = rescale(probs, 0.5)
    
    probs = gaussian(probs, sigma=2, preserve_range=True)
    markers = label(probs > thresh1)
    markers = remove_small_objects(markers, min_size=256 * rf) # Parameter
    labels = watershed(
        -probs,
        markers=markers,
        mask=probs > thresh2,
        compactness=1,
        watershed_line=True,
        )
    labels = clear_border(labels)
    
    return labels


def measure_objects(sLabels, cLabels, rf=1):
   
    def find_label(data, label):
        for i, d in enumerate(data):
            if "sLabel" in d.keys():
                if d["sLabel"] == label:
                    return i
            if "cLabel" in d.keys():
                if d["cLabel"] == label:
                    return i
                
    sData = []
    for sProp in regionprops(sLabels):
        
        # Extract shell data
        sLabel = sProp.label
        sArea = int(sProp.area)
        sVolum = (4 / 3) * np.pi * np.sqrt(sArea / np.pi) ** 3
        sPerim = sProp.perimeter
        sFeret = sProp.feret_diameter_max
        sMajor = sProp.axis_major_length
        sMinor = sProp.axis_minor_length
        sSolid = sProp.solidity
        sRound = 4 * sArea / (np.pi * sMajor ** 2)
        sCircl = 4 * np.pi * sArea / sPerim ** 2
        sY = int(sProp.centroid[0])
        sX = int(sProp.centroid[1])

        # Extract cores data
        sIdx = (sProp.coords[:, 0], sProp.coords[:, 1])
        cVal = cLabels[sIdx]
        cVal = cVal[cVal != 0]
        cLabel = list(np.unique(cVal))

        # Append
        sData.append({
            
            "sLabel" : sLabel,
            "sArea"  : sArea / rf ** 2,
            "sVolum" : sVolum / rf ** 3,
            "sPerim" : sPerim / rf,
            "sFeret" : sFeret/ rf,
            "sMajor" : sMajor/ rf,
            "sMinor" : sMinor/ rf,
            "sSolid" : sSolid,
            "sRound" : sRound,
            "sCircl" : sCircl,
            "sY" : sY / rf, "sX" : sX / rf,
            
            "sCore"    : len(cLabel),
            "s_cLabel" : cLabel,
            
            })
        
    cData = []
    for cProp in regionprops(cLabels, intensity_image=sLabels):
        
        # Extract shell data
        sLabel = int(cProp.intensity_max)
        sIdx = find_label(sData, sLabel)
        sArea = sData[sIdx]["sArea"] * rf ** 2
        sY = sData[sIdx]["sY"] * rf
        sX = sData[sIdx]["sX"] * rf
        
        # Extract cores data
        cLabel = cProp.label
        cArea = int(cProp.area)
        cVolum = (4 / 3) * np.pi * np.sqrt(cArea / np.pi) ** 3
        cPerim = cProp.perimeter
        cFeret = cProp.feret_diameter_max
        cMajor = cProp.axis_major_length
        cMinor = cProp.axis_minor_length
        cSolid = cProp.solidity
        cRound = 4 * cArea / (np.pi * cMajor ** 2)
        cCircl = 4 * np.pi * cArea / cPerim ** 2
        cY = int(cProp.centroid[0])
        cX = int(cProp.centroid[1])
        csRatio = cArea / sArea
        csDist = np.sqrt((cX - sX)**2 + (cY - sY)**2)

        # Append
        cData.append({
            
            "cLabel"  : cLabel,
            "cArea"   : cArea / rf ** 2,
            "cVolum"  : cVolum / rf ** 3,
            "cPerim"  : cPerim / rf,
            "cFeret"  : cFeret / rf,
            "cMajor"  : cMajor / rf,
            "cMinor"  : cMinor / rf,
            "cSolid"  : cSolid,
            "cRound"  : cRound,
            "cCircl"  : cCircl,
            "cY" : cY / rf, "cX" : cX / rf,
            "csRatio" : csRatio,
            "csDist"  : csDist / rf,

            "c_sLabel" : sLabel,
            
            })
        
    # Update sData
    for i, data in enumerate(sData):
        s_cArea, s_cVolum, s_cPerim = [], [], [],
        s_cFeret, s_cMajor, s_cMinor = [], [], [],
        s_cSolid, s_cRound, s_cCircl = [], [], [],
        s_csRatio, s_csDist = [], []
        for cLabel in data["s_cLabel"]:
            cIdx = find_label(cData, cLabel)
            s_cArea.append(cData[cIdx]["cArea"])
            s_cVolum.append(cData[cIdx]["cVolum"])
            s_cPerim.append(cData[cIdx]["cPerim"])
            s_cFeret.append(cData[cIdx]["cFeret"])
            s_cMajor.append(cData[cIdx]["cMajor"])
            s_cMinor.append(cData[cIdx]["cMinor"])
            s_cSolid.append(cData[cIdx]["cSolid"])
            s_cRound.append(cData[cIdx]["cRound"])
            s_cCircl.append(cData[cIdx]["cCircl"])
            s_csRatio.append(cData[cIdx]["csRatio"])
            s_csDist.append(cData[cIdx]["csDist"])
        sData[i]["s_cArea"] = np.mean(s_cArea)
        sData[i]["s_cVolum"] = np.mean(s_cVolum)
        sData[i]["s_cPerim"] = np.mean(s_cPerim)
        sData[i]["s_cFeret"] = np.mean(s_cFeret)
        sData[i]["s_cMajor"] = np.mean(s_cMajor)
        sData[i]["s_cMinor"] = np.mean(s_cMinor)
        sData[i]["s_cSolid"] = np.mean(s_cSolid)
        sData[i]["s_cRound"] = np.mean(s_cRound)
        sData[i]["s_cCircl"] = np.mean(s_cCircl)
        sData[i]["s_csRatio"] = np.mean(s_csRatio)
        sData[i]["s_csDist"] = np.mean(s_csDist)   
        
    return sData, cData

#%% Functions (display) -------------------------------------------------------

def display(img, sLabels, cLabels, sData, cData):
    
    sText = np.zeros_like(cLabels)
    for data in sData:
        
        # Extract data
        sLabel = data["sLabel"]
        sY = int(data["sY"]) 
        sX = int(data["sX"])
        
        # Draw object texts
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            sText, f"{sLabel:03d}", 
            (sX - 30, sY - 5), # depend on resolution !!!
            font, 1, 1, 2, cv2.LINE_AA
            ) 
        
    cText = np.zeros_like(cLabels)
    for data in cData:
        
        # Extract data
        cLabel = data["cLabel"]
        cY = int(data["cY"]) 
        cX = int(data["cX"])
        
        # Draw object texts
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            cText, f"{cLabel:03d}", 
            (cX - 30, cY + 25), # depend on resolution !!!
            font, 1, 1, 2, cv2.LINE_AA
            ) 
        
    sMask = sLabels > 0
    cMask = cLabels > 0
    sOutlines = sMask ^ binary_erosion(sMask, footprint=disk(3))
    cOutlines = cMask ^ binary_erosion(cMask, footprint=disk(3))
    sDisplay = (sOutlines * 192 + sText * 255)
    cDisplay = (cOutlines * 192 + cText * 255)
    sDisplay = gaussian(sDisplay, sigma=1, preserve_range=True)
    cDisplay = gaussian(cDisplay, sigma=1, preserve_range=True)
    
    # RGB display
    R = img * 128 + sDisplay
    G = img * 128 + sDisplay + cDisplay
    B = img * 128 + cDisplay
    R[R > 255] = 255
    G[G > 255] = 255
    B[B > 255] = 255
    rgbDisplay = np.stack((R, G, B), axis=-1).astype("uint8")
    
    return sDisplay, cDisplay, rgbDisplay

#%% Functions (process) -------------------------------------------------------

def process(path, size, overlap, model_name, rf=1, save=True):
    
    print(path.name)
    
    # Open & preprocess image
    img = io.imread(path).astype("float32")
    img = np.mean(img, axis=2) # RGB to float
    img = preprocess_image(img)   
    
    # Predict (shell & cores)
    sProbs, cProbs = predict(img, size, overlap, model_name)
    
    # Label objects
    t0 = time.time()
    print(" - label_objects : ", end='')
    
    sLabels = label_objects(sProbs, thresh1=0.50, thresh2=0.2, rf=rf) # Parameters
    cLabels = label_objects(cProbs, thresh1=0.85, thresh2=0.2, rf=rf) # Parameters
    cLabels[sLabels == 0] = 0
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
    
    # Measure objects
    t0 = time.time()
    print(" - measure_objects : ", end='')
    
    sData, cData = measure_objects(sLabels, cLabels, rf=rf)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s")
  
    # Rescale data 
    if rf != 1:
        sLabels = rescale(sLabels, 1 / rf, order=0)
        cLabels = rescale(cLabels, 1 / rf, order=0)
        
    # Measure objects
    t0 = time.time()
    print(" - display : ", end='')
    
    sDisplay, cDisplay, rgbDisplay = display(img, sLabels, cLabels, sData, cData)
    
    t1 = time.time()
    print(f"{(t1-t0):<5.2f}s \n")
    
    # Dataframes
    sData_df, cData_df = pd.DataFrame(sData), pd.DataFrame(cData)
    
    # Save 
    if save:
        
        # Paths
        sData_df_path = path.with_name(path.stem + "_sData.csv")
        cData_df_path = path.with_name(path.stem + "_cData.csv")
        rgbDisplay_path = path.with_name(path.stem + "_display.png")
        
        # Dataframes
        sData_df.to_csv(sData_df_path, index=False, float_format='%.3f')
        cData_df.to_csv(cData_df_path, index=False, float_format='%.3f')
        
        # Display
        io.imsave(rgbDisplay_path, rgbDisplay)
        
    # Outputs
    outputs = {
        "img"        : img,
        "sProbs"     : sProbs,
        "cProbs"     : cProbs,
        "sLabels"    : sLabels,
        "cLabels"    : cLabels,
        "sData"      : sData,
        "cData"      : cData,
        "sData_df"   : sData_df,
        "cData_df"   : cData_df,
        "sDisplay"   : sDisplay,
        "cDisplay"   : cDisplay,
        "rgbDisplay" : rgbDisplay,
        }
        
    return outputs

#%% Functions (analyse) -------------------------------------------------------

def get_paths(root_path, tags_in, tags_out):
    paths = []
    for path in root_path.glob("**/*.jpg"):
        if tags_in:
            check_tags_in = all(tag in str(path) for tag in tags_in)
        else:
            check_tags_in = True
        if tags_out:
            check_tags_out = not any(tag in str(path) for tag in tags_out)
        else:
            check_tags_out = True
        if check_tags_in and check_tags_out:
            paths.append(path)
    return paths

def merge_df(paths):
    for i, path in enumerate(paths):
        sData_df = pd.read_csv(path.with_name(path.stem + "_sData.csv"))
        sData_df.insert(0, 'name', path.stem)
        cData_df = pd.read_csv(path.with_name(path.stem + "_cData.csv"))
        cData_df.insert(0, 'name', path.stem)
        if i == 0:
            sData_df_merged = sData_df
            cData_df_merged = cData_df
        else:
            sData_df_merged = pd.concat([sData_df_merged, sData_df], axis=0)   
            cData_df_merged = pd.concat([cData_df_merged, cData_df], axis=0)
    return sData_df_merged, cData_df_merged