# ParticleTracking

![Python Badge](https://img.shields.io/badge/Python-3.10-rgb(69%2C132%2C182)?logo=python&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![TensorFlow Badge](https://img.shields.io/badge/TensoFlow-2.10-rgb(255%2C115%2C0)?logo=TensorFlow&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![CUDA Badge](https://img.shields.io/badge/CUDA-11.2-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))
![cuDNN Badge](https://img.shields.io/badge/cuDNN-8.1-rgb(118%2C185%2C0)?logo=NVIDIA&logoColor=rgb(149%2C157%2C165)&labelColor=rgb(50%2C60%2C65))    
![Date Badge](https://img.shields.io/badge/Created-2024--04--10-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))
![License Badge](https://img.shields.io/badge/Licence-GNU%20General%20Public%20License%20v3.0-blue?labelColor=rgb(50%2C60%2C65)&color=rgb(149%2C157%2C165))    

## CapsSeg  
Microcapsule segmentation tool

## Index
- [Installation](#installation)
- [Content](#content)
- [Outputs](#outputs)
- [Comments](#comments)

## Installation

Pease select your operating system

<details> <summary>Windows</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Run the downloaded `.exe` file  
    - Select "Add Miniforge3 to PATH environment variable"  

### Step 3: Setup Conda 
- Open the newly installed Miniforge Prompt  
- Move to the downloaded GitHub repository
- Run one of the following command:  
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf_gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf_nogpu.yml
```  
- Activate Conda environment:
```bash
conda activate CapsSeg
```
Your prompt should now start with `(CapsSeg)` instead of `(base)`

</details> 

<details> <summary>MacOS</summary>  

### Step 1: Download this GitHub Repository 
- Click on the green `<> Code` button and download `ZIP` 
- Unzip the downloaded file to a desired location

### Step 2: Install Miniforge (Minimal Conda installer)
- Download and install [Miniforge](https://github.com/conda-forge/miniforge) for your operating system   
- Open your terminal
- Move to the directory containing the Miniforge installer
- Run one of the following command:  
```bash
# Intel-Series
bash Miniforge3-MacOSX-x86_64.sh
# M-Series
bash Miniforge3-MacOSX-arm64.sh
```   

### Step 3: Setup Conda 
- Re-open your terminal 
- Move to the downloaded GitHub repository
- Run one of the following command: 
```bash
# TensorFlow with GPU support
mamba env create -f environment_tf_gpu.yml
# TensorFlow with no GPU support 
mamba env create -f environment_tf_nogpu.yml
```  
- Activate Conda environment:  
```bash
conda activate CapsSeg
```
Your prompt should now start with `(CapsSeg)` instead of `(base)`

</details>

## Content

### main.py  
Processes all JPG files contained in the user-provided `data_path` folder and subfolders. Each iteration produces two CSV files (`cData.csv` and `sData.csv`) and a display image (`display.png`) saved at the location of the processed image. See [Outputs](#Outputs) section for more details.

### display.py  
Display segmentation results (outlines, labels, probabilities...) in Napari for the user-provided `data_path` and `image_name`.

### analyse.py
Compile data from different CSV files, including `tags_in` and excluding `tags_out` containing file name.

### Others
- **functions.py** - contains all required functions
- **environment-gpu.yml** - dependencies with GPU support (NVIDIA GPU required)
- **environment-nogpu.yml** - dependencies with no GPU support
- **model_weights_cores.h5** - model weights for core segmentation
- **model_weights_shells.h5** - model weights for shell segmentation

## Outputs

### display.png  
Image showing results of segmentation with detected **shells** in yellow and **cores** in cyan.

<img src='utils/example_display.png' alt="example_display">

### cData.csv - inner cores data

```bash
# Core data
    - cLabel # core ID
    - cArea # area
    - cVolum # volume (assuming spherical shape)  
    - cPerim # perimeter
    - cFeret # max feret diameter
    - cMajor # fitted ellipse major axis length 
    - cMinor # fitted ellipse minor axis length
    - cSolid # solidity (object area / convex hull area)
    - cRound # roundness  
    - cCircl # circularity  
    - cY # centroid y position
    - cX # centroid x position
    - csRatio # core area / shell area
    - csDist # core to shell centroid distance

# Associated shell data
    - c_sLabel # associated shell ID
```

### sData.csv - outer shell data
```bash
# Shell data
    - sLabel # shell ID
    - sArea # area
    - sVolum # volume (assuming spherical shape)
    - sPerim # perimeter
    - sFeret # max feret diameter
    - sMajor # fitted ellipse major axis length 
    - sMinor # fitted ellipse minor axis length
    - sSolid # solidity (object area / convex hull area)
    - sRound # roundness
    - sCircl # circularity
    - sY # centroid y position
    - sX # centroid x position

# Associated core(s) data
    - sCore # number of associated core(s)
    - s_cLabel # core(s) ID
    - s_cArea # core(s) avg. area
    - s_cVolum # core(s) avg. volume
    - s_cPerim # core(s) avg. perimeter
    - s_cFeret # core(s) avg. max feret diameter
    - s_cMajor # core(s) avg. fitted ellipse major axis length 
    - s_cMinor # core(s) avg. fitted ellipse minor axis length
    - s_cSolid # core(s) avg. solidity
    - s_cRound # core(s) avg. roundness
    - s_cCircl # core(s) avg. circularity
    - s_csRatio # core(s) avg. core area / shell area
    - s_csDist # core(s) avg. core to shell centroid distance
```

```bash
# Volume = (4 / 3) * np.pi * np.sqrt(cArea / np.pi) ** 3
# Roundness = 4 * cArea / (np.pi * cMajor ** 2)
# Circularity = 4 * np.pi * cArea / cPerim ** 2
```

## Comments
```bash
# Error message
lxml.html.clean module is now a separate project lxml_html_clean. 
Install lxml[html_clean] or lxml_html_clean directly. 
# Fix
pip install --upgrade lxml_html_clean
```
