#%% Imports -------------------------------------------------------------------

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Functions
from functions import get_paths, merge_df

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path(Path.cwd(), 'data')

# Path (selection)
tags_in = ["RR"]
tags_out = []
tags_str = f"{tags_in}in_{tags_out}out"

# Plot (selection)
plot_type = "scatter"
plot_data = ["sArea", "sRound", "sCore"] 
    
#%% Execute -------------------------------------------------------------------

paths = get_paths(data_path, tags_in, tags_out)
sData_df_merged, cData_df_merged = merge_df(paths)

data = []
for name in plot_data:
    if name is not None:
        if name[0] == "s": data.append(np.array(sData_df_merged[name]))
        if name[0] == "c": data.append(np.array(cData_df_merged[name]))

if plot_type == "scatter":
    if len(data) == 1:
        print("scatter plot require 2 or 3 data_names")
    if len(data) == 2:
        plt.scatter(data[0], data[1], s=10)
        plt.title(f"{plot_data[0]} vs. {plot_data[1]}\n{tags_str}")
        plt.xlabel(f"{plot_data[0]}")
        plt.ylabel(f"{plot_data[1]}")
    if len(data) == 3:
        plt.scatter(data[0], data[1], c=data[2], s=10)
        plt.title(f"{plot_data[0]} vs. {plot_data[1]} vs.{plot_data[2]}\n{tags_str}")
        plt.xlabel(f"{plot_data[0]}")
        plt.ylabel(f"{plot_data[1]}")
        plt.colorbar(label=f"{plot_data[2]}")
        
# data = []
# for name in data_names:
#     if name is not None:
#         if name[0] == "s": data.append(np.array(sData_df_merged[name]))
#         if name[0] == "c": data.append(np.array(cData_df_merged[name]))
# if plot_type == "scatter":
#     if len(data) == 1:
#         print("scatter plot require 2 or 3 data_names")
#     if len(data) == 2:
#         plt.scatter(data[0], data[1])
#         plt.title(f"{data_names[0]} vs. {data_names[1]}\n{tags}")
#         plt.xlabel(f"{data_names[0]}")
#         plt.ylabel(f"{data_names[1]}")
#     if len(data) == 3:
#         plt.scatter(data[0], data[1], c=data[2])
#         plt.title(f"{data_names[0]} vs. {data_names[1]} vs.{data_names[2]}\n{tags}")
#         plt.xlabel(f"{data_names[0]}")
#         plt.ylabel(f"{data_names[1]}")
#         plt.colorbar(label=f"{data_names[2]}")
# if plot_type == "hist":
#     plt.figure(figsize=(10, 10 * len(data)))
#     for i, dat in enumerate(data):
#         plt.subplot(len(paths), 1, i + 1)
#         plt.hist(data[i], bins=100)
#         plt.title(f"{data_names[i]}")
#         plt.ylabel("count")
#     plt.tight_layout(pad=2)

# if plot_type == "hist":
#     plt.hist(data1, bins=100)
#     plt.title(f"{plot_data1} {tags}")
# if plot_type == "scatter":
#     plt.scatter(data1, data2, c=data3)
#     plt.title(f"{plot_data1} vs. {plot_data2} {tags}")