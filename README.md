# A Cluster-based Temporal Attention Approach for Predicting Cyclone-induced Compound Flood Dynamics
## Introduction
This repository contains the source code for a deep learning (DL) model designed to predict cyclone-induced compound flood dynamics, including flood extent and inundation depth over time, in coastal regions. The model is applied to Galveston Bay, TX, a region highly susceptible to compound flooding from storm surges, river overflow, and heavy rainfall. It leverages a hybrid architecture combining Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal regression, enhanced by a novel cluster-based temporal attention mechanism. This approach captures spatiotemporal dependencies and hysteresis effects, replicating the behavior of physically-based models like Delft3D-FM with greater computational efficiency. 

The model integrates multimodal inputs such as atmospheric pressure, wind speed, precipitation, river discharge, digital elevation models (DEM), and water level data from observation stations. It was trained and validated on historical flood events including Hurricanes Ike (2008), Harvey (2017), Nicholas (2021), Beryl (2024), and torrential rainfall events in 2015 (Memorial Day) and 2016 (Tax Day). Key features include:

- **Cluster-based Temporal Attention**: Uses Voronoi tessellation to partition the spatial domain into clusters aligned with observation stations, modulating predictions with localized temporal dynamics.
- **Bayesian Optimization**: Employs Optuna to tune hyperparameters, ensuring optimal model performance.
- **Performance Metrics**: Achieves flood timing accuracy (±1 hour), critical success index (CSI) above 60%, RMSE below 0.10 m, and an error bias near 1, demonstrating its potential for flood preparation and response.

The source code is organized into scripts for generating Voronoi clusters and training/predicting flood dynamics, with detailed instructions provided below.

## Requirements
To run this project, you will need the following dependencies and tools: 

- Python 3.8 or higher 
- TensorFlow 2.x 
- Keras 
- NumPy 
- Pandas 
- Matplotlib 
- Geopandas 
- Rasterio 
- Scipy 
- Optuna (for hyperparameter tuning)
- Pyproj 
- Shapely
- Optional: NVIDIA GPU with CUDA support for faster computation

Install the required packages using pip:
```bash
pip install tensorflow keras numpy pandas matplotlib geopandas rasterio scipy optuna pyproj shapely
```
## Hardware Recommendations
- A GPU (e.g., NVIDIA A100) is strongly recommended for training due to the computational intensity of ConvLSTM layers and Bayesian optimization. 
- At least 16 GB of RAM for data preprocessing and model training.

## Installation 
Follow these steps to set up the environment and install dependencies:
```bash
git clone https://github.com/CoRAL-Lab-VT/FloodDepthDL.git
cd FloodDepthDL
```

## Source Code Description
The repository includes two primary Python scripts:

**1. voronoi_clusters.py script**
This script generates Voronoi clusters based on observation station coordinates and a floodmap boundary, saving them as a shapefile (reordered_polygons.shp). These clusters are used in the cluster-based attention mechanism.
### - Key Functions:
  - *generate_voronoi_clusters_and_empty_areas*: Creates Voronoi polygons from station coordinates, clipped to the floodmap boundary.
  - *reorder_polygons_by_station*: Reorders polygons to match station indices.
  - *combine_specified_polygons*: Optionally combines specified polygon pairs (commented out by default).
  - *save_polygons_as_shapefile*: Saves the polygons as a shapefile.
  - *plot_floodmap_with_voronoi_and_labels*: Visualizes the floodmap with labeled clusters.
### - Inputs:
  - Station coordinates from CSV files in *observation_points/*.
  - Floodmap boundary from *GBay_cells_polygon.shp*.
  - 




