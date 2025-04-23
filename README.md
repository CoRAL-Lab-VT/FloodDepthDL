# A Cluster-based Temporal Attention Approach for Predicting Cyclone-induced Compound Flood Dynamics
## Introduction
This repository contains the source code for a deep learning (DL) model designed to predict cyclone-induced compound flood dynamics, including flood extent and inundation depth over time, in coastal regions. The model is applied to Galveston Bay, TX, a region highly susceptible to compound flooding from storm surges, river overflow, and heavy rainfall. It leverages a hybrid architecture combining Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal regression, enhanced by a novel cluster-based temporal attention mechanism. This approach captures spatiotemporal dependencies and hysteresis effects, replicating the behavior of physically-based models like Delft3D-FM with greater computational efficiency. 

The model leverages multimodal inputs, including:
- **Spatial features**: Atmospheric pressure, precipitation and wind speed (https://cds.climate.copernicus.eu/#!/home); digital elevation model (DEM) (https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.dem:403, https://coast.noaa.gov/); and river discharge (https://dashboard.waterdata.usgs.gov/app/nwd/en/), provided as GeoTIFF images at 50-meter resolution.
- **Temporal features**: Hourly water level data from 21 observation stations .

It was trained and validated on historical flood events including Hurricanes Ike (2008), Harvey (2017), Nicholas (2021), Beryl (2024), and torrential rainfall events in 2015 (Memorial Day) and 2016 (Tax Day). Key features include:

- **Cluster-based Temporal Attention**: Uses Voronoi tessellation to partition the spatial domain into clusters aligned with observation stations, modulating predictions with localized temporal dynamics.
- **Bayesian Optimization**: Employs Optuna to tune hyperparameters, ensuring optimal model performance.
- **Performance Metrics**: Achieves flood timing accuracy (±1 hour), critical success index (CSI) above 60%, RMSE below 0.10 m, and an error bias near 1, demonstrating its potential for flood preparation and response.

The source code is organized into scripts for generating Voronoi clusters and training/predicting flood dynamics, with detailed instructions provided below.

## Requirements
To run this project, you will need the following dependencies and tools: 
 
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

Install the required packages using pip:
```bash
pip install tensorflow keras numpy pandas matplotlib geopandas rasterio scipy optuna pyproj shapely
```
## Hardware Recommendations
- A GPU (e.g., NVIDIA A100) is strongly recommended for training due to the computational intensity of ConvLSTM layers and Bayesian optimization. 
- At least 16 GB of RAM for data preprocessing and model training.

## Software Recommendations
Python: 3.8 or higher
Delft3D-FM: Version 2021.03 (optional, for generating ground truth data; available at https://oss.deltares.nl/web/delft3d/downloads)

## Installation and Setup
**Clone the Repository**

Follow these steps to set up the environment and install dependencies:
```bash
git clone https://github.com/CoRAL-Lab-VT/FloodDepthDL.git
cd FloodDepthDL
```

## Source Code Description
The repository includes two primary Python scripts:

**1. VoronoiTessellationPolygons.py script**
This script generates Voronoi clusters based on observation station coordinates and a floodmap boundary, saving them as a shapefile (reordered_polygons.shp). These clusters are used in the cluster-based attention mechanism.
- ### Key Functions:
  - *generate_voronoi_clusters_and_empty_areas*: Creates Voronoi polygons from station coordinates, clipped to the floodmap boundary.
  - *reorder_polygons_by_station*: Reorders polygons to match station indices.
  - *combine_specified_polygons*: Optionally combines specified polygon pairs (commented out by default).
  - *save_polygons_as_shapefile*: Saves the polygons as a shapefile.
  - *plot_floodmap_with_voronoi_and_labels*: Visualizes the floodmap with labeled clusters.
- ### Inputs:
  - Station coordinates from CSV files in *observation_points/*.
  - Study area floodmap boundary/shapefile e.g *GBay_cells_polygon.shp*.
- ### Outputs:
  - *reordered_polygons.shp*: Shapefile of Voronoi clusters.
  - *floodmap_with_voronoi_and_labels_reordered.png*: Visualization of clusters.

**2. ModelSimulation.py**
This script implements the DL model, including data preprocessing, model construction, Bayesian optimization, and training. It predicts flood depth maps based on spatial and temporal inputs.
- ### Key Components:
  - ### Data Loading and Preprocessing:
    - Loads TIFF images (e.g., atmospheric pressure, wind speed) and CSV water level data.
    - Normalizes data, handling NaN values with a custom mask.
    - Creates sequences for temporal modeling (6-hour intervals).
  - ### Custom Layers:
    - *StandardCBAM*: Convolutional Block Attention Module for spatial attention.
    - *CustomAttentionLayer*: Temporal attention with emphasis on critical timesteps.
    - *ClusterBasedApplication*: Applies cluster-specific attention to spatial features.
  - ### Model Architecture:
    - *Spatial Branch*: Three ConvLSTM2D layers with CBAM for feature extraction.
    - *Temporal Branch*: Two LSTM layers with attention for water level dynamics.
    - *Integration*: Modulates spatial output with cluster-based temporal context.
  - **Bayesian Optimization**: Uses Optuna to tune hyperparameters (e.g., filters, units, learning rate).
  - **Loss Function**: Custom masked MSE to ignore invalid (NaN) pixels.
  - ### Inputs:
    - Spatial data directories: *atm_pressure/, wind_speed/, precipitation/, river_discharge/, DEM/, water_depth/*.
    - Temporal data: *training_water_level/*.
    - Cluster shapefile: *reordered_polygons.shp*.
  - ### Outputs:
    - Model checkpoints and optimization results in *checkpoint_BO/*.
    - Normalized data parameters: *normalization_params.npy*.

## Usage
### Step 1: Generate Voronoi Clusters
Run the Voronoi cluster generation script:
```bash
python voronoi_clusters.py
```
- **Expected Output**:
  - reordered_polygons.shp in the root directory.
  
### Step 2: Train the Model
Run the flood depth prediction script with Bayesian optimization:
```bash
python ModelSimulation.py
```
- **Process**:
  - Loads and preprocesses spatial and temporal data.
  - Performs 100 trials of hyperparameter tuning using Optuna.
  - Trains the model with the best hyperparameters, saving results in checkpoint_BO/.
- **Customization**:
  - Adjust sequence_length (default: 6) in the script to change the temporal window.
  - Modify hyperparameter ranges in build_model_with_cbam_weighted for different tuning options.

### Step 3: Make Predictions
After training, use the trained model for predictions *ModelPrediction.py*:

1. Load the best model from *checkpoint_BO/*.
2. Prepare test data in the same format as training data.
3. Run inference and denormalize predictions using saved *normalization_params.npy*.

Example modification (add to the end of *ModelSimulation.py*):
```bash
# Load best model (assumes model saved as 'best_model.h5')
model = tf.keras.models.load_model('checkpoint_BO/best_model.h5', custom_objects={
    'StandardCBAM': StandardCBAM, 'CustomAttentionLayer': CustomAttentionLayer,
    'ClusterBasedApplication': ClusterBasedApplication, 'masked_mse': masked_mse,
    'TrueLoss': TrueLoss
})

# Load test data (example paths)
test_X_norm = X_train_norm  # Replace with actual test data
test_masks = nan_masks
test_water_level = water_level_data_sequences

# Predict
predictions_norm = model.predict([test_X_norm, test_masks, test_water_level])

# Denormalize
params = np.load('checkpoint_BO/normalization_params.npy', allow_pickle=True).item()
y_pred = (predictions_norm - 0.1) * (params['y_train_max'] - params['y_train_min']) / 0.9 + params['y_train_min']

# Save predictions as TIFF 
with rasterio.open('prediction.tif', 'w', driver='GTiff', height=y_pred.shape[1], width=y_pred.shape[2], count=1, dtype=y_pred.dtype, crs=crs, transform=transform) as dst:
    dst.write(y_pred[0], 1)
```

## Data Availability
All data used in this study are publicly available:

- **Legacy DEM**: NOAA’s National Geophysical Data Center
- **CUDEM**: NOAA’s Data Access Viewer
- **Land Cover Maps**: Multi-Resolution Land Characteristics Consortium
- **Water Level Data**: NOAA’s Tides & Currents
- **River Discharge**: USGS National Water Dashboard
- **Barotropic Tides**: TPXO 8.0 Global Inverse Tide Model
- **Rainfall Data**: Harris County Flood Warning System
- **ERA5 Reanalysis**: European Centre for Medium-Range Weather Forecasts

## Contact
For questions or issues, please open an issue on GitHub or contact:

- Samuel Daramola (samueldaramola@vt.edu)
- David F. Muñoz (davidmunozpauta@vt.edu)

## Contributing
Contributions are welcome and highly appreciated. You can contribute by:

- Reporting Bugs
- Suggesting Enhancements
- Sending Pull Requests

## References
Refer to these papers for a detailed explanation:

- Daramola, S., et al. (2025). A Cluster-based Temporal Attention Approach for Predicting Cyclone-induced Compound Flood Dynamics.
- Muñoz, D.F., et al. (2024). Quantifying cascading uncertainty in compound flood modeling with linked process-based and machine learning models. Hydrology and Earth System Sciences, 28, 2531–2553.

