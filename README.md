# main_script.py README

## Introduction

This script, main_script.py, is designed to perform data analysis tasks on fruit dataset. It includes functionalities such as loading data, cleaning data, training a machine learning model, making predictions, and visualizing results.

## Setup Environment

### required pachages
To run the script, you'll need to set up the following environment:
Ensure you have the following Python packages installed:
    - numpy
    - pandas
    - seaborn
    - matplotlib
    - scikit-learn

To run the script, ensure you have the required packages installed. You can install them using pip and the `requirements.txt` file provided:

    - pip install -r requirements.txt

### Other Modules
In addition to the required packages listed in requirements.txt, the script may rely on other modules or libraries:

    - data_loads: Contains functions for loading data from external sources.
    - fruit_name_lookup: Provides a lookup dictionary for fruit names based on labels.
    - plot_knnplot_knn: Includes functions for visualizing the K-Nearest Neighbors classifier.

Make sure to install or have access to these modules before running the script.

### Data Files

The script expects a data file named fruit_data_with_colors.txt to be present in the same directory. This file contains the fruit dataset used for analysis. Make sure to download and place the data file in the correct location before running the script.

## Script Functionality
The script main_script.py provides the following functionality:

    - Data Loading: The script loads the fruit dataset from the fruit_data_with_colors.txt file.

    - Data Cleaning: It cleans the dataset by removing unnecessary columns and preprocessing the data.

    - Model Training: The script trains a machine learning model (K-Nearest Neighbors Classifier) using the cleaned data.

    - Prediction: It allows users to make predictions on new fruit data by - providing input parameters such as height, width, mass, and color score.

    - Visualization: The script includes functionality to visualize the K-Nearest Neighbors Classifier.

## Usage
To run the script, execute the following command in your terminal:

    - python main_script.py <file_path> <height> <width> <mass>             <color_score>   <output_file>

    - Replace <file_path> with the path to the fruit_data_with_colors.txt file, <height>, <width>, <mass>, and <color_score> with the respective parameters of the fruit you want to predict, and <output_file> with the path to save the visualization image.