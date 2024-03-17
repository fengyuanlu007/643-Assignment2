import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_loads import load_txt
from fruit_name_lookup import fruit_name_lookup

def data_loads(file_path):
    # Load the data
    fruits = load_txt(file_path)
    return fruits

def clean_data(fruits):
    # Load the data
    fruits = data_loads(file_path)
    # Remove the fruit subtype column
    fruits_cleaned = fruits.drop('fruit_subtype', axis=1)
    return fruits_cleaned

def pipeline():
    '''
    This function returns a pipeline object that can be used to fit and predict on the data
    '''
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.compose import ColumnTransformer
    numeric_features = ['mass', 'width', 'height', 'color_score']
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier())])
    return pipeline

def train(fruits_cleaned):
    '''
    This function trains a model on the cleaned data and returns the trained model
    '''
    # Load and clean the data
    fruits = data_loads(file_path)
    fruits_cleaned = clean_data(fruits)
    # Define the feature matrix and target vector
    feature_names_fruits = ['mass', 'width', 'height', 'color_score']
    X_fruits = fruits_cleaned[feature_names_fruits]
    y_fruits = fruits_cleaned['fruit_label']
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)
    trained_model = pipeline()
    trained_model.fit(X_train, y_train)
     # Store model metrics in a dictionary
    model_metrics = {
        "train_data": {
            "Accuracy of K-NN classifier": round(trained_model.score(X_train, y_train),3)
        },
        "test_data": {
            "Accuracy of K-NN classifier": round(trained_model.score(X_test, y_test),3)
        },
    }
    print(model_metrics)

    return trained_model

def predict(h, w, m, color_score,fruits_cleaned,trained_model):
    '''
    This function takes in the height, width, mass, and color score of a fruit and returns the predicted fruit type
    '''
    # Create a dataframe from the input data
    input_data = pd.DataFrame({'height': [h], 'width': [w], 'mass': [m], 'color_score': [color_score]})
    # Make a prediction
    prediction = trained_model.predict(input_data)
    # create fruit and name lookup dictionary
    lookup_fruit_name = fruit_name_lookup(fruits_cleaned)
    # Return the predicted fruit type
    return print('Predicted fruit type for ', input_data, ' is ',lookup_fruit_name[prediction[0]])

def visualize_knn(fruits_cleaned, output_fig):
    '''
    This function visualizes the KNN classifier
    '''
    from plot_knn import plot_fruit_knn
    # Define the feature matrix and target vector
    feature_names_fruits = ['mass', 'width', 'height', 'color_score']
    X_fruits = fruits_cleaned[feature_names_fruits]
    y_fruits = fruits_cleaned['fruit_label']
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_fruits, y_fruits, random_state=0)
    # Visualize the KNN classifier
    plot_fruit_knn(X_train, y_train, 5, 'uniform')
    plt.savefig(output_fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', help='Path to the fruit_data_with_colors.txt file')
    parser.add_argument('height',  help='height of the fruit')
    parser.add_argument('width', help='width of the fruit')
    parser.add_argument('mass', help='mass of the fruit')
    parser.add_argument('color_score', help='color_score of the fruit')
    parser.add_argument('output_file', help='Path to save the visualization image')
    args = parser.parse_args()

    file_path = args.file_path
    output_file = args.output_file
    fruits = data_loads(file_path)
    fruits_cleaned = clean_data(fruits)
    trained_model = train(fruits_cleaned)
    predict(float(args.height), float(args.width), float(args.mass), float(args.color_score),fruits_cleaned,trained_model)
    visualize_knn(fruits_cleaned, output_file)
