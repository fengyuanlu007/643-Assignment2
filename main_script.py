'''
This python script contains functions for
loading the data, cleaning the data,
training a model, making predictions,
and visualizing the KNN classifier and.
'''
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from data_loads import load_txt
from fruit_name_lookup import fruit_name_lookup
from plot_knn import plot_fruit_knn

def data_loads(file_path):
    '''
    This function loads the data from a text file
    '''
    fruits = load_txt(file_path)
    return fruits

def clean_data(fruits):
    '''
    This function cleans the data by removing the fruit subtype column
    '''
    fruits_cleaned = fruits.drop('fruit_subtype', axis=1)
    return fruits_cleaned

def pipeline():
    '''
    This function returns a pipeline object that can be used to fit and predict on the data
    '''
    numeric_features = ['mass', 'width', 'height', 'color_score']
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    pipeline_1 = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier())])
    return pipeline_1

def train(fruits_cleaned):
    '''
    This function trains a model on the cleaned data and returns the trained model
    '''
    feature_names_fruits = ['mass', 'width', 'height', 'color_score']
    x_fruits = fruits_cleaned[feature_names_fruits]
    y_fruits = fruits_cleaned['fruit_label']
    x_train, x_test, y_train, y_test = train_test_split(x_fruits, y_fruits, random_state=0)
    trained_model = pipeline()
    trained_model.fit(x_train, y_train)
    model_metrics = {
        "train_data": {
            "Accuracy of K-NN classifier": round(trained_model.score(x_train, y_train),3)
        },
        "test_data": {
            "Accuracy of K-NN classifier": round(trained_model.score(x_test, y_test),3)
        },
    }
    print(model_metrics)

    return trained_model

def predict(height, width, mass, color_score, fruits_cleaned, trained_model):
    '''
    This function takes in the height, width, mass,
    and color score of a fruit and returns the predicted fruit type
    '''
    input_data = pd.DataFrame(
        {'height': [height], 'width': [width], 'mass': [mass], 'color_score': [color_score]}
        )
    prediction = trained_model.predict(input_data)
    lookup_fruit_name = fruit_name_lookup(fruits_cleaned)
    print('Predicted fruit type for ', input_data, ' is ', lookup_fruit_name[prediction[0]])

def visualize_knn(fruits_cleaned, output_fig):
    '''
    This function visualizes the KNN classifier
    '''
    feature_names_fruits = ['mass', 'width', 'height', 'color_score']
    x_fruits = fruits_cleaned[feature_names_fruits]
    y_fruits = fruits_cleaned['fruit_label']
    x_train, _, y_train, _ = train_test_split(x_fruits, y_fruits, random_state=0)
    plot_fruit_knn(x_train, y_train, 5, 'uniform')
    plt.savefig(output_fig)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path_input', help='Path to the fruit_data_with_colors.txt file')
    parser.add_argument('height', help='height of the fruit')
    parser.add_argument('width', help='width of the fruit')
    parser.add_argument('mass', help='mass of the fruit')
    parser.add_argument('color_score', help='color_score of the fruit')
    parser.add_argument('output_png', help='Path to save the visualization image')
    args = parser.parse_args()
    fruits_load = data_loads(args.file_path_input)
    cleaned_fruit_df = clean_data(fruits_load)
    trained_fruit_model = train(cleaned_fruit_df)
    predict(float(args.height),
            float(args.width),
            float(args.mass),
            float(args.color_score),
            cleaned_fruit_df,
            trained_fruit_model)
    visualize_knn(cleaned_fruit_df, args.output_png)
