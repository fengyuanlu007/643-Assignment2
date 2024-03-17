'''
This module contains functions for plotting the decision boundaries of a k-NN classifier.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn import neighbors

def plot_fruit_knn(x_data, y_data, n_neighbors, weights): # pylint: disable=R0914
    '''
    This function takes in the feature matrix, target vector, number of neighbors,
    and weights and plots the decision boundaries of a k-NN classifier.
    '''
    x_mat = x_data[['height', 'width']].to_numpy()
    y_mat = y_data.to_numpy()

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(x_mat, y_mat)
    # Plot the decision boundary by assigning a color in the color map to each mesh point.
    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50
    x_min, x_max = x_mat[:, 0].min() - 1, x_mat[:, 0].max() + 1
    y_min, y_max = x_mat[:, 1].min() - 1, x_mat[:, 1].max() + 1
    mesh_xx, mesh_yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step_size),
        np.arange(y_min, y_max, mesh_step_size)
        )
    mesh_z = clf.predict(np.c_[mesh_xx.ravel(), mesh_yy.ravel()])
    # Put the result into a color plot
    mesh_z = mesh_z.reshape(mesh_xx.shape)
    plt.pcolormesh(mesh_xx, mesh_yy, mesh_z, cmap=cmap_light, shading='auto')
    # Plot training points
    plt.scatter(x_mat[:, 0], x_mat[:, 1], s=plot_symbol_size,
                c='y', cmap=cmap_bold, edgecolor='black')
    plt.xlim(mesh_xx.min(), mesh_xx.max())
    plt.ylim(mesh_yy.min(), mesh_yy.max())

    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])
    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title("4-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
