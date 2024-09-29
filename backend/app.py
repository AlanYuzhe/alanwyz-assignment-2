from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from kmeans import KMeans  # Assuming KMeans is defined in a separate file
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

import matplotlib
matplotlib.use('Agg')  

X = None  # Global variable to store the dataset
kmeans = None  # KMeans instance
current_step = 0  # Current step in KMeans algorithm

# Function to generate a new dataset
def generate_new_dataset():
    global X
    centers = [[-2, -3], [2, 2], [-3, 2], [2, -4]]
    X, _ = datasets.make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=None)
    return X

@app.route('/generate-dataset', methods=['POST'])
def generate_dataset():
    global X
    X = generate_new_dataset()

    FILE_PATH = 'dataset_image.png'
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c='blue')
    ax.set_title('Kmeans Clustering Data')
    fig.savefig(FILE_PATH)
    plt.close()

    return send_file(FILE_PATH, mimetype='image/png')

def initialize_kmeans(k, method='Random'):
    global kmeans, current_step

    if method == 'KMeans++':
        centers = []
        first_center = X[np.random.choice(X.shape[0])]
        centers.append(first_center)
        for _ in range(1, k):
            distances = np.min([np.sum((X - center) ** 2, axis=1) for center in centers], axis=0)
            next_center = X[np.random.choice(X.shape[0], p=distances / distances.sum())]
            centers.append(next_center)
        centers = np.array(centers)
        kmeans = KMeans(X, k, initial_centers=centers)
    elif method == 'Random':
        # Placeholder: Future logic for random initialization
        kmeans = KMeans(X, k)
    elif method == 'Farthest':
        # Placeholder: Future logic for Farthest First initialization
        kmeans = KMeans(X, k)
    elif method == 'Manual':
        # Placeholder: Future logic for manually setting initial centers
        centers = np.array([[0, 0], [2, 2], [-3, 2], [2, -4]])  # Example of manual centers
        kmeans = KMeans(X, k, initial_centers=centers)
    else:
        raise ValueError(f"Unknown method: {method}")

    current_step = 0  

@app.route('/step-kmeans', methods=['POST'])
def step_kmeans():
    data = request.get_json()
    k = int(data.get('k', 4))
    init_method = data.get('initMethod', 'Random')  # Get initialization method

    global kmeans
    if kmeans is None:
        initialize_kmeans(k, init_method)  # Initialize KMeans

    step_success = kmeans.step()  # Execute one step

    TEMPFILE = 'temp_step.png'
    if step_success:
        # Continue generating image
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=kmeans.assignment)
        ax.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], c='r')
        ax.set_title('Kmeans Clustering Data')
        fig.savefig(TEMPFILE)
        plt.close()
        return send_file(TEMPFILE, mimetype='image/png')
    else:
        return jsonify({"converged": True}), 200  # Return convergence status

# Route to run the complete KMeans algorithm until convergence
@app.route('/run-kmeans', methods=['POST'])
def run_kmeans():
    data = request.get_json()
    k = int(data.get('k', 4))  
    init_method = data.get('initMethod', 'Random')

    global kmeans
    if kmeans is None:
        initialize_kmeans(k, init_method)
    
    kmeans.lloyds()  # Run the full KMeans algorithm

    TEMPFILE = 'temp_converge.png'
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=kmeans.assignment)
    ax.scatter(kmeans.compute_centers()[:, 0], kmeans.compute_centers()[:, 1], c='r')  # Plot centroids
    ax.set_title('Kmeans Clustering Data')
    fig.savefig(TEMPFILE)
    plt.close()
    return send_file(TEMPFILE, mimetype='image/png')

# Route to reset KMeans and start over
@app.route('/reset-kmeans', methods=['POST'])
def reset_kmeans():
    data = request.get_json()
    k = int(data.get('k', 4))  # Get the value of k from the frontend
    init_method = data.get('initMethod', 'Random')

    initialize_kmeans(k, init_method)

    FILE_PATH = 'reset_dataset_image.png'
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c='blue')
    ax.set_title('Kmeans Clustering Data')
    fig.savefig(FILE_PATH)
    plt.close()
    return send_file(FILE_PATH, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)