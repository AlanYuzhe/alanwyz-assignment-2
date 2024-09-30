from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from kmeans import KMeans  # Assuming KMeans is defined in a separate file
import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
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

    data_points = X.tolist()

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    return jsonify({
        'data_points': data_points,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    })

def initialize_kmeans(k, method='Random', centers=None):
    global kmeans, current_step

    if method == 'KMeans++':
        kmeans = KMeans(X, k, method='KMeans++')
    elif method == 'Random':
        kmeans = KMeans(X, k)
    elif method == 'Farthest':
        kmeans = KMeans(X, k, method='Farthest')
    elif method == 'Manual':
        if centers is not None:
            kmeans = KMeans(X, k, initial_centers=centers)
        else:
            raise ValueError("Manual initialization requires centers to be provided.")
    else:
        raise ValueError(f"Unknown initialization method: {method}")

    current_step = 0

@app.route('/manual-kmeans', methods=['POST'])
def manual_kmeans():
    data = request.get_json()
    k = int(data.get('k', 4))
    centroids = data.get('centroids', [])

    centers = np.array([[point['x'], point['y']] for point in centroids])

    global kmeans
    initialize_kmeans(k, 'Manual', centers=centers)

    return jsonify({"message": "Centroids received, ready to run KMeans."}), 200

@app.route('/step-kmeans', methods=['POST'])
def step_kmeans():
    data = request.get_json()
    k = int(data.get('k', 4))
    init_method = data.get('initMethod', 'Random')

    global kmeans
    if kmeans is None:
        centers = None
        if init_method == 'Manual':
            centroids = data.get('centroids', None)
            if centroids is None or len(centroids) != k:
                return jsonify({"error": "Manual initialization requires centroids."}), 400
            centers = np.array([[point['x'], point['y']] for point in centroids])
        initialize_kmeans(k, init_method, centers=centers)

    step_success = kmeans.step()

    TEMPFILE = 'temp_step.png'
    if step_success:
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=kmeans.assignment)
        ax.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], c='red', marker='X', s=100)
        ax.set_title('KMeans Clustering Step')
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
        centers = None
        if init_method == 'Manual':
            centroids = data.get('centroids', None)
            if centroids is None or len(centroids) != k:
                return jsonify({"error": "Manual initialization requires centroids."}), 400
            centers = np.array([[point['x'], point['y']] for point in centroids])
        initialize_kmeans(k, init_method, centers=centers)

    kmeans.lloyds()  # Run the full KMeans algorithm

    TEMPFILE = 'temp_converge.png'
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=kmeans.assignment)
    ax.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], c='red', marker='X', s=100)
    ax.set_title('KMeans Clustering Converged')
    fig.savefig(TEMPFILE)
    plt.close()
    return send_file(TEMPFILE, mimetype='image/png')

# Route to reset KMeans and start over
@app.route('/reset-kmeans', methods=['POST'])
def reset_kmeans():
    data = request.get_json()
    k = int(data.get('k', 4))  # Get the value of k from the frontend
    init_method = data.get('initMethod', 'Random')

    global kmeans
    kmeans = None  # Reset the KMeans instance

    FILE_PATH = 'reset_dataset_image.png'
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c='blue')
    ax.set_title('KMeans Clustering Data')
    fig.savefig(FILE_PATH)
    plt.close()
    return send_file(FILE_PATH, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)