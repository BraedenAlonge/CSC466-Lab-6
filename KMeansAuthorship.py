import sys
import json
import numpy as np
import csv
from sklearn.cluster import KMeans

def load_vectors(vectors_file):
    # Load the vector json file
    with open(vectors_file, 'r') as f:
        vectors = json.load(f)
    return vectors

def convert_to_dense(vectors):
    # Convert sparse tf-idf vecs to dense matrix
    # To do so, we will use Numpy. vecs are dictionaries

    # First, determine vocab size
    vocab_size = 0
    for vec in vectors.values():
        for key in vec:
            index = int(key)
            if index + 1 > vocab_size:
                vocab_size = index + 1

    dense_matrix = []
    filenames = []
    for filename, vector in vectors.items():
        # Create a row vector of zeros for the ENTIRE vocabulary
        row = np.zeros(vocab_size)
        for key, value in vector.items():
            index = int(key)
            row[index] = value
        dense_matrix.append(row)
        filenames.append(filename)
    dense_matrix = np.array(dense_matrix)

    return filenames, dense_matrix


def run_k_means(matrix, num_clusters, max_iterations=1000):
    # matrix: np.array. each row is a document's tf-idf vector
    # num_clusters: should be 50 for this dataset, but left to decide
    # max_iterations: self-explanatory

    # Initialize KMeans with the specified number of clusters and maximum iterations.
    # random_state is set for reproducibility.
    kmeans = KMeans(n_clusters=num_clusters, max_iter=max_iterations, random_state=42)
    # Fit
    kmeans.fit(matrix)
    # Return cluster labels for each doc.
    return kmeans.labels_

def save_clusters(filenames, labels, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cluster_label"])
        for filename, label in zip(filenames, labels):
            writer.writerow([filename, label])
    print(f"Saved clustering results to {output_file}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python KMeansAuthorship.py <vectors_json> <num_clusters> [max_iter] [output_file]")
        sys.exit(1)

    vectors_file = sys.argv[1]
    try:
        num_clusters = int(sys.argv[2])
    except ValueError:
        print("Error: num_clusters must be an integer.")
        sys.exit(1)
    if len(sys.argv) > 3:
        max_iter = int(sys.argv[3])
    else:
        max_iter = 1000
    if len(sys.argv) > 4:
        output_file = sys.argv[4]
    else:
        output_file = "./output/kmeans_clusters.csv"


    print("Loading vectors...")
    vectors = load_vectors(vectors_file)
    print("Converting sparse vectors to a dense matrix...")
    filenames, dense_matrix = convert_to_dense(vectors)

    print("Running KMeans clustering with {} clusters and max_iter {}...".format(num_clusters, max_iter))
    labels = run_k_means(dense_matrix, num_clusters, max_iter)

    print("Saving clustering results...")
    save_clusters(filenames, labels, output_file)

    print("KMeans clustering complete.")

if __name__ == "__main__":
    main()


# Test run ex.:  python .\KMeansAuthorship.py .\output\output_vectors.json 50 850