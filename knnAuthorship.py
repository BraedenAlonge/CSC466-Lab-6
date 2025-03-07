import sys
import json
import csv
import math

def load_vectors(vectors_file):
    with open(vectors_file, "r") as f:
        vectors = json.load(f)
    return vectors

def load_ground_truth(gt_file):
    ground_truth = {}
    with open(gt_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                filename, author = row[0], row[1]
                ground_truth[filename] = author
    return ground_truth

def cosine_sim(vector1, vector2):
    """NOTE: vectors should be dicts mapping token to vals"""
    dot = 0.0
    # Iterate over smaller dictionary for efficiency :D
    if len(vector1) > len(vector2):
        vector1, vector2 = vector2, vector1
    for index, val in vector1.items():
        if index in vector2:
            dot += val * vector2[index]
    return dot


def cosine_distance(vec1, vec2):
    # Since vectors are normalized, cosine similarity is the dot product.
    return 1 - cosine_sim(vec1, vec2)

def okapi_sim(vector1, vector2):
    """ TODO: REFINE LATER for idf and term freq"""

    sim = 0.0
    for token, value in vector1.items():
        if token in vector2:
            sim += min(value, vector2[token])
    return sim

def okapi_distance(vector1, vector2):
    return 1 - okapi_sim(vector1, vector2)

def get_distance(vector1, vector2, metric):
    if metric == "cosine":
        return cosine_distance(vector1, vector2)
    elif metric == "okapi":
        return okapi_distance(vector1, vector2)
    else:
        print("Unknown metric. Defaulting to cosine.")
        return cosine_distance(vector1,vector2)

def majority_vote(neighbors, ground_truth):
    votes ={}
    for filename in neighbors:
        author  = ground_truth.get(filename, None)
        if author is not None:
            votes[author] = votes.get(author, 0) + 1
    if not votes:
        return None
    max_votes = max(votes.values())
    # Tie-breaker: choose alphabetically-first author
    #     among those with max votes
    candidates = []
    for authr, count in votes.items():
        if count == max_votes:
            candidates.append(authr)
    return sorted(candidates)[0]

def leave_one_out_knn(vectors, ground_truth, k, metric):
    filenames = list(vectors.keys())
    predictions = {}
    total  = len(filenames)
    for i,filename in enumerate(filenames):
        vector1 = vectors[filename]
        distances = []
        for other_filename in filenames:
            if filename == other_filename:
                continue
            vector2 = vectors[other_filename]
            distance = get_distance(vector1, vector2, metric)
            distances.append((distance, other_filename))
        # Sort by distance (smaller means they're more similar) and
        # take the k nearest neighs.
        def sort_key(item):
            return item[0]

        distances.sort(key=sort_key)

        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][1])

        predicted_auth = majority_vote(neighbors, ground_truth)
        predictions[filename] = predicted_auth
        # Progress bar
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} documents")

    return predictions

def save_predictions(predictions, output_file):
    with open(output_file, "w") as f:
        for filename, predicted_author in predictions.items():
            f.write(f"{filename},{predicted_author}\n")
        print(f"Saved predictions to {output_file}")


def main():
    if len(sys.argv) < 5:
        print("Usage: python knnAuthorship.py <vectors_json> <ground_truth_csv> <similarity_metric: cosine/okapi> <k> [output_predictions]")
        sys.exit(1)

    vectors_file = sys.argv[1]
    ground_truth_csv = sys.argv[2]
    similarity_metric = sys.argv[3]
    try:
        k = int(sys.argv[4])
    except ValueError:
        print("Error: k must be an integer!")
        sys.exit(1)

    if len(sys.argv) > 5:
        output_predictions = sys.argv[5]
    else:
        output_predictions = "knn_predictions.csv"

    print("Loading vectors...")
    vectors = load_vectors(vectors_file)
    print("Loading ground truth...")
    ground_truth = load_ground_truth(ground_truth_csv)

    print(f"Running leave-one-out KNN with k={k} and metric={similarity_metric}...")
    predictions = leave_one_out_knn(vectors, ground_truth, k, similarity_metric)

    print("Saving predictions...")
    save_predictions(predictions, output_predictions)

    print("KNN authorship attribution complete.")

if __name__ == "__main__":
    main()

# Example run: python .\knnAuthorship.py .\output\output_vectors.json .\output\output_ground_truth.csv okapi 3