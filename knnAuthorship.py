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