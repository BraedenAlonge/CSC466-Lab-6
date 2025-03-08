import sys
import csv
import numpy as np

def load_predict_or_gt(file):
    """Loads the predictions or ground truth from a CSV file,
    csv format: filename, predicted_author/true_author
    Returns a dictionary mapping filename -> predicted/true author. """
    pred_or_gt = {}
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                filename = row[0]
                author = row[1]
                pred_or_gt[filename] = author
    return pred_or_gt

def compute_metrics(ground_truth, predictions):
    """Computes the following:
        hits: correct documents
        misses: misclassified documents, but correct author
        false_positives: wrong author

        Also computes overall accuracy.
        Returns stats (dict of ech author to counts),
         total_correct, total_docs, authors (sorted list of unique auths)"""

    # Get sorted list of all authors present in ground truth.
    authors = sorted(list(set(ground_truth.values())))

    #intitialize per-author stats
    stats = {author : {"hits": 0, "misses": 0, "false_positives": 0} for author in authors}
    total_correct = 0
    total_docs = len(ground_truth)

    # For each doc, update the counts
    for filename, actual in ground_truth.items():
        pred = predictions.get(filename, None)
        if pred is None:
            continue
        if pred == actual:
            stats[actual]["hits"] += 1
            total_correct += 1
        else:
            stats[actual]["misses"] += 1
            # Increment false pos for predicted auth
            if pred in stats:
                stats[pred]["false_positives"] += 1
            else:
                # In case (somehow) author isnt in ground truth list
                stats[pred] = {"hits": 0, "misses": 0, "false_positives": 1}
                if pred not in authors:
                    authors.append(pred)

    # Sort again if somehow auths were added
    authors = sorted(authors)

    return stats, total_correct, total_docs, authors

# Reused code
def build_confusion_matrix(ground_truth, predictions, authors):
    """
    Builds a confusion matrix as a dictionary of dictionaries.
    The keys for both dimensions are the author names.
    Each cell [actual][predicted] contains the count of documents with that outcome.
    """
    matrix = {actual: {pred: 0 for pred in authors} for actual in authors}
    for filename, actual in ground_truth.items():
        pred = predictions.get(filename, None)
        if pred is None:
            continue
        if pred in matrix[actual]:
            matrix[actual][pred] += 1
        else:
            matrix[actual][pred] = 1
    return matrix

def save_confusion_matrix(confusion_matrix, authors, output_file):
    """Save confusion matrix to csv file. first row and col list the author names."""
    with open(output_file, 'w', newline="") as f:
        writer = csv.writer(f)
        header = ["Actual \\ Predicted"] + authors
        writer.writerow(header)
        for actual in authors:
            row = [actual]
            for pred in authors:
                row.append(confusion_matrix[actual].get(pred,0))
            writer.writerow(row)
    print(f"Saved confusion matrix to {output_file}")

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python classifierEvaluation.py <predictions_csv> <ground_truth_csv> [confusion_matrix_output_csv]")
        sys.exit(1)

    pred_file = sys.argv[1]
    gt_file = sys.argv[2]
    if len(sys.argv) > 3:
        conf_matrix_file = sys.argv[3]
    else:
        conf_matrix_file =  "confusion_matrix.csv"
    # Load predictions and gt files
    predictions = load_predict_or_gt(pred_file)
    ground_truth = load_predict_or_gt(gt_file)

    # Compute per-author metrics and overall accuracy
    stats, total_correct, total_docs, authors = compute_metrics(ground_truth, predictions)
    if total_docs > 0:
        overall_accuracy = total_correct / total_docs
    else:
        overall_accuracy = 0.00

    # Print the per-author evals
    print("Per-author evaluation:")
    for author in authors:
        hits = stats[author]["hits"]
        misses = stats[author]["misses"]
        false_positives = stats[author]["false_positives"]
        precision = hits / (hits + false_positives) if (hits + false_positives) > 0 else 0.0
        recall = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall )> 0 else 0.0
        print(f"Author: {author}")
        print(f"  Hits: {hits}")
        print(f"  Misses: {misses}")
        print(f"  False Positives: {false_positives}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}\n")

    # Print overall accuracy.
    print(f"Overall Accuracy: {overall_accuracy:.3f} ({total_correct}/{total_docs})")

    # Build and save the confusion matrix.
    confusion_matrix = build_confusion_matrix(ground_truth, predictions, authors)
    save_confusion_matrix(confusion_matrix, authors, conf_matrix_file)

if __name__ == "__main__":
    main()