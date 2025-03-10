import sys
import csv

def load_cluster_assignments(cluster_file):
    """Reads the cluster assignments from a CSV file. Maps filename -> cluster label."""
    files = {}
    with open(cluster_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            filename = row[0]
            cluster_label = row[1]
            files[filename] = cluster_label
    return files

def load_ground_truth(file):
    """Loads the ground truth from a CSV file. Maps filename -> true author."""
    pred_or_gt = {}
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                filename = row[0]
                author = row[1]
                pred_or_gt[filename] = author
    return pred_or_gt

def group_by_cluster(files, truth):
    """Groups files by cluster label and combines files with their ground truth author."""
    clusters = {}
    for filename, cluster in files.items():
        if cluster not in clusters:
            clusters[cluster] = [(filename, truth[filename])]
        else:
            clusters[cluster].append((filename, truth[filename]))
    return clusters

def rand_score(files, truth):
    """Calculate the rand score of a clustering given a dictionary of files->clusters and files->GT authors."""
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for file1, cluster1 in files.items():
        for file2, cluster2 in files.items():
            if file1 == file2:
                continue
            author1 = truth[file1]
            author2 = truth[file2]
            if cluster1 == cluster2 and author1 == author2:
                tp += 1
            elif cluster1 != cluster2 and author1 != author2:
                tn += 1
            elif cluster1 == cluster2 and author1 != author2:
                fp += 1
            elif cluster1 != cluster2 and author1 == author2:
                fn += 1

    return (tp + tn) / (tp + tn + fp + fn)


def evaluate_clusters(clusters):
    """Computes metrics for each cluster and the average purity of the clustering."""
    info = {}
    purity_running_total = 0

    for cluster, docs in clusters.items():
        n_files = len(docs)
        authors = {}
        for file, author in docs:
            if author not in authors:
                authors[author] = 1
            else:
                authors[author] += 1

        max_author = None
        max_author_count = None
        for author, count in authors.items():
            if not max_author or count > authors[max_author]:
                max_author = author
                max_author_count = count

        purity = max_author_count / n_files
        purity_running_total += purity
        distribution = {author: count / n_files for author, count in authors.items()}

        info[cluster] = {
            "size": n_files,
            "plurality": max_author,
            "purity": purity,
            "distribution": distribution
        }

    avg_purity = purity_running_total / len(clusters)

    return info, avg_purity

def per_author_metrics(clusters, truth, cluster_info):
    """Computes metrics for each author based on the clustering results."""
    authors = []
    info = {}

    for file, author in truth.items():
        if author not in authors:
            authors.append(author)

    for author in authors:
        n_clusters_found = 0
        n_clusters_plurality = 0
        total_docs_under_plurality = 0
        total_docs_correctly_placed = 0
        total_docs_written = 0

        for cluster, docs in clusters.items():
            n_docs = len(docs)
            n_docs_author = 0

            for doc in docs:
                if doc[1] == author:
                    n_docs_author += 1

            total_docs_written += n_docs_author

            if n_docs_author > 0:
                n_clusters_found += 1

            if cluster_info[cluster]["plurality"] == author:
                total_docs_correctly_placed += n_docs_author
                total_docs_under_plurality += n_docs
                n_clusters_plurality += 1

        if total_docs_written > 0:
            recall = total_docs_correctly_placed / total_docs_written
        else:
            recall = "N/A"
        if total_docs_under_plurality > 0:
            precision = total_docs_correctly_placed / total_docs_under_plurality
        else:
            precision = "N/A"

        info[author] = {
            "total_docs": total_docs_written,
            "clusters_found": n_clusters_found,
            "clusters_plurality": n_clusters_plurality,
            "recall": recall,
            "precision": precision
        }

    return info

def main():
    if len(sys.argv) < 3:
        print("Usage: python clusteringEvaluation.py <clusters_csv> <ground_truth_csv>")
        sys.exit(1)
    cluster_file = sys.argv[1]
    gt_file = sys.argv[2]

    files = load_cluster_assignments(cluster_file)
    gt = load_ground_truth(gt_file)
    clusters = group_by_cluster(files, gt)
    cluster_info, avg_purity = evaluate_clusters(clusters)

    print("Cluster Metrics:")
    for cluster, info in cluster_info.items():
        print(f"Cluster {cluster}:")
        print(f"  Size: {info['size']}")
        print(f"  Plurality Author: {info['plurality']}")
        print(f"  Purity: {info['purity']}")
        print(f"  Distribution: {info['distribution']}")

    print("\nAuthor Metrics:")
    author_metrics = per_author_metrics(clusters, gt, cluster_info)

    for author, metrics in sorted(author_metrics.items()):
        print(f"Author: {author}")
        print(f"  Clusters where Author is found: {metrics['clusters_found']}")
        print(f"  Clusters where Author is Plurality: {metrics['clusters_plurality']}")
        print(f"  Recall: {metrics['recall']}")
        print(f"  Precision: {metrics['precision']}")

    print(f"\nAverage cluster purity: {avg_purity}")
    rand = rand_score(files, gt)
    print(f"Clustering Rand Score: {rand}")

if __name__ == "__main__":
    main()
