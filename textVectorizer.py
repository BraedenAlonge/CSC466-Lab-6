import os
import sys
import re # To identify regular expressions
import json
import math
from collections import defaultdict, Counter
from porterstemmer import PorterStemmer

# Placeholder until my 90-minute file transfer finishes 😣😭
STOP_WORDS = {"the", "is", "at", "which", "on", "and", "a", "an", "of", "for", "to", "in"}

def tokenizer(text):
    """Tokenizes text by extracting word characters and converting to lowercase."""
    return re.findall(r'\b\w+\b', text.lower())

def process_tokens(tokens, remove_stopwords=False, use_stemming=False):
    """ Removes stop words (if applied) and uses stemming (if applied) to the token list"""
    processed_tokens = []
    if use_stemming:
        stemmer = PorterStemmer()
    else:
        stemmer = None
    for token in tokens:
        if remove_stopwords and token in STOP_WORDS:
            continue
        elif use_stemming and stemmer:
            token = stemmer.stem(token, 0, len(token) - 1)
        processed_tokens.append(token)
    return processed_tokens

def traverse_dataset(root_dir):
    """
    Traverse the dataset directories (expects 'C50train' and 'C50test' inside root_dir)
    and returns a list of documents as tuples: (filename, author, text).
    """
    docs = []
    for subset in ['C50train', 'C50test']:
        subset_dir = os.path.join(root_dir,subset)
        if not os.path.exists(subset_dir):
            print(f"traverse_dataset: Directory {subset_dir} does not exist.")
            sys.exit(1)

        for author in os.listdir(subset_dir):
            author_dir = os.path.join(subset_dir,author)
            if os.path.isdir(author_dir):
                for filename in os.listdir(author_dir):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(author_dir,filename)
                        try:
                            with open(file_path,'r', encoding = 'utf-8') as f:
                                text = f.read()
                            docs.append((filename, author, text))
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
    return docs


def build_vocabulary(documents, remove_stopwords=False, use_stemming=False,min_df=1):
    """ Build a vocab from the documents. Return a dictionary vocab mapping token to  unique index, as
    well as df_counts dictionary mapping token to document frequency"""
    df_counts = defaultdict(int)

    for a,b, text in documents:
        tokens = tokenizer(text)
        tokens  = process_tokens(tokens, remove_stopwords, use_stemming)

        # Count unique tokens
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df_counts[token] += 1
    # Filter tokens not meeting minimum document frequency
    vocabulary = {}
    index = 0
    for token, count in df_counts.items():
        if count >= min_df:
            vocabulary[token] = index
            index += 1
    return vocabulary, df_counts

def compute_tf_idf(documents, vocabulary, df_counts, remove_stopwords=False, use_stemming=False):
    """ Compute normalized tf-idf vectors for each doc.
    Return a dict mapping filename > sparse vector representation (token_index -> weight)"""
    N = len(documents)
    # Compute idf vals using smoothed formula
    idf = {}
    for token in vocabulary:
        df = df_counts[token]
        idf[token] = math.log((N + 1) / (df + 1)) + 1

    doc_vectors = {}
    for filename, x, text in documents:
        tokens = tokenizer(text)
        tokens = process_tokens(tokens, remove_stopwords, use_stemming)
        tf_counts = Counter(tokens)
        vector = {}
        norm = 0.0
        for token, count in tf_counts.items():
            if token in vocabulary:
                weight = count * idf[token]
                vector[vocabulary[token]] = weight
                norm += weight ** 2
        norm = math.sqrt(norm)
        if norm > 0:
            for key in vector:
                vector[key] /= norm
        doc_vectors[filename] = vector
    return doc_vectors

def save_vectors(vectors, output_file):
    """ Save doc vectors to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(vectors, f)
    print(f"Saved vectors to {output_file}")

def save_ground_truth(documents, output_file):
    """ Save ground truth mapping (filename, auth) to a CSV file"""
    with open(output_file, 'w') as f:
        for filename, author, text in documents:
            f.write(f"{filename}, {author}\n")
    print(f"Saved ground truth file to {output_file}")


def main():
    # NOTE: For optional params, use all 3 if you want to use min_df and at lest 2 for stem
    # Just do not specify true if not needed. ie. python textVectorizer.py ... false x 4
    # Or python textVectorizer.py ... f true
    # Example run:  python textVectorizer.py "./" "./output/output_vectors.txt" "./output/output_ground_truth.csv" true true
   if len(sys.argv) < 4 or len(sys.argv) > 7:
       print("Usage: python textVectorizer.py <dataset_root> <output_vectors>"
             " <output_ground_truth> [remove_stopwords] [stem] [min_df]")
       sys.exit(1)

   dataset_root = sys.argv[1]
   output_vectors = sys.argv[2]
   output_ground_truth = sys.argv[3]

   # Optional args
   if len(sys.argv) > 4 and sys.argv[4].upper() == 'TRUE':
       remove_stopwords = True
   else: remove_stopwords = False
   if len(sys.argv) > 5 and sys.argv[4].upper() == 'TRUE':
       stem = True
   else:
       stem = False
   if len(sys.argv) == 7:
       try:
           min_df = int(sys.argv[5])
       except ValueError:
           print("Error: min_df must be an integer.")
           sys.exit(1)
   else:
       min_df = 1

   print("Traversing the dataset...")
   documents = traverse_dataset(dataset_root)
   if not documents:
       print("No documents found. Check the dataset directory structure.")
       sys.exit(1)
   print("Found ", len(documents), " documents.")
   # Vocabulary
   print("Building vocabulary...")
   vocabulary, df_counts = build_vocabulary(documents, remove_stopwords, stem, min_df)
   print("Vocabulary size: ", len(vocabulary))
   # tf-idf vectors
   print("Computing tf-idf vectors...")
   doc_vectors = compute_tf_idf(documents, vocabulary, df_counts, remove_stopwords, stem)
   # Save output files
   print("Saving output files...")
   save_vectors(doc_vectors, output_vectors)
   save_ground_truth(documents, output_ground_truth)


if __name__ == "__main__":
    main()


