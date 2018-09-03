import os
import io
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

from src.utils import get_word_id


MONOLINGUAL_EVAL_PATH = 'data/monolingual'


def load_category_truth(language, word2id, lower):
    filepath = os.path.join(MONOLINGUAL_EVAL_PATH, language, 'categories.tsv')
    if not os.path.exists(filepath):
        return None

    cat_clusters = defaultdict(set)
    word_cats = []

    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            word, cat = line.split('\t')
            if word == "NULL":
                continue
            if len(word.split(" ")) > 1:
                continue
            if not get_word_id(word, word2id, lower):
                continue
            cat_clusters[cat].add(word)
            word_cats.append((word, cat))

    return cat_clusters, word_cats


def get_clustering_scores(language, word2id, embeddings, lower=False):
    true_clusters, word_cats = load_category_truth(language, word2id, lower)

    eval_embeddings = np.stack(
        [embeddings[get_word_id(word, word2id, lower)] for word, _ in word_cats])

    prediction = KMeans(n_clusters=20, random_state=0).fit_predict(eval_embeddings)
    truth = [cat for _, cat in word_cats]

    return {"ARI": adjusted_rand_score(truth, prediction)}
