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

    # cat_clusters = defaultdict(set)
    single_word_cats = []
    multi_word_cats = []

    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            word, cat = line.split('\t')
            if word == "NULL":
                continue
            if len(word.split(" ")) > 1:
                multi_word_cats.append((word, cat))
            else:
                if not get_word_id(word, word2id, lower):
                    continue
                single_word_cats.append((word, cat))

    return single_word_cats, multi_word_cats


def get_clustering_scores(language, word2id, embeddings, lower=False):
    single_word_cats, multi_word_cats = load_category_truth(language, word2id, lower)

    eval_embeddings = np.stack(
        [embeddings[get_word_id(word, word2id, lower)] for word, _ in single_word_cats])

    kmeans = KMeans(n_clusters=20, random_state=0)
    prediction = kmeans.fit_predict(eval_embeddings)

    for multi_word, _ in multi_word_cats:
        mw_embeddings = []
        for word in multi_word.split(' '):
            word_id = get_word_id(word, word2id, lower)
            if word_id:
                mw_embeddings.append(embeddings[word_id])
        word_pred = kmeans.transform(np.stack(mw_embeddings))
        mins = np.argmin(word_pred, axis=1)
        clusters = defaultdict(list)
        for pos, cluster in enumerate(mins):
            clusters[cluster].append(word_pred[pos][cluster])
        best = (None, 0, 0)
        for cluster, dists in clusters.items():
            cnt = len(dists)
            if cnt >= best[1]:
                _min = min(dists)
                if cnt > best[1] or _min < best[2]:
                    best = (cluster, cnt, _min)
        assert best[0] is not None
        prediction = np.append(prediction, best[0])

    truth = [cat for _, cat in single_word_cats + multi_word_cats]

    assert len(truth) == len(prediction)

    return {"ARI": adjusted_rand_score(truth, prediction)}
