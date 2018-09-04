import os
import io
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

from src.utils import get_word_id


MONOLINGUAL_EVAL_PATH = 'data/monolingual'


def load_category_data(language, word2id, embeddings, lower):
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
            word = word.strip()
            if word == "NULL":
                continue
            tokens = word.split(" ")
            if len(tokens) > 1:
                mw_embeddings = []
                for token in tokens:
                    word_id = get_word_id(token, word2id, lower)
                    if word_id:
                        mw_embeddings.append(embeddings[word_id])
                assert len(mw_embeddings)
                multi_word_cats.append((np.stack(mw_embeddings), cat))
            else:
                word_id = get_word_id(word, word2id, lower)
                if word_id:
                    single_word_cats.append((embeddings[word_id], cat))

    return single_word_cats, multi_word_cats


def get_multiword_predictions(kmeans, multi_word_cats):
    # note: kmeans has to be ALREADY TRAINED
    mw_predictions = []
    for mw_embeddings, _ in multi_word_cats:
        word_pred = kmeans.transform((mw_embeddings))
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
        mw_predictions.append(best[0])
    return np.array(mw_predictions)


def get_clustering_scores(
    language1, word2id1, embeddings1, language2=None, word2id2=None, embeddings2=None, lower=False
):

    single_word_cats, multi_word_cats = load_category_data(
        language1, word2id1, embeddings1, lower)

    if language2:
        single_word_cats2, multi_word_cats2 = load_category_data(
            language2, word2id2, embeddings2, lower)
        single_word_cats += single_word_cats2
        multi_word_cats += multi_word_cats2

    eval_embeddings = np.stack(embedding for embedding, _ in single_word_cats)

    kmeans = KMeans(n_clusters=20, random_state=0)
    prediction = kmeans.fit_predict(eval_embeddings)
    multiword_predictions = get_multiword_predictions(kmeans, multi_word_cats)
    prediction = np.append(prediction, multiword_predictions)

    truth = [cat for _, cat in single_word_cats + multi_word_cats]

    assert len(truth) == len(prediction)

    return {"ARI": adjusted_rand_score(truth, prediction)}
