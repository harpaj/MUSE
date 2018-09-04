import os
import io
from collections import defaultdict, Counter
import string

import hdbscan
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import adjusted_rand_score
import numpy as np

from src.utils import get_word_id


MONOLINGUAL_EVAL_PATH = 'data/monolingual'
DICT_PATH = 'data/crosslingual/dictionaries'


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
            line = line.replace('""', '"')
            word, cat = line.split('\t')
            word = word.strip()
            if word == "NULL":
                continue
            tokens = word.split(" ")
            if len(tokens) > 1:
                mw_embeddings = []
                for token in tokens:
                    token = token.strip(string.punctuation + " ")
                    if not token:
                        continue
                    word_id = get_word_id(token, word2id, lower)
                    if word_id:
                        mw_embeddings.append(embeddings[word_id])
                assert len(mw_embeddings)
                multi_word_cats.append((np.stack(mw_embeddings), word.lower(), cat))
            else:
                word_id = get_word_id(word, word2id, lower)
                if word_id:
                    single_word_cats.append((embeddings[word_id], word.lower(), cat))

    return single_word_cats, multi_word_cats


def get_multiword_predictions(cluster_model, multi_word_cats):
    # note: cluster model has to be ALREADY TRAINED
    mw_predictions = []
    for mw_embeddings, _, _ in multi_word_cats:
        clusters = defaultdict(list)
        if hasattr(cluster_model, "transform") or hasattr(cluster_model, "generate_prediction_data"):
            try:
                word_pred = cluster_model.transform(mw_embeddings)
            except AttributeError:
                word_pred = hdbscan.prediction.membership_vector(
                    cluster_model, mw_embeddings)
            mins = np.argmin(word_pred, axis=1)
            for pos, cluster in enumerate(mins):
                clusters[cluster].append(word_pred[pos][cluster])
        else:
            word_pred = cluster_model.predict(mw_embeddings)
            for cluster in word_pred:
                clusters[cluster].append(1)
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


def get_prediction_and_truth(single_word_cats, multi_word_cats, algorithm, kwargs):
    eval_embeddings = np.stack(embedding for embedding, _, _ in single_word_cats)
    if algorithm == "kmeans":
        cluster_model = KMeans(random_state=0, n_jobs=-1, **kwargs)
    elif algorithm == "affinity":
        cluster_model = AffinityPropagation(**kwargs)
    elif algorithm == "hdbscan":
        cluster_model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, **kwargs)
    else:
        raise AssertionError("Clustering algorithm {} is not supported".format(algorithm))
    prediction = cluster_model.fit_predict(eval_embeddings)
    multiword_predictions = get_multiword_predictions(cluster_model, multi_word_cats)
    prediction = np.append(prediction, multiword_predictions)

    truth = [cat for _, _, cat in single_word_cats + multi_word_cats]

    assert len(truth) == len(prediction)

    return prediction, truth


def get_clustering_scores(
    language1, word2id1, embeddings1, language2=None, word2id2=None, embeddings2=None, lower=False,
    algorithm="affinity", kwargs={}
):

    single_word_cats, multi_word_cats = load_category_data(
        language1, word2id1, embeddings1, lower)

    if language2:
        single_word_cats2, multi_word_cats2 = load_category_data(
            language2, word2id2, embeddings2, lower)
        single_word_cats += single_word_cats2
        multi_word_cats += multi_word_cats2

    prediction, truth = get_prediction_and_truth(
        single_word_cats, multi_word_cats, algorithm, kwargs)

    return {"ARI": adjusted_rand_score(truth, prediction)}


def load_dictionary(lang1, lang2):
    filepath = os.path.join(DICT_PATH, "{}-{}.txt".format(lang1, lang2))
    if not os.path.exists(filepath):
        return None

    dictionary = defaultdict(list)
    with open(filepath) as inp:
        for line in inp:
            line = line.rstrip()
            _from, _to = line.split('\t')
            dictionary[_from].append(_to)
    return dictionary


def get_clustering_scores_cluster_seperately(
    language1, word2id1, embeddings1, language2, word2id2, embeddings2, lower=False,
    algorithm="affinity", kwargs={}
):

    single_word_cats1, multi_word_cats1 = load_category_data(
        language1, word2id1, embeddings1, lower)

    prediction1, truth1 = get_prediction_and_truth(
        single_word_cats1, multi_word_cats1, algorithm, kwargs)

    single_word_cats2, multi_word_cats2 = load_category_data(
        language2, word2id2, embeddings2, lower)

    prediction2, truth2 = get_prediction_and_truth(
        single_word_cats2, multi_word_cats2, algorithm, kwargs)

    # create clusters with real words and position for pred 1 and pred 2
    # for each cluster in pred 1, for each word in cluster, check which category the translation is in
    # count the target lang categories
    # replace source cluster id with most frequent target cluster id
    dictionary = load_dictionary(language1, language2)

    clusters1 = defaultdict(set)
    for idx, cluster in enumerate(prediction1[:len(single_word_cats1)]):
        for translation in dictionary.get(single_word_cats1[idx][1], []):
            clusters1[translation].add(cluster)

    clusters2 = defaultdict(Counter)
    for idx, cluster in enumerate(prediction2[:len(single_word_cats2)]):
        for translation_cluster in clusters1[single_word_cats2[idx][1]]:
            clusters2[cluster][translation_cluster] += 1

    cluster_mapping = {c2: c1.most_common(1)[0][0] for c2, c1 in clusters2.items()}

    # TODO: Find a meaningful fallback if c is not in cluster_mapping
    prediction2 = [cluster_mapping.get(c, c) for c in prediction2]

    prediction = np.append(prediction1, prediction2)
    truth = np.append(truth1, truth2)

    return {"ARI": adjusted_rand_score(truth, prediction)}
