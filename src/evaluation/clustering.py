from collections import defaultdict, Counter
import io
from logging import getLogger
import os
import string

import hdbscan
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score
import numpy as np

from src.utils import get_word_id

logger = getLogger()

MONOLINGUAL_EVAL_PATH = 'data/monolingual'
DICT_PATH = 'data/crosslingual/dictionaries'
CLUSTER_PATH = 'data/crosslingual/clusters'


def load_training_data(language, word2id, embeddings, lower, maxlen=5000):
    filepath = os.path.join(MONOLINGUAL_EVAL_PATH, language, 'aspect_terms.tsv')
    if not os.path.exists(filepath):
        return None

    embedding_words = []
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word, _ = line.rstrip().split('\t')
            word_id = get_word_id(word, word2id, lower)
            if word_id:
                embedding_words.append((embeddings[word_id], word, None))
            if len(embedding_words) >= maxlen:
                break
    return embedding_words


def load_evalutation_data(language, word2id, embeddings, lower):
    filepath = os.path.join(MONOLINGUAL_EVAL_PATH, language, 'categories.tsv')
    if not os.path.exists(filepath):
        return None

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
                multi_word_cats.append((np.stack(mw_embeddings).astype("float64"), word.lower(), cat))
            else:
                word_id = get_word_id(word, word2id, lower)
                if word_id:
                    single_word_cats.append((embeddings[word_id], word.lower(), cat))

    return single_word_cats, multi_word_cats


def get_attention_clusters(language, cl=False):
    if not cl:
        filepath = os.path.join(MONOLINGUAL_EVAL_PATH, language, 'aspect_embeddings.txt')
    else:
        filepath = os.path.join(CLUSTER_PATH, language + ".txt")
    return np.loadtxt(filepath)


def get_multiword_predictions(cluster_model, multi_word_cats):
    # note: cluster model has to be ALREADY TRAINED
    mw_predictions = []
    for mw_embeddings, _, _ in multi_word_cats:
        clusters = defaultdict(list)
        if hasattr(cluster_model, "transform") or hasattr(cluster_model, "generate_prediction_data"):
            if hasattr(cluster_model, "transform"):
                word_pred = cluster_model.transform(mw_embeddings)
            else:
                word_pred = hdbscan.prediction.membership_vector(cluster_model, mw_embeddings)
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


def print_clusters(predictions, words):
    clusters = defaultdict(list)
    for cl, word in zip(predictions, words):
        clusters[cl].append(word)
    for cluster, values in clusters.items():
        logger.info("Cluster {}: {}".format(cluster, values))


def get_prediction_and_truth(training_data, single_word_cats, multi_word_cats, algorithm, kwargs):
    train_embeddings = np.stack(embedding for embedding, _, _ in training_data).astype("float64")
    eval_embeddings = np.stack(embedding for embedding, _, _ in single_word_cats).astype("float64")
    if algorithm == "kmeans":
        cluster_model = KMeans(random_state=0, n_jobs=-1, **kwargs)
    elif algorithm == "attention":
        centroids = kwargs["centroids"]
        cluster_model = KMeans(random_state=0, n_jobs=-1, n_clusters=len(centroids), init=centroids)
        cluster_model.cluster_centers_ = centroids
    elif algorithm == "affinity":
        cluster_model = AffinityPropagation(**kwargs)
    elif algorithm == "hdbscan":
        cluster_model = hdbscan.HDBSCAN(
            core_dist_n_jobs=-1, prediction_data=True, min_samples=1, **kwargs)
    else:
        raise AssertionError("Clustering algorithm {} is not supported".format(algorithm))
    if algorithm != "attention":
        train_prediction = cluster_model.fit_predict(train_embeddings)
    if algorithm != "hdbscan":
        prediction = cluster_model.predict(eval_embeddings)
    else:
        prediction = hdbscan.prediction.approximate_predict(cluster_model, eval_embeddings)[0]
    multiword_predictions = get_multiword_predictions(cluster_model, multi_word_cats)
    prediction = np.append(prediction, multiword_predictions)
    if algorithm == "attention":
        train_prediction = prediction[:len(single_word_cats)]

    print_clusters(prediction, [w for _, w, _ in single_word_cats + multi_word_cats])

    truth = [cat for _, _, cat in single_word_cats + multi_word_cats]

    assert len(truth) == len(prediction)

    return prediction, truth, train_prediction


def get_clustering_scores(
    language1, word2id1, embeddings1, language2=None, word2id2=None, embeddings2=None, lower=False,
    algorithm="affinity", kwargs={}, full_training_data=False
):

    single_word_cats, multi_word_cats = load_evalutation_data(
        language1, word2id1, embeddings1, lower)

    if full_training_data:
        training_data = load_training_data(language1, word2id1, embeddings1, lower)
    else:
        training_data = single_word_cats

    if language2:
        single_word_cats2, multi_word_cats2 = load_evalutation_data(
            language2, word2id2, embeddings2, lower)
        if full_training_data:
            training_data2 = load_training_data(language2, word2id2, embeddings2, lower)
        else:
            training_data2 = single_word_cats2
        single_word_cats += single_word_cats2
        multi_word_cats += multi_word_cats2
        training_data += training_data2

    if algorithm == "attention":
        if not language2:
            kwargs["centroids"] = get_attention_clusters(language1, cl=False)
        else:
            kwargs["centroids"] = get_attention_clusters(
                "{}-{}".format(language1, language2), cl=True)

    prediction, truth, _ = get_prediction_and_truth(
        training_data, single_word_cats, multi_word_cats, algorithm, kwargs)

    return {
        "ARI": adjusted_rand_score(truth, prediction),
        "HOMO": homogeneity_score(truth, prediction),
        "COMP": completeness_score(truth, prediction)
    }


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
    algorithm="affinity", kwargs={}, full_training_data=False
):

    single_word_cats1, multi_word_cats1 = load_evalutation_data(
        language1, word2id1, embeddings1, lower)

    if full_training_data:
        training_data1 = load_training_data(language1, word2id1, embeddings1, lower)
    else:
        training_data1 = single_word_cats1

    kwargs["centroids"] = get_attention_clusters(language1, cl=False)
    prediction1, truth1, train_prediction1 = get_prediction_and_truth(
        training_data1, single_word_cats1, multi_word_cats1, algorithm, kwargs)

    single_word_cats2, multi_word_cats2 = load_evalutation_data(
        language2, word2id2, embeddings2, lower)

    if full_training_data:
        training_data2 = load_training_data(language2, word2id2, embeddings2, lower)
    else:
        training_data2 = single_word_cats2

    kwargs["centroids"] = get_attention_clusters(language2, cl=False)
    prediction2, truth2, train_prediction2 = get_prediction_and_truth(
        training_data2, single_word_cats2, multi_word_cats2, algorithm, kwargs)

    dictionary = load_dictionary(language1, language2)

    clusters1 = defaultdict(set)
    for idx, cluster in enumerate(train_prediction1):
        for translation in dictionary.get(training_data1[idx][1], []):
            clusters1[translation].add(cluster)

    clusters2 = defaultdict(Counter)
    for idx, cluster in enumerate(train_prediction2):
        for translation_cluster in clusters1[training_data2[idx][1]]:
            clusters2[cluster][translation_cluster] += 1

    cluster_mapping = {c2: c1.most_common(1)[0][0] for c2, c1 in clusters2.items()}

    # TODO: Find a meaningful fallback if c is not in cluster_mapping
    prediction2 = [cluster_mapping.get(c, c) for c in prediction2]

    prediction = np.append(prediction1, prediction2)
    truth = np.append(truth1, truth2)

    return {
        "ARI": adjusted_rand_score(truth, prediction),
        "HOMO": homogeneity_score(truth, prediction),
        "COMP": completeness_score(truth, prediction)
    }
