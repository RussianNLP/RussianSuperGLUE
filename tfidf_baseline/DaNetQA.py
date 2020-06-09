import codecs
import json
from sklearn.linear_model import LogisticRegression


def build_feature_DaNetQA(row):
    res = str(row["question"]).strip()
    label = row["label"]
    return res, label


def build_features_DaNetQA(path, vect):
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))
    res = list(map(build_feature_DaNetQA, lines))
    texts = list(map(lambda x: x[0], res))
    labels = list(map(lambda x: x[1], res))
    return vect.transform(texts), labels


def fit_DaNetQA(train, labels):
    clf = LogisticRegression()
    return clf.fit(train, labels)


def eval_DaNetQA(train_path, val_path, test_path, vect):
    train = build_features_DaNetQA(train_path, vect)
    val = build_features_DaNetQA(val_path, vect)
    test = build_features_DaNetQA(test_path, vect)
    clf = fit_DaNetQA(*train)
    return clf, {
        "train": clf.score(*train),
        "val": clf.score(*val),
        "test": clf.score(*test)
    }
