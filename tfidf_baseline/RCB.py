import codecs
import json
from sklearn.linear_model import LogisticRegression


def build_feature_RCB(row):
    premise = str(row["premise"]).strip()
    hypothesis = row["hypothesis"]
    label = row["label"]
    res = f"{premise} {hypothesis}"
    return res, label


def build_features_RCB(path, vect):
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))
    res = list(map(build_feature_RCB, lines))
    texts = list(map(lambda x: x[0], res))
    labels = list(map(lambda x: x[1], res))
    return vect.transform(texts), labels


def fit_RCB(train, labels):
    clf = LogisticRegression()
    return clf.fit(train, labels)


def eval_RCB(train_path, val_path, test_path, vect):
    train = build_features_RCB(train_path, vect)
    val = build_features_RCB(val_path, vect)
    test = build_features_RCB(test_path, vect)
    clf = fit_RCB(*train)
    return clf, {
        "train": clf.score(*train),
        "val": clf.score(*val),
        "test": clf.score(*test)
    }
