import codecs
import json
from sklearn.linear_model import LogisticRegression


def build_feature_RWSD(row):
    premise = str(row["text"]).strip()
    span1 = row["target"]["span1_text"]
    span2 = row["target"]["span2_text"]
    label = row["label"]
    res = f"{premise} {span1} {span2}"
    return res, label


def build_features_RWSD(path, vect):
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))
    res = list(map(build_feature_RWSD, lines))
    texts = list(map(lambda x: x[0], res))
    labels = list(map(lambda x: x[1], res))
    return vect.transform(texts), labels


def fit_RWSD(train, labels):
    clf = LogisticRegression()
    return clf.fit(train, labels)


def eval_RWSD(train_path, val_path, test_path, vect):
    train = build_features_RWSD(train_path, vect)
    val = build_features_RWSD(val_path, vect)
    test = build_features_RWSD(test_path, vect)
    clf = fit_RWSD(*train)
    return clf, {
        "train": clf.score(*train),
        "val": clf.score(*val),
        "test": clf.score(*test)
    }
