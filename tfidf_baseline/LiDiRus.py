from sklearn.metrics import matthews_corrcoef
import codecs
import json
from sklearn.linear_model import LogisticRegression


def build_feature_LiDiRus(row):
    if row.get("sentence1") is None:
        premise = str(row["premise"]).strip()
        hypothesis = row["hypothesis"]
    else:
        premise = str(row["sentence1"]).strip()
        hypothesis = row["sentence2"]
    label = row.get("label")
    res = f"{premise} {hypothesis}"
    return res, label


def build_features_LiDiRus(path, vect):
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))
    res = list(map(build_feature_LiDiRus, lines))
    texts = list(map(lambda x: x[0], res))
    labels = list(map(lambda x: x[1], res))
    ids = [x["idx"] for x in lines]
    return (vect.transform(texts), labels), ids


def fit_LiDiRus(train, labels):
    clf = LogisticRegression()
    return clf.fit(train, labels)


def eval_LiDiRus(train_path, val_path, test_path, vect):
    train, _ = build_features_LiDiRus(train_path, vect)
    val, _ = build_features_LiDiRus(val_path, vect)
    test, ids = build_features_LiDiRus(test_path, vect)
    clf = fit_LiDiRus(*train)
    test_pred = clf.predict(test[0])
    return clf, {
        "train": matthews_corrcoef(train[1], clf.predict(train[0])),
        "val": matthews_corrcoef(val[1], clf.predict(val[0])),
        "test": matthews_corrcoef(test[1], test_pred),
        "test_pred": [{"idx": idx, "label": label} for idx, label in zip(ids, test_pred)]
    }
