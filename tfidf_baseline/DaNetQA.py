import codecs
import json
from sklearn.linear_model import LogisticRegression


def build_feature_DaNetQA(row):
    res = str(row["question"]).strip()
    label = row.get("label")
    return res, label


def build_features_DaNetQA(path, vect):
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))
    res = list(map(build_feature_DaNetQA, lines))
    texts = list(map(lambda x: x[0], res))
    labels = list(map(lambda x: x[1], res))
    ids = [x["idx"] for x in lines]
    return (vect.transform(texts), labels), ids


def fit_DaNetQA(train, labels):
    clf = LogisticRegression()
    return clf.fit(train, labels)


def eval_DaNetQA(train_path, val_path, test_path, vect):
    train, _ = build_features_DaNetQA(train_path, vect)
    val, _ = build_features_DaNetQA(val_path, vect)
    test, ids = build_features_DaNetQA(test_path, vect)
    clf = fit_DaNetQA(*train)
    try:
        test_score = clf.score(*test)
    except ValueError:
        test_score = None
    test_pred = clf.predict(test[0])
    return clf, {
        "train": clf.score(*train),
        "val": clf.score(*val),
        "test": test_score,
        "test_pred": [{"idx": idx, "label": str(label).lower()} for idx, label in zip(ids, test_pred)]
    }
