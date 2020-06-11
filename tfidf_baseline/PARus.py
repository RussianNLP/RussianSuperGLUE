import codecs
import json
from sklearn.linear_model import LogisticRegression


def build_feature_PARus(row):
    premise = str(row["premise"]).strip()
    choice1 = row["choice1"]
    choice2 = row["choice2"]
    label = row.get("label")
    question = "Что было ПРИЧИНОЙ этого?" if row["question"] == "cause" else "Что случилось в РЕЗУЛЬТАТЕ?"
    res = f"{premise} {question} {choice1} {choice2}"
    return res, label


def build_features_PARus(path, vect):
    with codecs.open(path, encoding='utf-8-sig') as reader:
        lines = reader.read().split("\n")
        lines = list(map(json.loads, filter(None, lines)))
    res = list(map(build_feature_PARus, lines))
    texts = list(map(lambda x: x[0], res))
    labels = list(map(lambda x: x[1], res))
    ids = [x["idx"] for x in lines]
    return (vect.transform(texts), labels), ids


def fit_PARus(train, labels):
    clf = LogisticRegression()
    return clf.fit(train, labels)


def eval_PARus(train_path, val_path, test_path, vect):
    train, _ = build_features_PARus(train_path, vect)
    val, _ = build_features_PARus(val_path, vect)
    test, ids = build_features_PARus(test_path, vect)
    clf = fit_PARus(*train)
    try:
        test_score = clf.score(*test)
    except ValueError:
        test_score = None
    test_pred = clf.predict(test[0])
    return clf, {
        "train": clf.score(*train),
        "val": clf.score(*val),
        "test": test_score,
        "test_pred": [{"idx": idx, "label": int(label)} for idx, label in zip(ids, test_pred)]
    }
