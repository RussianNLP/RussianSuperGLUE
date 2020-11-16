from sklearn.metrics.pairwise import cosine_similarity
import jsonlines
import numpy as np
from collections import Counter
import string
import re
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = [0]
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    correct_ids = []
    for prediction, passage in zip(predictions, dataset):
        prediction = prediction["label"]
        for qa in passage['qas']:
            total += 1
            ground_truths = list(map(lambda x: x['text'], qa.get("answers", "")))

            _exact_match = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            if int(_exact_match) == 1:
                correct_ids.append(qa['idx'])
            exact_match += _exact_match

            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = exact_match / total
    f1 = f1 / total
    return exact_match, f1


def eval_RuCoS(train_path, val_path, test_path, vect):
    test_score, test_pred = eval_part(test_path, vect)
    return None, {
        "train": eval_part(train_path, vect)[0],
        "val": eval_part(val_path, vect)[0],
        "test": test_score,
        "test_pred": test_pred
    }


def eval_part(path, vect):
    with jsonlines.open(path) as reader:
        lines = list(reader)
    preds = []
    for row in lines:
        pred = get_row_pred(row, vect)
        preds.append({
            "idx": row["idx"],
            "label": pred
        })
    return evaluate(lines, preds), preds


def get_row_pred(row, vect):
    text = vect.transform([row["passage"]["text"].replace("\n@highlight\n", " ")])
    res = []
    words = [
        row["passage"]["text"][x["start"]: x["end"]]
        for x in row["passage"]["entities"]]
    for line in row["qas"]:
        line_candidates = []
        for word in words:
            line_candidates.append(line["query"].replace("@placeholder", word))
        cos = cosine_similarity(text, vect.transform(line_candidates))
        pred = np.array(words)[cos.argsort()[0][-1]]
        res.append(pred)
    return " ".join(res)
