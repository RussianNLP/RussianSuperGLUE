import functools
import jsonlines
from sklearn.metrics.pairwise import cosine_similarity


class MuSeRCMetrics:

    @staticmethod
    def per_question_metrics(dataset, output_map):
        P = []
        R = []
        for n, example in enumerate(dataset):
                    predictedAns = example
                    correctAns = output_map[n]
                    predictCount = sum(predictedAns)
                    correctCount = sum(correctAns)
                    assert math.ceil(sum(predictedAns)) == sum(predictedAns), "sum of the scores: " + str(sum(predictedAns))
                    agreementCount = sum([a * b for (a, b) in zip(correctAns, predictedAns)])
                    p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
                    r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
                    P.append(p1)
                    R.append(r1)

        pAvg = Measures.avg(P)
        rAvg = Measures.avg(R)
        f1Avg = 2 * Measures.avg(R) * Measures.avg(P) / (Measures.avg(P) + Measures.avg(R))
        return [pAvg, rAvg, f1Avg]

    @staticmethod
    def exact_match_metrics_origin(dataset, output_map, delta):
        EM = []
        for n, example in enumerate(dataset):
            predictedAns = example
            correctAns = output_map[n]

            em = 1.0 if sum([abs(i - j) for i, j in zip(correctAns, predictedAns)]) <= delta  else 0.0
            EM.append(em)
        return Measures.avg(EM)

    @staticmethod
    def exact_match_simple(dataset, output_map):
        EM = []
        for n, example in enumerate(dataset):
            predictedAns = example
            correctAns = output_map[n]
            if predictedAns == correctAns:
                em = 1
            else:
                em = 0
            EM.append(em)
        return sum(EM)/len(EM)

    @staticmethod
    def per_dataset_metric(dataset, output_map):
        """
        dataset = [[0,1,1], [0,1]]
        output_map = [[0,1,0], [0,1]]
        """
        agreementCount = 0
        correctCount = 0
        predictCount = 0
        for n, example in enumerate(dataset):
                predictedAns = example
                correctAns = output_map[n]
                predictCount += sum(predictedAns)
                correctCount += sum(correctAns)
                agreementCount += sum([a * b for (a, b) in zip(correctAns, predictedAns)])

        p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
        r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
        return [p1, r1, 2 * r1 * p1 / (p1 + r1)]

    @staticmethod
    def avg(l):
        return functools.reduce(lambda x, y: x + y, l) / len(l)

    
def MuSeRC_metrics(pred, labels):
    metrics = MuSeRCMetrics()
    em = metrics.exact_match_simple(pred, labels)
    em0 = metrics.exact_match_metrics_origin(pred, labels, 0)
    f1 = metrics.per_dataset_metric(pred, labels)
    f1a = f1[-1]
    return em0, f1a


Measures = MuSeRCMetrics


def eval_MuSeRC(train_path, val_path, test_path, vect):
    test_score, test_pred = eval_part_MuSeRC(test_path, vect)
    return None, {
        "train": eval_part_MuSeRC(train_path, vect)[0],
        "val": eval_part_MuSeRC(val_path, vect)[0],
        "test": test_score,
        "test_pred": test_pred
    }


def eval_part_MuSeRC(path, vect):
    with jsonlines.open(path) as reader:
        lines = list(reader)
    preds = []
    labels = []
    res = []
    for row in lines:
        pred, lbls, res_ids = get_row_pred_MuSeRC(row, vect)
        preds.extend(pred)
        labels.extend(lbls)
        res.append(res_ids)
    return MuSeRC_metrics(preds, labels), res


def get_row_pred_MuSeRC(row, vect):
    text = vect.transform([row["passage"]["text"]])
    res = []
    labels = []
    res_ids = {"idx": row["idx"], "passage": {"questions": []}}
    for line in row["passage"]["questions"]:
        res_line = {"idx": line["idx"], "answers": []}
        line_answers = []
        line_labels = []
        for answ in line["answers"]:
            line_labels.append(answ.get("label", 0))
            answ = f"{line['question']} {answ['text']}"
            line_answers.append(answ)
        cos = cosine_similarity(text, vect.transform(line_answers))
        pred = cos.argsort()[0][-2:]
        pred = [int(idx in pred) for idx in range(len(line["answers"]))]
        res.append(pred)
        labels.append(line_labels)
        for answ, p in zip(line["answers"], pred):
            res_line["answers"].append({"idx": answ["idx"], "label": p})
        res_ids["passage"]["questions"].append(res_line)
    return res, labels, res_ids
