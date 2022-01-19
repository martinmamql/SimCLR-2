import torch
#https://github.com/martinmamql/lcifr/blob/master/code/metrics/group_fairness.py
# modified according to https://arxiv.org/pdf/2103.06503.pdf

def evaluate(model, model_linear, dataloader):
    y_scores = []
    y_true = []
    for i, (inputs, target) in enumerate(dataloader):
        inputs, target = inputs.cuda(), target.float().cuda()

        feat = model(inputs)
        pred = model_linear(feat).detach()

        y_scores.append(pred[:, 0].data.cpu().numpy())
        y_true.append(target.data.cpu().numpy())

    y_scores = np.concatenate(y_scores)
    y_true = np.concatenate(y_true)
    ap = average_precision_score(y_true, y_scores)
    return ap, np.mean(y_scores)


def confusion_matrix(labels, predictions):
    true_negatives = torch.sum((labels == 0) & (predictions == 0)).float()
    false_positives = torch.sum((labels == 0) & (predictions == 1)).float()
    false_negatives = torch.sum((labels == 1) & (predictions == 0)).float()
    true_positives = torch.sum((labels == 1) & (predictions == 1)).float()

    return true_negatives, false_positives, false_negatives, true_positives


def equality_of_opportunity(labels, predictions, protected):
    protected = protected.bool()

    labels_protected = torch.masked_select(labels, protected)
    labels_unprotected = torch.masked_select(labels, ~protected)
    predictions_protected = torch.masked_select(predictions, protected)
    predictions_unprotected = torch.masked_select(predictions, ~protected)

    tn, fp, fn, tp = confusion_matrix(labels_protected, predictions_protected)
    true_positive_rate_protected = tp / (tp + fn)

    tn, fp, fn, tp = confusion_matrix(labels_unprotected, predictions_unprotected)
    true_positive_rate_unprotected = tp / (tp + fn)

    # return 1 - torch.abs(true_positive_rate_protected - true_positive_rate_unprotected)
    return torch.abs(true_positive_rate_protected - true_positive_rate_unprotected)


def equalized_odds(labels, predictions, protected):
    protected = protected.bool()

    labels_protected = torch.masked_select(labels, protected)
    labels_unprotected = torch.masked_select(labels, ~protected)
    predictions_protected = torch.masked_select(predictions, protected)
    predictions_unprotected = torch.masked_select(predictions, ~protected)

    tn, fp, fn, tp = confusion_matrix(labels_protected, predictions_protected)
    true_positive_rate_protected = tp / (tp + fn)
    false_positive_rate_protected = fp / (fp + tn)

    tn, fp, fn, tp = confusion_matrix(labels_unprotected, predictions_unprotected)
    true_positive_rate_unprotected = tp / (tp + fn)
    false_positive_rate_unprotected = fp / (fp + tn)

    # return 2 - torch.add(
    #     torch.abs(true_positive_rate_protected - true_positive_rate_unprotected),
    #     torch.abs(false_positive_rate_protected - false_positive_rate_unprotected)
    # )
    return torch.add(
        torch.abs(true_positive_rate_protected - true_positive_rate_unprotected),
        torch.abs(false_positive_rate_protected - false_positive_rate_unprotected)
    )


def statistical_parity(predictions, protected):
    predictions, protected = predictions.float(), protected.bool()

    predictions_protected = torch.masked_select(predictions, protected)
    predictions_unprotected = torch.masked_select(predictions, ~protected)

    # return 1 - torch.abs(predictions_protected.mean() - predictions_unprotected.mean())
    return torch.abs(predictions_protected.mean() - predictions_unprotected.mean())

