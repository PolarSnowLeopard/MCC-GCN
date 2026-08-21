import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(labels, preds, num_classes=4):
    """Per-class accuracy, macro accuracy, and overall accuracy."""
    class_labels = list(range(num_classes))
    cm = confusion_matrix(labels, preds, labels=class_labels)

    per_class_acc = np.zeros(num_classes)
    for i in range(num_classes):
        total = np.sum(cm[i, :])
        per_class_acc[i] = cm[i, i] / total if total > 0 else 0

    macro_acc = np.mean(per_class_acc)
    overall_acc = accuracy_score(labels, preds)
    return per_class_acc, macro_acc, overall_acc


def calculate_detailed_metrics(labels, preds, num_classes=4):
    """Per-class and macro precision, recall, F1."""
    class_labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    return {
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }


def calculate_chance_baselines(labels, num_classes=4):
    """Return prevalence-weighted and majority-class chance accuracies."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a non-empty one-dimensional array")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("labels fall outside the configured classes")
    counts = np.bincount(labels, minlength=num_classes)
    prevalence = counts / counts.sum()
    return {
        "sample_count": int(counts.sum()),
        "class_counts": counts,
        "class_prevalence": prevalence,
        "prevalence_weighted_expected_accuracy": float(np.square(prevalence).sum()),
        "majority_class_accuracy": float(prevalence.max()),
    }
