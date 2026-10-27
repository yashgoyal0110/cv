"""Step 4: measure each model on the test set and save the numbers + confusion matrix images."""
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_recall_fscore_support,
                             roc_auc_score)

from features import CLASSES

SEED = 42


def accuracy_confidence_interval(y_true, y_pred, n_rounds=1000):
    """95% CI for accuracy: resample the test set 1000 times and take the 2.5% / 97.5% points."""
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(n_rounds):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(accuracy_score(y_true[idx], y_pred[idx]))
    return np.percentile(scores, [2.5, 97.5])


def macro_auc(y_true, scores):
    """AUC for each class vs the rest (from SVM decision scores), averaged over 4 classes."""
    return np.mean([roc_auc_score(y_true == k, scores[:, k]) for k in range(len(CLASSES))])


def save_confusion_matrix(cm, name, results_dir):
    ConfusionMatrixDisplay(cm, display_labels=range(len(CLASSES))).plot(cmap="Blues", colorbar=False)
    plt.title(name)
    plt.savefig(os.path.join(results_dir, f"cm_{name.replace(' ', '_')}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_model(name, model, X_test, y_test, results_dir):
    """Predict on the test set, print a report, save the confusion matrix, return all metrics."""
    start = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0)
    _, class_recall, _, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0)
    auc = macro_auc(y_test, model.decision_function(X_test))
    ci = accuracy_confidence_interval(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy {accuracy*100:.2f}% (95% CI {ci[0]*100:.1f}-{ci[1]*100:.1f}) | "
          f"Precision {precision*100:.2f}% | Recall {recall*100:.2f}% | F1 {f1*100:.2f}% | "
          f"Macro AUC {auc:.4f} | test time {test_time:.2f}s")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4, zero_division=0))
    save_confusion_matrix(cm, name, results_dir)

    return {"model": name, "accuracy": accuracy, "acc_ci95": ci.tolist(),
            "precision": precision, "recall": recall, "f1": f1, "macro_auc": auc,
            "per_class_recall": dict(zip(CLASSES, class_recall.tolist())),
            "confusion_matrix": cm.tolist()}


def save_results(results, results_dir, use_clahe, pca_components):
    """Write results.json (everything) and summary.csv (one row per model, in %)."""
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump({"clahe": use_clahe, "pca_components": pca_components,
                   "results": results}, f, indent=2)

    with open(os.path.join(results_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "acc_ci_low", "acc_ci_high", "precision", "recall",
                         "f1", "macro_auc"] + [f"recall_{c}" for c in CLASSES])
        for r in results:
            percents = [r["accuracy"], *r["acc_ci95"], r["precision"], r["recall"], r["f1"]]
            class_recalls = list(r["per_class_recall"].values())
            writer.writerow([r["model"]]
                            + [round(v * 100, 2) for v in percents]
                            + [round(r["macro_auc"], 4)]
                            + [round(v * 100, 2) for v in class_recalls])


def print_summary(results):
    print("\nSummary")
    print(f"{'Model':36s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AUC':>7s}")
    for r in results:
        print(f"{r['model']:36s} {r['accuracy']*100:7.2f} {r['precision']*100:7.2f} "
              f"{r['recall']*100:7.2f} {r['f1']*100:7.2f} {r['macro_auc']:7.4f}")
