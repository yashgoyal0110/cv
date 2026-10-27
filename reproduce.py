"""
Reproduction of: Warsito et al., "Lightweight Brain Tumor Classification with HOG
Features and Class-Weighted SVM", JADS 7(1), 2026.

Steps:
  1. Load images -> grayscale -> 16x16 -> HOG features        (features.py)
  2. PCA keeping 98% of variance, fitted on training data only
  3. Train the SVMs: grid search + the 6 models of Table 2      (models.py)
  4. Evaluate on the test set, save metrics + confusion matrices (evaluate.py)

Usage:
    python reproduce.py --data data                  # baseline reproduction
    python reproduce.py --data data --skip-grid      # same, skip the slow grid search
    python reproduce.py --data data --clahe          # our extension: CLAHE before resizing
"""
import argparse
import os
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA

from evaluate import evaluate_model, print_summary, save_results
from features import load_dataset
from models import paper_models, run_grid_search


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data", help="folder containing Training/ and Testing/")
    parser.add_argument("--clahe", action="store_true", help="apply CLAHE before resizing")
    parser.add_argument("--skip-grid", action="store_true", help="skip the grid search")
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = "clahe" if args.clahe else "baseline"
    results_dir = f"results_{run_name}"  # numbers + images
    models_dir = f"models_{run_name}"    # trained PCA + SVMs
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Step 1: images -> HOG features
    X_train, y_train = load_dataset(os.path.join(args.data, "Training"), args.clahe)
    X_test, y_test = load_dataset(os.path.join(args.data, "Testing"), args.clahe)
    print(f"Train {X_train.shape}, Test {X_test.shape}")
    print("Train counts:", np.bincount(y_train), "Test counts:", np.bincount(y_test))

    # Step 2: PCA (fit on train only, so the test set never leaks into training)
    pca = PCA(0.98)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f"PCA components kept: {pca.n_components_}")
    joblib.dump(pca, os.path.join(models_dir, "pca.joblib"))

    # Step 3: train the models
    trained = {}
    if not args.skip_grid:
        trained["SVM GridSearch Best"] = run_grid_search(X_train, y_train)
    for name, model in paper_models().items():
        start = time.time()
        model.fit(X_train, y_train)
        print(f"[{name}] train time {time.time() - start:.2f}s")
        trained[name] = model

    # Step 4: evaluate and save
    results = []
    for name, model in trained.items():
        joblib.dump(model, os.path.join(models_dir, f"{name.replace(' ', '_')}.joblib"))
        results.append(evaluate_model(name, model, X_test, y_test, results_dir))

    save_results(results, results_dir, args.clahe, int(pca.n_components_))
    print_summary(results)


if __name__ == "__main__":
    main()
