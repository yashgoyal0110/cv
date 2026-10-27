"""Step 3: the SVM models from the paper (Table 1 grid search + Table 2 models)."""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Table 1: hyperparameter search space
PARAM_GRID = [
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.0001]},
    {"kernel": ["poly"], "C": [0.1, 1, 10, 100], "degree": [2, 3, 4],
     "gamma": ["scale", 0.001, 0.0001]},
]

# Sec. 3.5: class weights as written in the paper (glioma, meningioma, no_tumor, pituitary)
PAPER_CLASS_WEIGHTS = {0: 0.69, 1: 0.69, 2: 0.90, 3: 0.60}


def run_grid_search(X_train, y_train):
    """5-fold cross-validation over PARAM_GRID, scored by accuracy. Returns the best SVM."""
    search = GridSearchCV(SVC(), PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print(f"GridSearch best: {search.best_params_}, CV acc {search.best_score_:.4f}")
    return search.best_estimator_


def paper_models():
    """The six SVMs compared in Table 2 (not trained yet)."""
    return {
        "SVM Custom Class Weight (Poly)": SVC(kernel="poly", degree=3, C=10,
                                              class_weight=PAPER_CLASS_WEIGHTS),
        "SVM Best Hyperparameters (Poly)": SVC(kernel="poly", degree=4, C=10),
        "SVM Polynomial": SVC(kernel="poly"),
        "SVM RBF": SVC(kernel="rbf"),
        "SVM Linear": SVC(kernel="linear"),
        "SVM Sigmoid": SVC(kernel="sigmoid"),
    }
