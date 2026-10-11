"""
Machine learning classification module for PPG Signal Quality Assessment.
Includes Random Forest, Support Vector Machine (RBF kernel), and Decision Tree classifiers.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score


def train_random_forest_classifier(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 12,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Trains a Random Forest ensemble classifier.
    Minimizes variance and overfitting while providing superior out-of-fold generalization.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    return rf


def train_svm_classifier(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    param_grid: Dict[str, list] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[SVC, Dict[str, Any]]:
    """
    Trains an RBF-kernel Support Vector Machine with GroupKFold cross-validation.
    Ensures non-linear decision boundary optimization without volunteer leakage.
    """
    if param_grid is None:
        param_grid = {
            'C': [1, 5, 10],
            'gamma': ['scale', 0.05, 0.1]
        }

    cv = GroupKFold(n_splits=n_splits)
    base_svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=random_state)
    grid = GridSearchCV(base_svm, param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train_scaled, y_train, groups=groups_train)

    best_svm = grid.best_estimator_
    info = {
        'best_params': grid.best_params_,
        'best_cv_score': float(grid.best_score_)
    }
    return best_svm, info


def train_decision_tree_classifier(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 8,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Trains an interpretable Decision Tree classifier for feature importance analysis."""
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state
    )
    dt.fit(X_train_scaled, y_train)
    return dt


def evaluate_model(model: Any, X_test_scaled: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluates classifier predictions and returns accuracy, macro F1, and detailed report."""
    y_pred = model.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'report': report,
        'predictions': y_pred
    }
