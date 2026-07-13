import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# All models import
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ── Data load & split ──────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Scaler ────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Define all models ─────────────────────────────────
models = {
    "logistic_regression": {
        "model":       LogisticRegression(max_iter=10000, random_state=42),
        "name":        "Logistic Regression",
        "description": "Linear model — fast & interpretable. Best for binary classification.",
        "icon":        "📈",
        "color":       "#0d6efd",
        "needs_scale": True,
    },
    "random_forest": {
        "model":       RandomForestClassifier(n_estimators=100, random_state=42),
        "name":        "Random Forest",
        "description": "Combines multiple decision trees. High accuracy, robust model.",
        "icon":        "🌳",
        "color":       "#198754",
        "needs_scale": False,
    },
    "svm": {
        "model":       SVC(kernel="rbf", probability=True, random_state=42),
        "name":        "Support Vector Machine",
        "description": "Uses a hyperplane to classify. Good for complex data.",
        "icon":        "⚡",
        "color":       "#dc3545",
        "needs_scale": True,
    },
    "decision_tree": {
        "model":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "name":        "Decision Tree",
        "description": "Works like a set of if-else rules. Easy to understand & visualize.",
        "icon":        "🌿",
        "color":       "#fd7e14",
        "needs_scale": False,
    },
    "knn": {
        "model":       KNeighborsClassifier(n_neighbors=5),
        "name":        "K-Nearest Neighbors",
        "description": "Decides based on the nearest 5 neighbors. Simple & effective.",
        "icon":        "🔵",
        "color":       "#6f42c1",
        "needs_scale": True,
    },
    "naive_bayes": {
        "model":       GaussianNB(),
        "name":        "Naive Bayes",
        "description": "Probability-based model. Fast & works well with medical data.",
        "icon":        "🧮",
        "color":       "#20c997",
        "needs_scale": False,
    },
}

# ── Train & save each model ────────────────────────────────────
trained = {}
print("Training all models...\n")
for key, info in models.items():
    m = info["model"]
    X_tr = X_train_sc if info["needs_scale"] else X_train
    X_te = X_test_sc  if info["needs_scale"] else X_test

    m.fit(X_tr, y_train)
    acc = accuracy_score(y_test, m.predict(X_te))

    trained[key] = {
        "model":       m,
        "name":        info["name"],
        "description": info["description"],
        "icon":        info["icon"],
        "color":       info["color"],
        "needs_scale": info["needs_scale"],
        "accuracy":    round(acc * 100, 2),
    }
    print(f"  {info['icon']} {info['name']:<30} Accuracy: {acc*100:.2f}%")

# ── Save everything ───────────────────────────────────────────
with open("all_models.pkl", "wb") as f:
    pickle.dump(trained, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

feature_names = list(data.feature_names)
with open("features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\n✅ All models saved to all_models.pkl!")
