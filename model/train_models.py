import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(MODELS_DIR)
CSV_PATH = os.path.join(ROOT_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")
TEST_CSV_PATH = os.path.join(ROOT_DIR, "test_data.csv")

df = pd.read_csv(CSV_PATH)
print(f"Samples: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Target classes: {sorted(df['NObeyesdad'].unique())}\n")

# encode categorical columns
binary_cols = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
for col in binary_cols:
    df[col] = (df[col] == "yes").astype(int)

df["Gender"] = (df["Gender"] == "Male").astype(int)

freq_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
df["CAEC"] = df["CAEC"].map(freq_map)
df["CALC"] = df["CALC"].map(freq_map)

df = pd.get_dummies(df, columns=["MTRANS"], drop_first=False)

le = LabelEncoder()
y = le.fit_transform(df["NObeyesdad"])
X = df.drop("NObeyesdad", axis=1)
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))

print(f"Features after encoding: {X.shape[1]}")
print(f"Class mapping: {dict(enumerate(le.classes_))}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv(TEST_CSV_PATH, index=False)
print(f"Saved test_data.csv ({len(test_df)} rows)\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# LR and kNN need scaled input; tree-based models and NB work on raw values
NEEDS_SCALING = {"Logistic Regression", "kNN"}

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"),
    "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN":                 KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

results = {}

for name, model in models.items():
    X_tr = X_train_scaled if name in NEEDS_SCALING else X_train.values
    X_te = X_test_scaled if name in NEEDS_SCALING else X_test.values

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
        "F1":        round(f1_score(y_test, y_pred, average="weighted"), 4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }

    safe_name = name.replace(" ", "_")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))
    print(f"[OK] {name}")

results_df = pd.DataFrame(results).T
print("\nResults:")
print(results_df.to_string())
results_df.to_csv(os.path.join(MODELS_DIR, "results.csv"))
print("\nTraining complete\!")
