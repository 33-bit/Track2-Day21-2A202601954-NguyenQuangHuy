import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

EVAL_THRESHOLD = 0.68

# Bonus 1: Neu co bien moi truong MLFLOW_TRACKING_URI (vi du DagsHub),
# dung tracking server tu xa thay vi file cuc bo.
# Vi du DagsHub:
#   MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
#   MLFLOW_TRACKING_USERNAME=<user>
#   MLFLOW_TRACKING_PASSWORD=<token>
# Neu khong set, dung tracking URI cuc bo (relative) de MLflow khong ghi
# duong dan tuyet doi vao mlruns/. Neu khong, artifact_location se bi cung
# hoa thanh path may local (vd /Users/...) va gay loi Permission denied
# khi chay tren CI Linux.
mlflow.set_tracking_uri(
    os.environ.get("MLFLOW_TRACKING_URI", f"file://{os.getcwd()}/mlruns")
)


def _make_model(model_type: str, params: dict):
    """Chon thuật toán theo model_type. Params khong hop le se duoc loai bo."""
    if model_type == "gradient_boosting":
        valid = {
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "max_features", "learning_rate",
        }
        return GradientBoostingClassifier(
            random_state=42, **{k: v for k, v in params.items() if k in valid}
        )
    if model_type == "logistic_regression":
        valid = {"C", "max_iter", "solver", "penalty"}
        return LogisticRegression(
            random_state=42, **{k: v for k, v in params.items() if k in valid}
        )
    # Mac dinh: random_forest
    valid = {
        "n_estimators", "max_depth", "min_samples_split",
        "min_samples_leaf", "max_features", "class_weight",
    }
    return RandomForestClassifier(
        random_state=42, **{k: v for k, v in params.items() if k in valid}
    )


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho mo hinh.
                     Co the chua "model_type" de chon thuật toán.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """
    model_type = params.get("model_type", "random_forest")
    model_params = {k: v for k, v in params.items() if k != "model_type"}

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: Kiem tra phan phoi nhan cua tap huan luyen
    label_dist = y_train.value_counts(normalize=True).sort_index().to_dict()
    for label, ratio in sorted(label_dist.items()):
        if ratio < 0.10:
            print(
                f"[DATA DRIFT WARNING] Lop {label} chi chiem {ratio:.2%} "
                f"(< 10%). Tap du lieu lech lac, model co the biased."
            )

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen mo hinh
        model = _make_model(model_type, model_params)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "model_type": model_type,
        }
        # Bonus 5: Ghi phan phoi nhan vao metrics.json
        for label in sorted(y_train.unique()):
            metrics[f"label_dist_{label}"] = label_dist.get(label, 0.0)
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f)

        # Bonus 3: Tao bao cao hieu suat dang van ban
        _write_report(
            y_eval, preds, acc, f1, label_dist, model_type,
        )

        # TODO 9: Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len GCS o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


def _write_report(y_true, preds, acc, f1, label_dist, model_type):
    """Bonus 3: Ghi confusion matrix + precision/recall tung lop ra report.txt."""
    labels = sorted(set(y_true) | set(preds))
    cm = confusion_matrix(y_true, preds, labels=labels)
    prec = precision_score(y_true, preds, labels=labels, average=None, zero_division=0)
    rec  = recall_score(y_true, preds, labels=labels, average=None, zero_division=0)

    lines = []
    lines.append("=== Model Performance Report ===")
    lines.append(f"model_type: {model_type}")
    lines.append(f"accuracy : {acc:.4f}")
    lines.append(f"f1_score : {f1:.4f}")
    lines.append("")
    lines.append("Confusion matrix (rows=actual, cols=predicted):")
    header = "        " + "".join(f"  pred_{l}" for l in labels)
    lines.append(header)
    for i, l in enumerate(labels):
        lines.append(f"actual_{l}: " + "".join(f"{cm[i][j]:>10d}" for j in range(len(labels))))
    lines.append("")
    lines.append("Per-class precision / recall:")
    for i, l in enumerate(labels):
        lines.append(f"  class {l}: precision={prec[i]:.4f}, recall={rec[i]:.4f}")
    lines.append("")
    lines.append("Label distribution (train):")
    for l in sorted(label_dist):
        lines.append(f"  class {l}: {label_dist[l]:.4f}")
    lines.append("")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/report.txt", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
