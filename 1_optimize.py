import optuna
from ultralytics import YOLO
import torch


def objective(trial):
    lr = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float("momentum", 0.6, 0.95)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data="neu-det.yaml",
        epochs=5,
        imgsz=224,  # 224 è perfetto: è multiplo di 32 per le NPU
        lr0=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        device="mps",  # Forza l'uso dell'acceleratore Mac
        verbose=False,
        project="models/YOLOv8_Optuna",  # Salva in modo ordinato nella tua cartella models
        name=f"trial_{trial.number}"
    )

    return results.box.map50


if __name__ == "__main__":
    print("[*] Avvio Optuna HPO su Mac M1...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("\n[+] Ottimizzazione Completata!")
    print(f"Migliori Iperparametri trovati: {study.best_params}")
    