import time
import os
import psutil
import torch
from PIL import Image
import torchvision.transforms as T
import sys
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Importa l'architettura dal modulo esterno
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architecture_utils import get_faster_rcnn_model
from src_training.train_m1 import NEUDETDataset, \
    collate_fn  # Assicurati che i nomi corrispondano al tuo file di training
from torch.utils.data import DataLoader


def measure_system_metrics():
    """
    Raccoglie RAM dinamica (MB) e Carico CPU (%)
    Questo codice va bene per i test base per dimostrare quanto male
    si presta se il modello non è ottimizzato pe rlo specifico HW
    """
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_load = psutil.cpu_percent(interval=None)
    return ram_mb, cpu_load


def run_comprehensive_benchmark(model_path, data_root, estimated_tdp_w=15.0):
    print("\n[*] INIZIALIZZAZIONE BENCHMARK UNIVERSALE (6 METRICHE)")
    print("=" * 60)

    # 1. Configurazione Hardware
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        hw_name = "Apple_M1_MPS"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        hw_name = "Nvidia_CUDA"
    else:
        device = torch.device("cpu")
        hw_name = "CPU_Standard"

    print(f"[+] Target Hardware Rilevato: {hw_name}")

    # 2. Caricamento Modello
    model = get_faster_rcnn_model(num_classes=7)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 3. Setup Dataset Validation (Per Accuratezza mAP)
    print("[*] Caricamento Dataset di Validazione NEU-DET...")
    val_dataset = NEUDETDataset(root_dir=data_root, split='validation', transforms=T.Compose([T.ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # --- INIZIO TEST VELOCITA E RISORSE (SPEED & RESOURCE TEST) ---
    print("\n[*] FASE 1: Misurazione Throughput, Latenza e Overhead")
    # Prendiamo una singola immagine di test per isolare la latenza di inferenza
    test_img, _ = val_dataset[0]
    test_img = test_img.unsqueeze(0).to(device)

    # Riscaldamento
    with torch.no_grad():
        for _ in range(10): _ = model(test_img)

    # Avvia monitoraggio CPU
    psutil.cpu_percent(interval=1)

    iterations = 50
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(test_img)
            if device.type == 'mps': torch.mps.synchronize()
            if device.type == 'cuda': torch.cuda.synchronize()

    end_time = time.time()

    # Calcolo Metriche Hardware
    ram_peak_mb, cpu_overhead = measure_system_metrics()
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    fps = 1000 / avg_latency_ms
    inf_per_watt = fps / estimated_tdp_w

    # --- INIZIO TEST ACCURATEZZA (mAP TEST) ---
    print("\n[*] FASE 2: Calcolo Accuratezza (mAP@50) sull'intero validation set")
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validazione"):
            images = list(img.to(device) for img in images)
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

            predictions = model(images)
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in predictions]
            metric.update(preds_cpu, targets_cpu)

    metrics_result = metric.compute()
    map_50 = metrics_result['map_50'].item() * 100  # In percentuale

    # --- REPORT FINALE ---
    print("\n" + "=" * 60)
    print(" REPORT DECISION FRAMEWORK - MATRICE DELLE METRICHE")
    print("=" * 60)
    print(f" [1] Accuratezza (mAP@50)      : {map_50:.2f} %")
    print(f" [2] Latenza Media (L_max)     : {avg_latency_ms:.2f} ms")
    print(f" [3] Throughput                : {fps:.2f} FPS")
    print(f" [4] RAM Allocata (M_peak)     : ~{ram_peak_mb:.2f} MB")
    print(f" [5] Efficienza Energetica     : {inf_per_watt:.2f} Inf/Watt (Stimato su {estimated_tdp_w}W)")
    print(f" [6] CPU/NPU Overhead          : {cpu_overhead:.1f} %")
    print("=" * 60)

    # Salvataggio in CSV
    csv_path = "../results/unified_benchmarks.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a') as f:
        if not file_exists:
            f.write("Hardware,Model,Precision,mAP_50,Latency_ms,FPS,RAM_MB,Inf_per_Watt,CPU_Overhead_Pct\n")
        f.write(
            f"{hw_name},FasterRCNN_RN50,FP32,{map_50:.2f},{avg_latency_ms:.2f},{fps:.2f},{ram_peak_mb:.2f},{inf_per_watt:.2f},{cpu_overhead:.1f}\n")
    print(f"[+] Dati inseriti in: {csv_path}")


if __name__ == "__main__":
    MODEL_PATH = "../models/best_model.pth"
    DATA_ROOT = "../data/neu-det"  # Percorso della cartella dataset

    if not os.path.exists(MODEL_PATH):
        print(f"ERRORE CRITICO: Pesi non trovati in {MODEL_PATH}")
    else:
        run_comprehensive_benchmark(MODEL_PATH, DATA_ROOT)