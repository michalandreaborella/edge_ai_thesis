import time
import os
import psutil
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
import openvino as ov

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_training.train_m1 import NEUDETDataset, collate_fn


def measure_system_metrics():
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_load = psutil.cpu_percent(interval=None)
    return ram_mb, cpu_load


def run_movidius_benchmark(model_xml, data_root, hw_name="Intel_Movidius_NCS2", tdp_w=2.0):
    print(f"\n[*] INIZIALIZZAZIONE BENCHMARK UNIVERSALE (6 METRICHE)")
    print("=" * 60)
    print(f"[+] Target Hardware Rilevato: {hw_name}")

    # 1. Setup OpenVINO
    core = ov.Core()
    if "MYRIAD" not in core.available_devices:
        print("[-] ERRORE: Chiavetta Movidius (MYRIAD) non trovata!")
        return

    print("[*] Compilazione modello su VPU MYRIAD...")
    compiled_model = core.compile_model(model=core.read_model(model_xml), device_name="MYRIAD")
    infer_request = compiled_model.create_infer_request()

    # 2. Setup Dataset
    print("[*] Caricamento Dataset di Validazione NEU-DET...")
    val_dataset = NEUDETDataset(root_dir=data_root, split='val', transforms=T.Compose([T.ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    test_img = val_dataset[0][0].unsqueeze(0).numpy()

    # --- FASE 1: VELOCITÀ E RISORSE ---
    print("\n[*] FASE 1: Misurazione Throughput, Latenza e Overhead")
    for _ in range(10): infer_request.infer([test_img])  # Warmup

    psutil.cpu_percent(interval=1)
    iterations = 50
    start_time = time.time()
    for _ in range(iterations):
        infer_request.infer([test_img])
    end_time = time.time()

    ram_peak_mb, cpu_overhead = measure_system_metrics()
    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_latency_ms
    inf_per_watt = fps / tdp_w

    # --- FASE 2: ACCURATEZZA mAP ---
    print("\n[*] FASE 2: Calcolo Accuratezza (mAP@50) sull'intero val set")
    metric = MeanAveragePrecision(iou_type="bbox")

    for images, targets in tqdm(val_loader, desc="Validazione Movidius"):
        img_np = images[0].unsqueeze(0).numpy()

        results = infer_request.infer([img_np])

        # Estrazione tensori da OpenVINO (Adattare gli indici se l'ordine differisce)
        output_tensors = list(results.values())
        boxes, labels, scores = output_tensors[0], output_tensors[1], output_tensors[2]

        preds = [{"boxes": torch.tensor(boxes), "scores": torch.tensor(scores),
                  "labels": torch.tensor(labels, dtype=torch.int64)}]
        targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
        metric.update(preds, targets_cpu)

    map_50 = metric.compute()['map_50'].item() * 100

    # --- REPORT FINALE ---
    print("\n" + "=" * 60)
    print(" REPORT DECISION FRAMEWORK - MATRICE DELLE METRICHE")
    print("=" * 60)
    print(f" [1] Accuratezza (mAP@50)      : {map_50:.2f} %")
    print(f" [2] Latenza Media (L_max)     : {avg_latency_ms:.2f} ms")
    print(f" [3] Throughput                : {fps:.2f} FPS")
    print(f" [4] RAM Allocata (M_peak)     : ~{ram_peak_mb:.2f} MB")
    print(f" [5] Efficienza Energetica     : {inf_per_watt:.2f} Inf/Watt (Stimato su {tdp_w}W)")
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
            f"{hw_name},FasterRCNN_OpenVINO,FP16/INT8,{map_50:.2f},{avg_latency_ms:.2f},{fps:.2f},{ram_peak_mb:.2f},{inf_per_watt:.2f},{cpu_overhead:.1f}\n")
    print(f"[+] Dati inseriti in: {csv_path}")


if __name__ == "__main__":
    MODEL_PATH = "../models/quantized/openvino/faster_rcnn_baseline.xml"
    DATA_ROOT = "../data/neu-det"
    if not os.path.exists(MODEL_PATH):
        print(f"ERRORE: Modello non trovato in {MODEL_PATH}")
    else:
        run_movidius_benchmark(MODEL_PATH, DATA_ROOT)