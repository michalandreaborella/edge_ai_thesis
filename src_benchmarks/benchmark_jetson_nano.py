import time
import os
import psutil
import torch
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_training.train_m1 import NEUDETDataset, collate_fn


def measure_system_metrics():
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_load = psutil.cpu_percent(interval=None)
    return ram_mb, cpu_load


def run_benchmark(model_path, data_root, hw_name="Nvidia_Jetson_Nano", tdp_w=10.0):
    is_onnx = model_path.endswith('.onnx')
    format_name = "ONNX (CUDA)" if is_onnx else "PyTorch Nativo"

    print(f"\n[*] INIZIALIZZAZIONE BENCHMARK - {format_name}")
    print("=" * 60)

    # 1. Caricamento Dataset
    print("[*] Caricamento Dataset NEU-DET...")
    val_dataset = NEUDETDataset(root_dir=data_root, split='validation', transforms=T.Compose([T.ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_img, _ = val_dataset[0]

    # 2. Inizializzazione Modello
    if is_onnx:
        import onnxruntime as ort
        print("[*] Avvio Motore ONNX con CUDAExecutionProvider...")
        # Limito la memoria CUDA per evitare esplosioni RAM
        provider_options = [
            {'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo', 'gpu_mem_limit': 2 * 1024 * 1024 * 1024}]
        session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                       provider_options=provider_options)
        input_name = session.get_inputs()[0].name
        test_input = np.expand_dims(test_img.numpy(), axis=0)
    else:
        print("[*] Avvio Modello PyTorch...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load(model_path, map_location=device)
        model.eval()
        test_input = [test_img.to(device)]

    # --- FASE 1: VELOCITÀ ---
    print("\n[*] FASE 1: Warmup e Velocità...")
    iterations = 50
    with torch.no_grad():
        for _ in range(5):  # Warmup ridotto a 5 per salvare memoria
            if is_onnx:
                session.run(None, {input_name: test_input})
            else:
                model(test_input)

    psutil.cpu_percent(interval=1)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            if is_onnx:
                session.run(None, {input_name: test_input})
            else:
                model(test_input)
    end_time = time.time()

    ram_peak_mb, cpu_overhead = measure_system_metrics()
    avg_latency_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_latency_ms
    inf_per_watt = fps / tdp_w

    # --- FASE 2: ACCURATEZZA ---
    print("\n[*] FASE 2: Calcolo mAP@50...")
    metric = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Validazione {format_name}"):
            if is_onnx:
                img_np = images[0].unsqueeze(0).numpy()
                boxes, labels, scores = session.run(None, {input_name: img_np})
                preds = [{"boxes": torch.tensor(boxes), "scores": torch.tensor(scores),
                          "labels": torch.tensor(labels, dtype=torch.int64)}]
            else:
                images_cuda = [img.to(device) for img in images]
                out = model(images_cuda)
                preds = [{"boxes": out[0]['boxes'].cpu(), "scores": out[0]['scores'].cpu(),
                          "labels": out[0]['labels'].cpu()}]

            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds, targets_cpu)

    map_50 = metric.compute()['map_50'].item() * 100

    # --- REPORT FINALE ---
    print("\n" + "=" * 60)
    print(f" RISULTATI: {format_name}")
    print(f" [1] Accuratezza (mAP@50): {map_50:.2f} %")
    print(f" [2] Latenza Media       : {avg_latency_ms:.2f} ms")
    print(f" [3] Throughput          : {fps:.2f} FPS")
    print(f" [4] RAM Allocata        : ~{ram_peak_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    DATA_ROOT = "../data/neu-det"

    # Scelta:
    MODEL_PATH = "../models/quantized/faster_rcnn_baseline_shaped.onnx"
    # MODEL_PATH = "../models/baseline/faster_rcnn_baseline.pth"

    run_benchmark(MODEL_PATH, DATA_ROOT)