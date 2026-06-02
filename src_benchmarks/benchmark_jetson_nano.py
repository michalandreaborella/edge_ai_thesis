import time
import os
import sys
import psutil
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# 1. FIX: Import compatibile con la Jetson (TorchMetrics 0.7.3)
from torchmetrics.detection.map import MAP
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src_training.train_m1 import NEUDETDataset, collate_fn

# 2. FIX: Risoluzione incollata a 200x200 (come l'ONNX Statico) per non esplodere a 5W
FIXED_INPUT_SIZE = (200, 200)

def measure_system_metrics():
    process  = psutil.Process(os.getpid())
    ram_mb   = process.memory_info().rss / (1024 * 1024)
    cpu_load = psutil.cpu_percent(interval=None)
    return ram_mb, cpu_load

def _resize_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    """Ridimensiona a 200x200 e converte per ONNX"""
    h, w = FIXED_INPUT_SIZE
    resized = T.functional.resize(img_tensor, [h, w])
    return resized.unsqueeze(0).numpy()

def run_benchmark(model_path: str, data_root: str,
                  hw_name: str = "Nvidia_Jetson_Nano", tdp_w: float = 5.0): # TDP settato a 5W reali
    is_onnx     = model_path.endswith('.onnx')
    format_name = "ONNX Statico (CUDA)" if is_onnx else "PyTorch Nativo"

    print(f"\n[*] BENCHMARK — {format_name}  |  Hardware: {hw_name} (5 Watt)")
    print("=" * 65)

    print("[1] Caricamento dataset NEU-DET...")
    val_dataset = NEUDETDataset(
        root_dir=data_root, split='val',
        transforms=T.Compose([T.ToTensor()])
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    test_img, _ = val_dataset[0]

    if is_onnx:
        import onnxruntime as ort
        print("[2] Inizializzazione ONNX Runtime (CUDA)...")
        # FIX: Niente provider_options complesse. Usa le variabili d'ambiente di sicurezza che passiamo da terminale.
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        test_input_np = _resize_tensor(test_img)

        def run_once():
            session.run(None, {input_name: test_input_np})

    else:
        print("[2] Inizializzazione modello PyTorch Legacy...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            # FIX: Comando legacy compatibile con PyTorch 1.8
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
            model.load_state_dict(state_dict)
        else:
            model = checkpoint

        model.to(device).eval()

        # Test input per PyTorch deve essere ridimensionato per un confronto equo!
        h, w = FIXED_INPUT_SIZE
        resized_pt = T.functional.resize(test_img, [h, w]).to(device)
        test_input_pt = [resized_pt]

        def run_once():
            with torch.no_grad():
                model(test_input_pt)

    print("\n[3] Warmup (10 iterazioni)...")
    with torch.no_grad():
        for _ in range(10):
            run_once()

    if not is_onnx and torch.cuda.is_available():
        torch.cuda.synchronize()

    iterations = 50
    psutil.cpu_percent(interval=1)

    print(f"[3] Timing di velocità ({iterations} iterazioni)...")
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            run_once()

    if not is_onnx and torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ram_mb, cpu_pct        = measure_system_metrics()
    avg_latency_ms         = ((t1 - t0) / iterations) * 1000
    fps                    = 1000.0 / avg_latency_ms
    inf_per_watt           = fps / tdp_w

    print("\n[4] Calcolo mAP@50 su val set...")
    # FIX: Sintassi corretta per torchmetrics 0.7.3
    metric = MAP()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Inferenza {format_name}"):
            if is_onnx:
                img_np  = _resize_tensor(images[0])
                boxes_np, labels_np, scores_np = session.run(None, {input_name: img_np})
                preds = [{
                    'boxes':  torch.from_numpy(boxes_np),
                    'scores': torch.from_numpy(scores_np),
                    'labels': torch.from_numpy(labels_np).to(torch.int64),
                }]
            else:
                imgs_dev = [T.functional.resize(images[0], [FIXED_INPUT_SIZE[0], FIXED_INPUT_SIZE[1]]).to(device)]
                out      = model(imgs_dev)
                preds    = [{k: v.cpu() for k, v in out[0].items()}]

            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(preds, targets_cpu)

    results = metric.compute()
    map_50  = results['map_50'].item() * 100

    print("\n" + "=" * 65)
    print(f"  RISULTATI  —  {format_name}  ({hw_name})")
    print(f"  mAP@50        : {map_50:.2f} %")
    print(f"  Latenza media : {avg_latency_ms:.2f} ms/img")
    print(f"  Throughput    : {fps:.2f} FPS")
    print(f"  RAM processo  : {ram_mb:.1f} MB")
    print(f"  CPU overhead  : {cpu_pct:.1f} %")
    print(f"  Inf/Watt      : {inf_per_watt:.3f}  (TDP={tdp_w} W)")
    print("=" * 65)


if __name__ == "__main__":
    # Aggiorna questi percorsi in base alla tua cartella se necessario
    DATA_ROOT  = "../data/neu-det"
    MODEL_PATH = "../models/quantized/best_model_shaped.onnx"

    run_benchmark(MODEL_PATH, DATA_ROOT)