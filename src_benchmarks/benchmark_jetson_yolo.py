import time
import os
import psutil
import numpy as np
import onnxruntime as ort
from PIL import Image

# Configurazione
MODEL_PATH = "best.onnx"
TEST_IMAGE_PATH = "immagine_di_test.jpg"  # Metti una foto a caso del tuo dataset
TDP_W = 5.0
ITERATIONS = 50


def measure_system_metrics():
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_load = psutil.cpu_percent(interval=None)
    return ram_mb, cpu_load


def preprocess_image(image_path):
    """Replica il preprocessing di YOLOv8 in puro NumPy"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)

    # Converte in array, normalizza [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0

    # Da HWC (Altezza, Larghezza, Canali) a CHW (Canali, Altezza, Larghezza)
    img_np = np.transpose(img_np, (2, 0, 1))

    # Aggiunge la dimensione del batch [1, 3, 224, 224]
    img_np = np.expand_dims(img_np, axis=0)
    return img_np


def main():
    print(f"\n[*] BENCHMARK — ONNX Statico (CUDA) | Hardware: Nvidia Jetson Nano ({TDP_W} Watt)")
    print("=" * 65)

    print("[1] Caricamento e preprocessing immagine...")
    input_tensor = preprocess_image(TEST_IMAGE_PATH)

    print("[2] Inizializzazione ONNX Runtime (CUDA Execution Provider)...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name

    print("[3] Warmup (Riscaldamento GPU - 10 iterazioni)...")
    for _ in range(10):
        session.run(None, {input_name: input_tensor})

    print(f"[4] Benchmark di Velocità ({ITERATIONS} iterazioni)...")
    psutil.cpu_percent(interval=1)  # Reset sensore CPU

    t0 = time.perf_counter()
    for _ in range(ITERATIONS):
        session.run(None, {input_name: input_tensor})
    t1 = time.perf_counter()

    # Rilevamento metriche fisiche
    ram_mb, cpu_pct = measure_system_metrics()

    avg_latency_ms = ((t1 - t0) / ITERATIONS) * 1000
    fps = 1000.0 / avg_latency_ms
    inf_per_watt = fps / TDP_W

    print("\n" + "=" * 65)
    print(f"  RISULTATI DI PERFORMANCE FISICA")
    print(f"  Modello       : YOLOv8 Nano (224x224)")
    print(f"  Latenza media : {avg_latency_ms:.2f} ms/img")
    print(f"  Throughput    : {fps:.2f} FPS")
    print(f"  RAM processo  : {ram_mb:.1f} MB")
    print(f"  CPU overhead  : {cpu_pct:.1f} %")
    print(f"  Inf/Watt      : {inf_per_watt:.3f}  (TDP={TDP_W} W)")
    print("=" * 65)


if __name__ == "__main__":
    main()