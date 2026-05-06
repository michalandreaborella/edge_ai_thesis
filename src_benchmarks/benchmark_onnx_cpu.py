import onnxruntime as ort
import time
import psutil
import os
import numpy as np
from PIL import Image
import glob


def measure_system_metrics():
    """Misura RAM in MB e Carico CPU in %"""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_load = psutil.cpu_percent(interval=None)
    return ram_mb, cpu_load


def run_cpu_benchmark(onnx_path, image_dir, hw_name="Raspberry_Pi_CPU", tdp_w=5.0):
    print(f"\n[*] AVVIO BENCHMARK HARDWARE: {hw_name}")
    print("=" * 60)

    # 1. Setup Motore ONNX
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # 2. Caricamento Immagini di Test
    test_images = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)[:50]
    if not test_images:
        print("[-] ERRORE: Nessuna immagine trovata.")
        return

    # Pre-processamento di una singola immagine per il test
    img = Image.open(test_images[0]).convert('RGB').resize((224, 224))
    img_tensor = np.expand_dims(np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0)

    # 3. Warm-up
    print("[*] Riscaldamento processore...")
    for _ in range(10):
        _ = session.run(None, {input_name: img_tensor})

    # Avvio Monitoraggio Risorse
    psutil.cpu_percent(interval=0.5)

    # 4. Stress Test
    print(f"[*] Esecuzione inferenze in corso...")
    iterations = 50
    start_time = time.time()

    for _ in range(iterations):
        _ = session.run(None, {input_name: img_tensor})

    end_time = time.time()

    # 5. Estrazione Metriche
    ram_peak_mb, cpu_overhead = measure_system_metrics()
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    fps = 1000 / avg_latency_ms
    inf_per_watt = fps / tdp_w

    print("\n" + "=" * 60)
    print(" RISULTATI BENCHMARK (ONNX Runtime)")
    print("=" * 60)
    print(f" Modello               : INT8 ONNX")
    print(f" Latenza Media (L_max) : {avg_latency_ms:.2f} ms")
    print(f" Throughput            : {fps:.2f} FPS")
    print(f" RAM Allocata (M_peak) : ~{ram_peak_mb:.2f} MB")
    print(f" CPU Overhead          : {cpu_overhead:.1f} %")
    print("=" * 60)

    # 6. Salvataggio in CSV
    csv_path = "../results/unified_benchmarks.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a') as f:
        if not file_exists:
            f.write("Hardware,Model,Latency_ms,FPS,RAM_MB,Inf_per_Watt,CPU_Overhead_Pct\n")
        f.write(
            f"{hw_name},FasterRCNN_INT8_ONNX,{avg_latency_ms:.2f},{fps:.2f},{ram_peak_mb:.2f},{inf_per_watt:.2f},{cpu_overhead:.1f}\n")
    print(f"[+] Risultati salvati in: {csv_path}")


if __name__ == "__main__":
    ONNX_MODEL = "../models/quantized/faster_rcnn_int8.onnx"
    DATA_ROOT = "../data/neu-det/validation/images"

    # Variare nome in base alla scheda su cui lo si fa girare: attualmente questo benchmark va bene su schede senza acceleratore
    # Esempio: "Raspberry_Pi_3", "Raspberry_Pi_4", "Raspberry_Pi_5"
    HARDWARE_NAME = "Raspberry_Pi_5_CPU"
    TDP_STIMATO = 5.0  # 3.0 per Pi3, 5.0 per Pi4, 8.0 per Pi5

    run_cpu_benchmark(ONNX_MODEL, DATA_ROOT, HARDWARE_NAME, TDP_STIMATO)