import time
import psutil
import os
import numpy as np
from PIL import Image
import glob
import openvino as ov  # Libreria Intel


def measure_system_metrics():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024), psutil.cpu_percent(interval=None)


def run_movidius_benchmark(model_xml, image_dir):
    print("\n[*] AVVIO BENCHMARK HARDWARE: Intel Movidius NCS2")

    # Inizializza OpenVINO
    core = ov.Core()

    # Controlla se la chiavetta Movidius (MYRIAD) è rilevata
    if "MYRIAD" not in core.available_devices:
        print("[-] ERRORE: Chiavetta Movidius non rilevata! Inseriscila nella porta USB.")
        return

    print("[*] Caricamento e compilazione modello su VPU MYRIAD...")
    model = core.read_model(model=model_xml)
    compiled_model = core.compile_model(model=model, device_name="MYRIAD")
    infer_request = compiled_model.create_infer_request()

    # Pre-processamento immagine
    test_images = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)[:1]
    img = Image.open(test_images[0]).convert('RGB').resize((224, 224))
    img_tensor = np.expand_dims(np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0)

    # Warm-up
    for _ in range(10):
        infer_request.infer([img_tensor])

    # Stress Test
    iterations = 50
    start_time = time.time()
    for _ in range(iterations):
        infer_request.infer([img_tensor])
    avg_latency_ms = ((time.time() - start_time) / iterations) * 1000
    fps = 1000 / avg_latency_ms

    print(f"[+] Latenza Movidius : {avg_latency_ms:.2f} ms | Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    MODEL_XML = "../models/quantized/openvino/faster_rcnn_baseline.xml"
    run_movidius_benchmark(MODEL_XML, "../data/neu-det/validation/images")