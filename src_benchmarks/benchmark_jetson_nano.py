import onnxruntime as ort
import time
import psutil
import os
import numpy as np
from PIL import Image
import glob


def measure_system_metrics():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024), psutil.cpu_percent(interval=None)


def run_jetson_benchmark(onnx_path, image_dir):
    print("\n[*] AVVIO BENCHMARK HARDWARE: NVIDIA Jetson Nano")

    # Configura ONNX Runtime per usare TensorRT e CUDA
    # Questo farà compilare il modello per la GPU al primo avvio (potrebbe richiedere 2-3 minuti)
    providers = [
        ('TensorrtExecutionProvider', {'trt_engine_cache_enable': True, 'trt_engine_cache_path': './trt_cache'}),
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]

    print("[*] Caricamento modello sulla GPU NVIDIA...")
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    test_images = glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True)[:1]
    img = Image.open(test_images[0]).convert('RGB').resize((224, 224))
    img_tensor = np.expand_dims(np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0)

    print("[*] Riscaldamento GPU...")
    for _ in range(10):
        session.run(None, {input_name: img_tensor})

    iterations = 50
    start_time = time.time()
    for _ in range(iterations):
        session.run(None, {input_name: img_tensor})
    avg_latency_ms = ((time.time() - start_time) / iterations) * 1000
    fps = 1000 / avg_latency_ms

    print(f"[+] Latenza Jetson Nano : {avg_latency_ms:.2f} ms | Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    ONNX_MODEL = "../models/quantized/faster_rcnn_baseline.onnx"
    run_jetson_benchmark(ONNX_MODEL, "../data/neu-det/validation/images")