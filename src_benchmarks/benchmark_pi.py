import time
import os
import psutil
from ultralytics import YOLO

def measure_system_metrics():
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    cpu_load = psutil.cpu_percent(interval=None)
    return ram_mb, cpu_load

def run_yolo_benchmark(model_path: str, data_yaml: str, hw_name: str, tdp_w: float):
    print(f"\n[*] BENCHMARK — YOLOv8 Edge  |  Hardware: {hw_name} ({tdp_w} Watt)")
    print("=" * 65)

    if not os.path.exists(model_path):
        print(f"[!] ERRORE: Modello {model_path} non trovato.")
        return

    print(f"[1] Inizializzazione motore di inferenza per: {model_path}...")
    model = YOLO(model_path, task='detect')

    test_img_dir = "data/neu-det-yolo/val/images"
    test_img = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])

    print("[2] Warmup (10 iterazioni)...")
    for _ in range(10):
        model.predict(test_img, imgsz=224, verbose=False)

    iterations = 50
    print(f"[3] Timing di inferenza ({iterations} iterazioni)...")
    psutil.cpu_percent(interval=1)

    total_inf_time_ms = 0.0
    for _ in range(iterations):
        results = model.predict(test_img, imgsz=224, verbose=False)
        total_inf_time_ms += results[0].speed['inference']

    ram_mb, cpu_pct = measure_system_metrics()
    avg_latency_ms = total_inf_time_ms / iterations
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0
    inf_per_watt = fps / tdp_w

    print("\n[4] Calcolo mAP@50 sull'intero validation set...")
    metrics = model.val(data=data_yaml, imgsz=224, split='val', verbose=False, plots=False)
    map_50 = metrics.box.map50 * 100

    print("\n" + "=" * 65)
    print(f"  RISULTATI  —  {hw_name}")
    print(f"  mAP@50        : {map_50:.2f} %")
    print(f"  Latenza inf.  : {avg_latency_ms:.2f} ms/img")
    print(f"  Throughput    : {fps:.2f} FPS")
    print(f"  RAM processo  : {ram_mb:.1f} MB")
    print(f"  CPU overhead  : {cpu_pct:.1f} %")
    print(f"  Inf/Watt      : {inf_per_watt:.3f}  (TDP={tdp_w} W)")
    print("=" * 65)

if __name__ == "__main__":
    DATA_YAML = "neu-det.yaml"

    # ===================================================================== #
    # DECOMMENTA SOLO LA RIGA DEL TEST CHE STAI ESEGUENDO
    # ===================================================================== #

    # 1. TEST BASELINE: RASPBERRY PI 4 (SOLO CPU) - TDP ~5.0W
    run_yolo_benchmark(
        model_path="models/YOLOv8/Tesi_Finale/weights/best_saved_model/best_int8.tflite",
        data_yaml=DATA_YAML, hw_name="Raspberry Pi 4 (Base CPU INT8)", tdp_w=5.0)

    # 2. TEST INTEL MOVIDIUS NCS2 - TDP ~6.5W (5W Pi + 1.5W Movidius)
    # run_yolo_benchmark(
    #    model_path="models/YOLOv8/Tesi_Finale/weights/best_openvino_model",
    #    data_yaml=DATA_YAML, hw_name="Raspberry Pi 4 + Intel Movidius", tdp_w=6.5)

    # 3. TEST GOOGLE CORAL TPU - TDP ~7.0W (5W Pi + 2W Coral)
    # run_yolo_benchmark(
    #    model_path="models/YOLOv8/Tesi_Finale/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite",
    #    data_yaml=DATA_YAML, hw_name="Raspberry Pi 4 + Google Coral", tdp_w=7.0)