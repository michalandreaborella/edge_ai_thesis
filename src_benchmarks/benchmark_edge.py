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

    # 1. Caricamento Modello Universale
    # Ultralytics rileva in automatico il motore giusto (ONNX, OpenVINO, PyCoral) dall'estensione del file
    print(f"[1] Inizializzazione motore di inferenza per: {model_path}...")
    model = YOLO(model_path, task='detect')

    # Peschiamo la prima immagine di test reale dal validation set
    test_img_dir = "../data/neu-det-yolo/val/images"
    test_img = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])

    # 2. Warmup (Riscaldamento dell'acceleratore hardware)
    print("[2] Warmup (10 iterazioni)...")
    for _ in range(10):
        model.predict(test_img, imgsz=224, verbose=False)

    # 3. Timing puro
    iterations = 50
    print(f"[3] Timing di inferenza ({iterations} iterazioni)...")
    psutil.cpu_percent(interval=1) # Reset del sensore CPU

    total_inf_time_ms = 0.0
    for _ in range(iterations):
        results = model.predict(test_img, imgsz=224, verbose=False)
        # CHICCA PER LA TESI: YOLO calcola i tempi separando preprocess, inferenza pura e postprocess.
        # Noi estraiamo SOLO il tempo di inferenza hardware, che è il dato ingegneristico reale.
        total_inf_time_ms += results[0].speed['inference']

    ram_mb, cpu_pct = measure_system_metrics()

    avg_latency_ms = total_inf_time_ms / iterations
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0
    inf_per_watt = fps / tdp_w

    # 4. Calcolo mAP@50 Ufficiale
    print("\n[4] Calcolo mAP@50 sull'intero validation set...")
    # Questo comando valuta tutte le 360 immagini usando la metrica COCO standard
    metrics = model.val(data=data_yaml, imgsz=224, split='val', verbose=False)
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
    # Il percorso del file yaml
    DATA_YAML = "../neu-det.yaml"

    # ===================================================================== #
    # COMMENTA/DECOMMENTA LA RIGA CORRISPONDENTE ALLA SCHEDA CHE STAI USANDO
    # ===================================================================== #

    # 1. NVIDIA JETSON NANO (Motore: ONNXRuntime con CUDA)
    run_yolo_benchmark(
        model_path="../models/YOLOv8/Tesi_Finale/weights/best.onnx",
        data_yaml=DATA_YAML, hw_name="Nvidia Jetson Nano", tdp_w=5.0)

    # 2. RASPBERRY PI 4 - SOLO CPU (Motore: TFLite CPU o ONNX CPU)
    # run_yolo_benchmark(
    #    model_path="../models/YOLOv8/Tesi_Finale/weights/best_saved_model/best_int8.tflite",
    #    data_yaml=DATA_YAML, hw_name="Raspberry Pi 4 (Base CPU)", tdp_w=5.0)

    # 3. RASPBERRY PI 4 + GOOGLE CORAL TPU (Motore: TFLite EdgeTPU Delegate)
    # run_yolo_benchmark(
    #    model_path="../models/YOLOv8/Tesi_Finale/weights/best_saved_model/best_full_integer_quant_edgetpu.tflite",
    #    data_yaml=DATA_YAML, hw_name="Raspberry Pi 4 + Google Coral", tdp_w=7.0)

    # 4. RASPBERRY PI 4 + INTEL MOVIDIUS NCS2 (Motore: OpenVINO)
    # run_yolo_benchmark(
    #    model_path="../models/YOLOv8/Tesi_Finale/weights/best_openvino_model/",
    #    data_yaml=DATA_YAML, hw_name="Raspberry Pi 4 + Intel Movidius", tdp_w=6.5)