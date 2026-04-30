import time
import os
import psutil
import torch
from PIL import Image
import torchvision.transforms as T
import sys

# Aggiungiamo la root al path per poter importare la cartella models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architecture_utils import get_faster_rcnn_model


def measure_memory_footprint():
    """Restituisce la memoria RAM dinamica residente (RSS) in Megabyte."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_baseline_benchmark(model_path, image_path, iterations=50):
    print(f"[*] Avvio Profilazione Baseline su Architettura Apple Silicon (M1/M2)")

    # 1. Configurazione Hardware (MPS per Mac)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[+] Hardware Acceleratore: Apple Neural Engine / GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("[-] Attenzione: MPS non disponibile. Uso CPU standard.")

    mem_baseline = measure_memory_footprint()

    # 2. Caricamento Modello
    print(f"[*] Caricamento pesi da: {model_path}")
    model = get_faster_rcnn_model(num_classes=7)

    # Caricamento sicuro del checkpoint (gestisce il formato che hai usato nel training)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # Fondamentale: disattiva dropout e batchnorm per inferenza
    mem_loaded = measure_memory_footprint()

    # 3. Preparazione Input (Tensore)
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)  # Aggiunge la dimensione del batch [1, C, H, W]

    # 4. Warm-up (Riscaldamento Cache)
    print("[*] Esecuzione Warm-up (10 iterazioni) per stabilizzare cache L2/L3 e JIT...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(img_tensor)

    # 5. Benchmark Reale
    print(f"[*] Campionamento Inferenze ({iterations} iterazioni)...")
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(img_tensor)
            # Sincronizzazione GPU M1 per avere il tempo ESATTO
            if device.type == 'mps':
                torch.mps.synchronize()

    end_time = time.time()
    mem_peak = measure_memory_footprint()

    # 6. Derivazione Metriche del Framework
    total_time = end_time - start_time
    avg_latency_ms = (total_time / iterations) * 1000
    throughput_fps = 1000 / avg_latency_ms
    dynamic_ram_mb = mem_peak - mem_baseline

    # 7. Output Accademico
    print("\n" + "=" * 55)
    print(" RISULTATI BENCHMARK FASE 4 - Baseline Edge (Mac M1)")
    print("=" * 55)
    print(f" Modello                : Faster R-CNN ResNet50-FPN")
    print(f" Precisione Numerica    : FP32")
    print(f" Latenza Media          : {avg_latency_ms:.2f} ms")
    print(f" Throughput Teorico     : {throughput_fps:.2f} FPS")
    print(f" RAM Allocata (M_peak)  : ~{dynamic_ram_mb:.2f} MB")
    print(f" Storage Modello Statico: {(os.path.getsize(model_path) / (1024 * 1024)):.2f} MB")
    print("=" * 55)

    # 8. Scrittura automatica su file CSV per la Fase 5
    csv_path = "../results/hardware_benchmarks.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a') as f:
        if not file_exists:
            f.write("Hardware,Model,Format,Latency_ms,FPS,RAM_MB\n")
        f.write(
            f"Apple_M1_MPS,FasterRCNN_ResNet50,FP32,{avg_latency_ms:.2f},{throughput_fps:.2f},{dynamic_ram_mb:.2f}\n")
    print(f"[*] Metriche salvate per il Decision Framework in: {csv_path}")


if __name__ == "__main__":
    import glob

    MODEL_PATH = "../models/best_model.pth"
    IMAGE_DIR = "../data/neu-det/validation/images"

    # Cerca dinamicamente la prima immagine .jpg disponibile, anche nelle sottocartelle
    available_images = glob.glob(os.path.join(IMAGE_DIR, "**", "*.jpg"), recursive=True)

    if not os.path.exists(MODEL_PATH):
        print(f"ERRORE CRITICO: I pesi del modello non si trovano in {MODEL_PATH}")
    elif len(available_images) == 0:
        print(
            f"ERRORE CRITICO: Nessuna immagine trovata in {IMAGE_DIR}. Controlla di aver estratto i file correttamente.")
    else:
        # Prende automaticamente la prima immagine che trova
        TEST_IMAGE = available_images[0]
        print(f"[*] Immagine di test rilevata automaticamente: {os.path.basename(TEST_IMAGE)}")
        run_baseline_benchmark(MODEL_PATH, TEST_IMAGE)