import torch
import os
import sys

# Importa la tua architettura
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architecture_utils import get_faster_rcnn_model


def re_export_model():
    print("[*] Avvio esportazione PyTorch -> ONNX (Strict Mode)...")

    # 1. Caricamento pesi originali
    model_path = "../models/best_model.pth"
    device = torch.device('cpu')
    model = get_faster_rcnn_model(num_classes=7)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. Creazione tensore fittizio per tracciare il grafo
    # Fissiamo la dimensione a 224x224 come richiesto dai tuoi benchmark
    dummy_input = torch.randn(1, 3, 200, 200)
    output_path = "../models/quantized/faster_rcnn_baseline_fixed.onnx"

    # 3. Esportazione con regole rigide per Jetson/TensorRT
    print("[*] Tracciamento del grafo in corso (potrebbe richiedere un minuto)...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # <--- LA CHIAVE DI TUTTO: Supporto nativo per NMS e If/Else
        do_constant_folding=True,  # Ottimizza la matematica statica
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'batch_size', 1: 'num_boxes'},
            'labels': {0: 'batch_size', 1: 'num_boxes'},
            'scores': {0: 'batch_size', 1: 'num_boxes'}
        }
    )

    print(f"[+] Modello ONNX rigenerato con successo: {output_path}")


if __name__ == "__main__":
    re_export_model()