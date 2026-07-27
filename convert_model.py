import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import onnx
from onnx import shape_inference

# L'IMPORT SPOSTATO QUI IN CIMA (Così Python non impazzisce)
import torch.onnx.utils

# HACK 1: Bendiamo Torchvision per fargli credere di essere già in modalità "Trace"
torchvision._is_tracing = lambda: True


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model([images[0]])
        return out[0]['boxes'], out[0]['labels'], out[0]['scores']


def get_model(num_classes=7):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    pth_path = "models/best_model.pth"
    output_dir = "models/quantized"
    os.makedirs(output_dir, exist_ok=True)

    onnx_raw_path = os.path.join(output_dir, "best_model_raw.onnx")
    onnx_shaped_path = os.path.join(output_dir, "best_model_shaped.onnx")

    print("[*] Inizio conversione modello (Mac)...")
    device = torch.device('cpu')

    print(f"[-] Caricamento pesi da {pth_path}...")
    checkpoint = torch.load(pth_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint.state_dict()

    print("[-] Adattamento pesi per compatibilità Jetson...")
    fixed_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if "backbone.fpn.inner_blocks" in k or "backbone.fpn.layer_blocks" in k:
            new_key = k.replace(".0.weight", ".weight").replace(".0.bias", ".bias")
        elif "rpn.head.conv" in k:
            new_key = k.replace(".0.0.weight", ".weight").replace(".0.0.bias", ".bias")
        fixed_state_dict[new_key] = v

    print("[-] Costruzione architettura Faster R-CNN...")
    model = get_model(num_classes=7)
    model.load_state_dict(fixed_state_dict, strict=True)
    model.eval()

    print("[-] Incapsulamento nel Wrapper...")
    wrapped_model = ONNXWrapper(model)
    wrapped_model.eval()

    dummy_input = torch.randn(1, 3, 200, 200, device=device)

    print("[-] Esportazione ONNX (Utilizzo forzato del Modulo Legacy Interno)...")

    # Usiamo direttamente la via di fuga
    torch.onnx.utils.export(
        wrapped_model,
        dummy_input,
        onnx_raw_path,
        opset_version=11,
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores']
    )

    print("[-] Calcolo shape inference (Bypass per TensorRT)...")
    inferred = shape_inference.infer_shapes(onnx.load(onnx_raw_path))
    onnx.save(inferred, onnx_shaped_path)

    if os.path.exists(onnx_raw_path):
        os.remove(onnx_raw_path)

    print(f"[+] FINITO! Modello ONNX STATICO perfetto salvato in: {onnx_shaped_path}")


if __name__ == "__main__":
    main()