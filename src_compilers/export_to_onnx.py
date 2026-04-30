import os
import sys
import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architecture_utils import get_faster_rcnn_model


class FasterRCNNExportWrapper(nn.Module):
    """
    Wrapper semplice per ONNX:
    - input: Tensor [1, 3, H, W]
    - output: boxes, labels, scores
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()

    def forward(self, image: torch.Tensor):
        # Accetta [1, 3, H, W] oppure [3, H, W]
        if image.dim() == 4:
            image = image[0]

        predictions = self.model([image])[0]
        boxes = predictions["boxes"]
        labels = predictions["labels"]
        scores = predictions["scores"]
        return boxes, labels, scores


def export_model_to_onnx(pth_path: str, onnx_path: str, image_size=(224, 224)):
    print("=" * 70)
    print("[*] Export ONNX Faster R-CNN (legacy exporter, fixed image size)")
    print("=" * 70)

    device = torch.device("cpu")

    # 1) Caricamento modello
    model = get_faster_rcnn_model(num_classes=7).to(device)

    print(f"[*] Caricamento pesi da: {pth_path}")
    checkpoint = torch.load(pth_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2) Wrapper ONNX
    wrapper = FasterRCNNExportWrapper(model).to(device).eval()

    # 3) Dummy input: batch fisso = 1, dimensione fissa
    dummy_input = torch.rand(1, 3, image_size[0], image_size[1], device=device)

    # 4) Export ONNX
    print("[*] Avvio export con torch.onnx.export(..., dynamo=False)")
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        onnx_path,
        opset_version=11,
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        dynamo=False,
        dynamic_axes={
            "boxes": {0: "num_detections"},
            "labels": {0: "num_detections"},
            "scores": {0: "num_detections"},
        },
    )

    print("\n[+] Export completato con successo")
    print(f"[+] ONNX salvato in: {onnx_path}")
    print(f"[+] Dimensione file: {os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    PTH_MODEL = "../models/best_model.pth"
    ONNX_OUTPUT = "../models/quantized/faster_rcnn_baseline.onnx"

    os.makedirs(os.path.dirname(ONNX_OUTPUT), exist_ok=True)

    if os.path.exists(PTH_MODEL):
        export_model_to_onnx(PTH_MODEL, ONNX_OUTPUT, image_size=(224, 224))
    else:
        print(f"[-] ERRORE CRITICO: modello non trovato in {PTH_MODEL}")