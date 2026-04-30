import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_onnx_model(input_model_path, output_model_path):
    print("=" * 50)
    print("[*] AVVIO PIPELINE DI QUANTIZZAZIONE DINAMICA INT8")
    print("=" * 50)

    if not os.path.exists(input_model_path):
        print(f"[-] ERRORE: Modello ONNX sorgente non trovato in {input_model_path}")
        return

    print(f"[*] Lettura grafo originale: {input_model_path}")
    print(f"[*] Peso originale su disco: {os.path.getsize(input_model_path) / (1024 * 1024):.2f} MB")

    # La quantizzazione dinamica mappa i pesi FP32 in INT8 (8-bit).

    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8  # Quantizzazione intera a 8 bit
    )

    print(f"[+] Compressione matematica completata con successo!")
    print(f"[+] Modello quantizzato salvato in: {output_model_path}")
    print(f"[+] Nuovo peso su disco: {os.path.getsize(output_model_path) / (1024 * 1024):.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    IN_ONNX = "../models/quantized/faster_rcnn_baseline.onnx"
    OUT_ONNX_INT8 = "../models/quantized/faster_rcnn_int8.onnx"

    quantize_onnx_model(IN_ONNX, OUT_ONNX_INT8)