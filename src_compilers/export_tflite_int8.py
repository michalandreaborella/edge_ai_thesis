import tensorflow as tf
import os
import numpy as np
from PIL import Image


def representative_dataset_gen():
    """
    Fornisce un set di immagini campione al convertitore.
    La quantizzazione INT8 HA BISOGNO di dati reali per calibrare le attivazioni
    e calcolare i valori Min/Max per lo scaling dei tensori.
    """
    image_folder = "../data/neu-det/validation/images/"
    # Prendo le prime 50 immagini per la calibrazione
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')][:50]

    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))  # Ridimensionamento richiesto dal target Edge

        # Normalizzazione
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        yield [img_array]


def convert_to_tflite_int8(saved_model_dir, output_tflite_path):
    print(f"[*] Avvio Compilazione TF Lite Full INT8 (Google Coral Target)...")

    # 1. Carico il modello (convertito precedentemente da ONNX a TF SavedModel)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # 2. Configuro le ottimizzazioni di default
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 3. Fornisce il dataset rappresentativo
    converter.representative_dataset = representative_dataset_gen

    # 4. IMPONE il formato INT8 stretto (necessario per Coral Edge TPU)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # 5. Compilazione
    print("[*] Esecuzione calibrazione INT8. Questo processo richiederà alcuni minuti...")
    tflite_quant_model = converter.convert()

    # 6. Salvataggio
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_quant_model)

    print(f"[+] Quantizzazione completata! Modello pronto per Edge TPU: {output_tflite_path}")


if __name__ == "__main__":
    TF_SAVED_MODEL_DIR = "../models/quantized/tf_saved_model"
    OUTPUT_TFLITE = "../models/quantized/faster_rcnn_int8_coral.tflite"

    if os.path.exists(TF_SAVED_MODEL_DIR):
        convert_to_tflite_int8(TF_SAVED_MODEL_DIR, OUTPUT_TFLITE)
    else:
        print(f"[-] Directory SavedModel non trovata: {TF_SAVED_MODEL_DIR}")
        print("[!] Assicurarsi di aver convertito il file .onnx in TF prima di eseguire la quantizzazione.")