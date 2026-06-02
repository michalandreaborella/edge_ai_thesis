from ultralytics import YOLO


def main():
    best_model_path = "models/YOLOv8/Tesi_Finale/weights/best.pt"
    model = YOLO(best_model_path)

    print("[*] 1/3: Esportazione Jetson Nano (ONNX Statico)")
    model.export(format="onnx", imgsz=224, dynamic=False, opset=11)

    print("[*] 2/3: Esportazione Intel Movidius (OpenVINO)")
    model.export(format="openvino", imgsz=224)

    print("[*] 3/3: Esportazione Google Coral (Edge TPU TFLite INT8)")
    model.export(format="edgetpu", imgsz=224, int8=True, data="neu-det.yaml")

    print(
        "[+] Finito! Troverai i modelli pronti per Jetson, Coral e Movidius nella cartella: models/YOLOv8/Tesi_Finale/weights/")


if __name__ == "__main__":
    main()