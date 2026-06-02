from ultralytics import YOLO


def main():
    print("[*] Avvio Addestramento Finale YOLOv8 Nano...")
    model = YOLO("yolov8n.pt")

    model.train(
        data="neu-det.yaml",
        epochs=100,  # Fai girare 100 epoche (ci metterà un po', lascialo fare)
        imgsz=224,
        device="mps",
        project="models/YOLOv8",
        name="Tesi_Finale",
        # INSERISCI I RISULTATI DI OPTUNA QUI:
        lr0=0.001,  # Cambia questo
        momentum=0.9,  # Cambia questo
        weight_decay=0.0005  # Cambia questo
    )

    print("[+] Finito! Modello in: models/YOLOv8/Tesi_Finale/weights/best.pt")


if __name__ == "__main__":
    main()