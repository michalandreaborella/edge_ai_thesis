import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

DATA_ROOT = 'data/neu-det'
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x_center * dw, y_center * dh, w * dw, h * dh)


def process_split(split):
    print(f"[*] Conversione split: {split}")
    ann_dir = os.path.join(DATA_ROOT, split, 'annotations')
    img_dir = os.path.join(DATA_ROOT, split, 'images')
    labels_root = os.path.join(DATA_ROOT, split, 'labels')
    os.makedirs(labels_root, exist_ok=True)

    xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()

        filename = root.find('filename').text
        img_path = None
        detected_cat = None

        # Troviamo in quale sottocartella di classe si trova l'immagine originaria
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            p = os.path.join(img_dir, filename + ext)
            if os.path.exists(p):
                img_path = p
                break
            for cat in CLASSES:
                p_cat = os.path.join(img_dir, cat, filename + ext)
                if os.path.exists(p_cat):
                    img_path = p_cat
                    detected_cat = cat
                    break

        if not img_path:
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        # FIX STRUTTURALE: Se l'immagine è dentro una sottocartella di classe,
        # creiamo la stessa identica sottocartella anche dentro 'labels/'
        if detected_cat:
            target_labels_dir = os.path.join(labels_root, detected_cat)
        else:
            target_labels_dir = labels_root

        os.makedirs(target_labels_dir, exist_ok=True)

        base_name = os.path.splitext(filename)[0]
        txt_out_path = os.path.join(target_labels_dir, base_name + '.txt')

        with open(txt_out_path, 'w') as out_file:
            for obj in root.findall('object'):
                cls_name = obj.find('name').text
                if cls_name not in CLASS_TO_IDX: continue
                cls_id = CLASS_TO_IDX[cls_name]

                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)
                out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")


if __name__ == '__main__':
    process_split('train')
    process_split('val')
    print("[+] Conversione terminata con struttura speculare! Ora YOLO troverà tutto.")