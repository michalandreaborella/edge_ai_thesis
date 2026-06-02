import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

DATA_ROOT = 'data/neu-det'
OUT_ROOT = 'data/neu-det-yolo'

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


def flatten_and_convert(input_name, output_name):
    print(f"\n[*] Appiattimento: leggo da '{input_name}', salvo in '{output_name}'")

    ann_dir = os.path.join(DATA_ROOT, input_name, 'annotations')
    img_dir = os.path.join(DATA_ROOT, input_name, 'images')

    out_img_dir = os.path.join(OUT_ROOT, output_name, 'images')
    out_lbl_dir = os.path.join(OUT_ROOT, output_name, 'labels')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files):
        tree = ET.parse(os.path.join(ann_dir, xml_file))
        root = tree.getroot()

        filename = root.find('filename').text
        base_name = os.path.splitext(filename)[0]

        # IL FIX: Evita di cercare .jpg.jpg! (Preso dal tuo script originale)
        possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG'] if not os.path.splitext(filename)[1] else ['']

        img_path = None
        for ext in possible_extensions:
            p = os.path.join(img_dir, filename + ext)
            if os.path.exists(p):
                img_path = p
                break
            for cat in CLASSES:
                p_cat = os.path.join(img_dir, cat, filename + ext)
                if os.path.exists(p_cat):
                    img_path = p_cat
                    break

        if not img_path:
            print(f"Errore: Immagine per {xml_file} non trovata in {img_dir}")
            continue

        new_img_path = os.path.join(out_img_dir, base_name + os.path.splitext(img_path)[1])
        shutil.copy2(img_path, new_img_path)

        with Image.open(new_img_path) as img:
            w, h = img.size

        txt_out_path = os.path.join(out_lbl_dir, base_name + '.txt')
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
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)

    flatten_and_convert('train', 'train')
    # IL FIX: Legge dalla cartella originale "validation", salva nella cartella YOLO "val"
    flatten_and_convert('validation', 'val')