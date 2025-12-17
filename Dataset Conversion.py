import json, os
from collections import defaultdict
from datetime import datetime as dt
st = dt.now()
# === Load COCO JSON ===
with open("instances_train2017.json", "r") as f:
    data = json.load(f)

images = {img['id']: img for img in data['images']}
annotations = data['annotations']
categories = data['categories']

# === Create COCO → YOLO class ID mapping ===
# Sort categories by their original COCO ID to keep consistency
category_ids = sorted([cat['id'] for cat in categories])
coco_to_yolo = {coco_id: i for i, coco_id in enumerate(category_ids)}

print("✅ Category mapping (COCO → YOLO):")
for cat in categories:
    print(f"{cat['name']:15s}  COCO:{cat['id']:>2} → YOLO:{coco_to_yolo[cat['id']]:>2}")

# === Group annotations by image ===
image_to_anns = defaultdict(list)
for ann in annotations:
    image_to_anns[ann['image_id']].append(ann)

# === Output labels ===
labels_folder = "dataset/labels"
os.makedirs(labels_folder, exist_ok=True)

for image_id, anns in image_to_anns.items():
    image_info = images[image_id]
    image_name = image_info['file_name']
    img_w, img_h = image_info['width'], image_info['height']

    label_path = os.path.join(labels_folder, image_name.replace(".jpg", ".txt"))

    with open(label_path, "w") as f:
        for ann in anns:
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            class_id = coco_to_yolo[ann['category_id']]  # Correct class ID

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("✅ YOLO txt files created successfully.")
cur = dt.now()
print(f"TIme taken : { ((cur - st).total_seconds ) / 60}")