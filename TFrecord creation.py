import tensorflow as tf
from pathlib import Path
import cv2, os, gc
from tqdm import tqdm
import numpy as np
import random
# --- Paths ---

image_dir = Path('train2017')
annotation_dir = Path('dataset/train_labels')

        
def train_annotations(image_dir, annotation_dir, class_filter, path):
    one_obj, one_annot = [], []
    app = True
    image_paths = sorted(list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')))

    for img_path in tqdm(image_paths):
        name = Path(img_path).stem
        ann_path = annotation_dir / f"{name}.txt"
        if not ann_path.exists():
            continue
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] in class_filter:
                    app = True
                    break
        if app:
            one_obj.append(img_path)
            one_annot.append(ann_path)
            app = False
    return one_obj, one_annot

img_full_dir, train_annot_full_dir = train_annotations(image_dir, annotation_dir, class_filter = '1', path = True)

img_dir, train_annot_dir = img_full_dir[:40000], train_annot_full_dir[:40000]


def train_letter_box(img_path, bboxes, target=512):
    hor, ver = False, False
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(
        img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    img = img.astype(np.float32) / 255.0

    
    annots = []
    for cid, x, y, w_box, h_box in bboxes:
        cid = int(cid)
        x, y, w_box, h_box = x * w, y * h, w_box * w, h_box * h
        x_min, x_max = x, x + w_box
        y_min, y_max = y, y + h_box
        x_min = int(round((x_min * scale) + pad_left))
        x_max = int(round((x_max * scale) + pad_left))
        y_min = int(round((y_min * scale) + pad_top))
        y_max = int(round((y_max * scale) + pad_top))
        annots.append([cid, x_min, y_min, x_max, y_max])

    return img, annots

def train_parse_annotations(image_paths, annotation_dir, class_filter):
    
    listed = list(zip(image_paths, annotation_dir))
    random.shuffle(listed)
    image_paths, annotation_dir = zip(*listed)
    image_paths = image_paths[:5000]
    annotation_dir = annotation_dir[:5000]
    
    for img_path, ann_path in zip(image_paths, annotation_dir):
        if not ann_path.exists():
            continue
        with open(ann_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            continue
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            
            if parts[0] in class_filter:
                bboxes.append(list(map(float, parts)))

        if len(bboxes) == 0:
            continue
        yield str(img_path), bboxes


def train_generator():
    for img_path, bboxes in tqdm(train_parse_annotations(img_dir, train_annot_dir, ['1']),
                                 desc="Training data", unit="img"):
        img, bboxes = train_letter_box(img_path, bboxes)
        bbox = tf.ragged.constant(bboxes, dtype=tf.float32)
        yield img, bbox


train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        tf.TensorSpec(shape=(512, 512, 3), dtype=tf.float32),
        tf.RaggedTensorSpec(shape=(None, 5), dtype=tf.float32)
    )
).batch(1, drop_remainder = True).prefetch(1)

batch_size = 1
val_batch_size = 1 


def create_tfrecord(path, dataset):
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    with tf.io.TFRecordWriter(path) as writer:
        for c, v in val_dataset:
            img = tf.io.encode_png(tf.image.convert_image_dtype(tf.squeeze(c, axis = 0), dtype=tf.uint8))
            image_bytes = img.numpy() 
            flatten = v.flat_values.numpy()
            row_num = v.row_lengths().numpy()
    
            features = {
                'image': bytes_feature(image_bytes),
                'annot': float_list_feature(flatten),
                'len': int64_list_feature(row_num)
            }
            example = tf.train.Example(features = tf.train.Features(feature = features))
            writer.write(example.SerializeToString())
    print(f"Created the Byte Dataset successfully")
path = r"C:\Users\Monesh\FCOS\tfrecords\train.tfrecord"
create_tfrecord(path, dataset)