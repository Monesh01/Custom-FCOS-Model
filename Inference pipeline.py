import ctypes
from collections import deque
import tensorflow as tf



####--------------------------------------


Model architecture to be initiated using dummy variable then load the saved weights of the fcos model before inference


####--------------------------------------


fps_window = deque(maxlen = 10)
user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)  # width
screen_h = user32.GetSystemMetrics(1)  # height
import time

st = time.perf_counter()

@tf.function
def forward(img):
    return model(img, training = False)


@tf.function
def padding(img):
    return tf.expand_dims(tf.image.resize_with_pad(img, 512, 512) / 255, axis = 0)

cap = cv2.VideoCapture(0)

while True:
    
    st = time.perf_counter()
    ref, frame = cap.read()
    if not ref:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    target = 512
    H, W, _ = img.shape
    '''
    scale = min(target/ H, target/ W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    top = (target - new_h) // 2
    bottom = target - new_h - top
    left = (target - new_w) // 2
    right = target - new_w - left
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType = cv2.BORDER_CONSTANT, value = (114, 114, 114))
    img = cv2.resize(img, (512, 512))

    #img = tf.image.resize_with_pad(img, 512, 512)
    
    #img = img.astype(np.float32) / 255
    '''
    img = padding(img) 
    #img = tf.convert_to_tensor(img, dtype = tf.float32)
    #img = tf.expand_dims(img, axis = 0)
    cls, reg, ctr = forward(img)
    strides = [4.0, 8.0, 16.0, 32.0]
    cls_ids, score, bboxes, probab = [], [], [], []

    num_classes = 1
    for i in range(4):
        stride = strides[i]
        cls_inter, reg_inter, ctr_inter = cls[i], reg[i], ctr[i]
        H, W = tf.shape(cls_inter)[1:3] 
        y, x = tf.meshgrid(tf.range(H), tf.range(W), indexing = "ij")
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])
        scores_all = tf.sigmoid(cls_inter) * tf.sigmoid(ctr_inter)
        scores_all = tf.reshape(scores_all, [-1, num_classes])
        reg_inter = tf.reshape(reg_inter, [-1, 4])
        scores = tf.reduce_max(scores_all, axis = -1)
        cls_id = tf.argmax(scores_all, axis = -1)
        topk = 1000
        k = tf.minimum(topk, tf.shape(scores)[0])
        scores, topk_idx = tf.math.top_k(scores, k=k)
        cls_id   = tf.gather(cls_id, topk_idx)
        box_pred = tf.gather(reg_inter, topk_idx)
        x        = tf.gather(x, topk_idx)
        y        = tf.gather(y, topk_idx)
    
        img_x = tf.cast(x, tf.float32)  * stride + stride / 2.0
        img_y = tf.cast(y, tf.float32) * stride + stride / 2.0
        l, t, r, b = tf.unstack(tf.cast(box_pred, tf.float32) * stride, axis = -1)
        bbox = tf.stack([img_y - t, img_x - l, img_y + b, img_x + r], axis = -1)
        
        cls_ids.append(cls_id)
        score.append(scores)
        bboxes.append(bbox)
    print("Completed conversion")

    cls_ids = tf.concat(cls_ids, 0)
    score = tf.concat(score, 0)
    bboxes = tf.concat(bboxes, 0)
    bboxes = tf.cast(bboxes, tf.float32)
    score  = tf.cast(score,  tf.float32)
    final_label, final_score, final_bbox = [], [], []
    for i in range(num_classes):
        mask = cls_ids == i
        bboxes_ = tf.boolean_mask(bboxes, mask)
        score_ = tf.boolean_mask(score, mask)
        if tf.shape(bboxes_)[0] == 0:
            continue
        selected = tf.image.non_max_suppression(bboxes_, score_, iou_threshold = 0.3, score_threshold = 0.2, max_output_size = 100)
        tf.print(f"Selected shape : {selected.shape}")
        final_label.append(tf.fill([tf.shape(selected)[0]], i))
        final_score.append(tf.gather(score_, selected))
        final_bbox.append(tf.gather(bboxes_, selected))
    final_label = tf.concat(final_label, axis = 0)
    final_score = tf.concat(final_score, axis = 0)
    final_bbox = tf.concat(final_bbox, axis = 0)
    
    
    
    tf.print(f"Final bbox shape : {final_bbox.shape}")

    
    img_num = img.numpy()[0]
    img_num = cv2.cvtColor(img_num, cv2.COLOR_RGB2BGR)
    for i in range(final_bbox.shape[0]):
    
        y1, x1, y2, x2 = tf.unstack(final_bbox[i], axis = -1)
        score = np.clip(np.cbrt(float(final_score[i])), 0.0, 0.99)
        label = f"Person : {score:.2f}"
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        img_num = cv2.rectangle(img_num, (x1, y1), (x2, y2), (0.0, 1.0, 0.0), 2)
        img_num = cv2.putText(img_num, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0.0, 1.0, 0.0), 1, cv2.LINE_AA)
    
    cur = time.perf_counter()
    fps_window.append(cur - st)
    fps = f"FPS : {1.0/ (sum(fps_window)/ len(fps_window)) :.2f}"
    print(f"TIme taken per frame : {cur-st}")
    disp = f"Custom FCOS Object Detector One Class"
    x_cen = 100
    print(x_cen)
    img_num = cv2.putText(img_num, fps, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 0.0, 0.0), 1, cv2.LINE_AA)
    img_num = cv2.putText(img_num, disp, (x_cen, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 0.0, 0.0), 1, cv2.LINE_AA)
    img_disp = (img_num * 255).astype(np.uint8)
    img_disp = cv2.resize(img_disp, (screen_w, screen_h))
    cv2.imshow("Image", img_disp)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
       