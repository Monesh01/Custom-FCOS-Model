# Light Weight Custom FCOS Object Detector (From Scratch - Tensorflow & Keras)
A lightweight, resource-aware implementation of the Fully Convolutional One Stage ( FCOS ) object detection framework
built completely from scratch without pretrained Backbone using TensorFlow.

## Motivation
This project was developed to understand and implement the lightweight FCOS object detection
architecture from first principles, without relying on prebuilt detection frameworks.

Due to limited system resources (RAM, VRAM, GPU power and long training times),
the model architecture and training strategy were carefully redesigned for **3.25 million parameters**
to remain stable, technical and trainable under the conditions and limitations.
This project urges to sacrifice the stable over raw performance.

## Resource Constraints and System Configuration

The model was developed and trained under the following hardware constraints:

- **System RAM:** 16 GB  
- **GPU:** NVIDIA RTX 3050 (Laptop)  
- **GPU Memory (VRAM):** ~4 GB (≈ 3.6 GB available for TensorFlow training)  
- **CPU:** Intel i5-11400H  

These constraints significantly influenced architectural choices, dataset size, batch size, and training strategy.


## Dataset

The **MS COCO (train2017)** dataset was used for training.  
The original dataset contains approximately **118k images across 80 object classes**, with images varying in resolution and scale.

Due to limited computational resources and long training times, the problem was **reformulated as a single-class detection task**, focusing exclusively on the **`person`** category.  
A subset of **40,000 images** containing the *person* class was selected to ensure stable training and practical experimentation.


## Dataset Conversion and Preprocessing

### Data Extraction and Filtering

From the original COCO annotations, only samples corresponding to the **person class (class ID = 1)** were extracted.  
Associated images and annotations were separated into dedicated directories to simplify the data pipeline.

Although approximately **60k person-class images** were initially extracted, the dataset was further reduced to **40k images** to accommodate memory and training-time constraints.

---

### Preprocessing Pipeline

Each image–annotation pair underwent the following preprocessing steps:

- Letterbox padding to preserve aspect ratio  
- Pixel normalization  
- Bounding box conversion to YOLO-style format:  
  `[class_id, x_min, x_max, y_min, y_max]`

Images were converted from NumPy arrays to TensorFlow tensors.  
Since each image contains a **variable number of bounding boxes**, annotations were stored using **Ragged Tensors**, enabling flexible per-image annotation handling.

---

## Data Pipeline and TFRecords

To optimize I/O performance, the dataset was serialized into **TFRecords**:

- Images were encoded in **PNG format**
- Ragged bounding box tensors were flattened and stored along with their row lengths
- This structure enables efficient reconstruction during training

Using TFRecords significantly improves throughput compared to Python-based generators, which repeatedly open, read, and close files from disk. Serialized byte streams allow faster sequential access and are well-suited for long training runs.

---

## Training–Validation Split

From the **40,000 TFRecords**:

- **38,800 samples** were used for training  
- **1,200 samples** were reserved for validation  

The training pipeline was configured with:
- **Batch size:** 4  
- **Prefetch:** 1 (training and validation)

This setup balances memory usage and data throughput under limited VRAM conditions.


## Custom FCOS Model's Architecture 
The CUstom FCOS Model's Architecture is in a conventional way of Backbone + FPN Generator, Heads Tower and output layers.
---

## 🏗️ Model Architecture
The custom FCOS architecture utilizes a **Feature Pyramid Network (FPN)** backbone with anchor-free detection heads. By predicting centerness, the model avoids the complexity of manual anchor box tuning.

<details>

Model: "fcos_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 FPN (Functional)            [(None, 128, 128, 128),   2885024   
                              (None, 64, 64, 128),               
                              (None, 32, 32, 128),               
                              (None, 16, 16, 128)]               
                                                                 
 sequential (Sequential)     (None, None, None, 64)    184832    
                                                                 
 sequential_1 (Sequential)   (None, None, None, 64)    184832    
                                                                 
 Classification (Conv2D)     multiple                  65        
                                                                 
 Regression (Conv2D)         multiple                  260       
                                                                 
 Centerness (Conv2D)         multiple                  65        
                                                                 
=================================================================
Total params: 3,255,082
Trainable params: 3,255,082
Non-trainable params: 0
_________________________________________________________________
</details>

## FCOS Model's Backbone
The Model uses Resnet Style -  Custom Backbone of 2.88 Million parameters along with the FPN generator.
The FCOS model utilizes a custom backbone incorporating bottleneck blocks and an FPN generator to extract multi-scale features, making specific use of activation functions and GroupNormalization layers throughout the process.

### Custom Backbone Architecture and Activation Functions

The custom backbone begins processing an input image shape of $$. The feature extraction relies heavily on the **bottleneck block** and specific activation functions:

1. Backbone Feature Extraction (C1, C2, C3, C4, C5):
The backbone uses the **Exponential Linear Unit (ELU)** activation function denoted mathematically as `ELU(x) = { x ​if x > 0 else α(ex−1) }` for the stability of the Gradients and the smooth curve for the negative side which also gives smooth gradient flow.

A bottleneck block structure generally includes several components:
   -> Initial Convolutions:** 1x1 Conv, followed by GroupNormalization (GN), and then ELU activation.
   -> 3x3 Conv: Followed by GN and ELU activation.
   -> Expansion (1x1 Conv): Which increases the filter count (multiplying by 2).
   -> Shortcut Path: If downsampling occurs (stride > 1) or the channel count changes, a 1x1 Conv is applied to the shortcut. The     shortcut is then added to the main path output, followed by GN and ELU activation.
   -> Final Block Layers: The block concludes with a 3x3 Conv, GN, and ELU activation.

The initial stages use `filters=32`. For the first feature map (`c1`), the sequence is a 5x5 convolution (stride 2), GroupNormalization (16 groups), ELU activation, and a 2x2 MaxPooling (stride 2). Subsequent feature maps are derived using bottleneck blocks:
*   `c2` uses two blocks (filters = 32).
*   `c3` uses two blocks, including one downsampling block (filters = 64).
*   `c4` uses one downsampling block (filters = 128).
*   `c5` uses one downsampling block (filters = 128).

2. FPN Generator and Activation Functions:

The Feature Pyramid Network (FPN) generator processes the feature maps (C2, C3, C4, C5) to create multi-scale pyramid features (P2, P3, P4, P5).

The FPN uses the **Leaky Rectified Linear Unit (LeakyReLU)** activation function, defined with an alpha value of 0.01 and denated as 
LeakyRELU(x) = { x ​if x>0 else αx) }. Here leakyRelu gives a equal scale magnitude for the all the different p levels so the features and the spatial informations where preserved and conservated for the next layer which gives smooth leaning over the spatial informations across the layers.

In the `FPN_generator`:
*   The input feature map C is processed by a lateral 1x1 convolution and lateral GroupNormalization (`lat_convs`, `lat_bns`).
*   This output is immediately followed by **LeakyReLU** activation.
*   If an upstream merged feature map is present, it is upsampled and added to the current feature.
*   The resulting merged feature  is normalized using `fuse_bns` and then activated using **LeakyReLU**.
*   The final pyramid feature  is generated by a 3x3 smooth convolution, `smooth_bns` (GroupNormalization), and another **LeakyReLU** activation.

The difference in activation function usage is clear: the main backbone components rely on the **ELU** function, while the FPN generator utilizes the **LeakyReLU** function.

### Use of GroupNormalization (GN)

The model consistently uses **GroupNormalization (GN)** layers, implemented as `tfa.layers.GroupNormalization`. Normalization helps stabilize the training process.

1.  **Backbone:** Within the bottleneck blocks, GN is applied with **32 groups**. In the very first layer of the backbone (generating `c1`), GN is applied with **16 groups**.
2.  **FPN Generator:** GN is used in the lateral convolutions (`lat_bns`), the smooth convolutions (`smooth_bns`), and the fusion steps (`fuse_bns`), all specified to use **32 groups**.

### Backbone Size and FPN Levels

**1. Backbone Size (Filters):**
The backbone uses a relatively smaller filter count, starting with 32 filters in `c2` and gradually increasing to 128 filters in `c4` and `c5` to remain the semantic information across the feature maps and layers without increasing the Channels count and keep the model's backbone lightweight.
**2. Number of FPN Levels (4 P Levels):**
The architecture utilizes **four pyramid levels**: P2, P3, P4, and P5. These levels correspond to feature maps with strides of 4, 8, 16, and 32 pixels relative to the **512 x 512 image size**.

*   P2 (stride 4) has a size of $512/4 = 128$.
*   P3 (stride 8) has a size of $512/8 = 64$.
*   P4 (stride 16) has a size of $512/16 = 32$.
*   P5 (stride 32) has a size of $512/32 = 16$.

The use of four levels (P2 to P5) is standard in many FPN-based detectors, as it provides a comprehensive range of resolutions necessary for detecting objects across various scales. This multi-scale approach is integral to FCOS, as ground-truth bounding boxes are assigned to one of these levels based on their area.

Here, the objects with areas:
*   Less than or equal to 4096 are assigned to P2 (stride 4).
*   Between 4096 and 16384 are assigned to P3 (stride 8).
*   Between 16384 and 65536 are assigned to P4 (stride 16).
*   Larger than 65536 are assigned to P5 (stride 32).

The sources show that these four levels are essential for the target assignment process by mapping objects of different sizes to the appropriate feature map resolution.

## FCOS Model's Head Conv towers
The FCOS model utilizes a structure composed of shared head towers and three distinct final prediction heads (Classification, Regression, and Centerness) to produce the final outputs.

### Shared Head Towers and Swish Activation

The FCOS architecture employs two separate but identically structured head towers, one for classification and centerness (`self.cls_tower`) and one for regression (`self.reg_tower`). Both towers are shared across all levels of the Feature Pyramid Network (FPN) features.

#### Head Tower Structure (`create_head_tower`):
Each tower is built sequentially using a filter size of 64 (`fill = 64`). Each tower consists of ** four repeated convolutional blocks**.
A single convolutional block in the head tower includes:
1.  A 3x3 Convolution (`layers.Conv2D(fill, 3, padding='same', use_bias=False)`).
2.  **GroupNormalization** with 16 groups (`tfa.layers.GroupNormalization(groups=16, axis=-1)`).
3.  **Swish Activation** (`layers.Activation(tf.keras.activations.swish)`).

#### Swish Activation
The **Swish activation function** is used consistently throughout the head towers. Swish is known for being a self-gated activation function that generally performs better than ReLU in deeper models for a smoothly convergence over the negative tail and better gradient flow even with the negative side. Mathematically denoted as **Swish(x) = x * sigmoid(x)**


### Final Prediction Heads (Output Layers)

The outputs from the head towers are passed into three distinct 1x1 convolution layers, which serve as the final prediction heads for the FCOS model. None of these final heads use an explicit activation function (`activation=None`).

**1. Classification Head (`self.cls_head`):**
*   **Purpose:** Predicts the class score for each spatial location.
*   **Structure:** A 1x1 Convolution layer that outputs **$N$ channels**, where `N` is `self.num_classes`.
*   **Initialization:** It uses a specialized bias initialization (`bias\_initializer=tf.constant\_initializer(bias\_val)`), calculated using a prior probability `p=0.05: bias = -np.log((1 - p) / p)`. This initialization helps stabilize training by biasing the model toward predicting negative (background) samples initially.

**2. Regression Head (`self.reg_head`):**
*   **Purpose:** Predicts the distance from the spatial location to the four boundaries of the bounding box (Left, Top, Right, Bottom).
*   **Structure:** A 1x1 Convolution layer that outputs **4 channels**.
*   **Initialization:** It uses a bias initialization of `-np.log(100)`.

**3. Centerness Head (`self.ctr_head`):**
*   **Purpose:** Predicts the measure of centerness for a potential object, which down-weights bounding boxes generated far from the center of an object.
*   **Structure:** A 1x1 Convolution layer that outputs **1 channel**.
*   **Initialization:** It uses a bias initialization of -2.0.

### Final Output Calculation

When the model executes the forward pass (`call` function), the feature map $f$ from an FPN level is processed by the heads to produce the final predictions.

1.  **Classification Output: The final classification output is obtained by applying the sigmoid function to the prediction logits when evaluating the output for metrics like mAP (`tf.nn.sigmoid(Class_pred[i])`).
2.  **Centerness Output: Similar to classification, the final centerness score for metrics is obtained by applying the sigmoid function (`tf.nn.sigmoid(Ctr_pred[i]))`).
3.  **Regression Output: The raw output is then transformed using a level-specific, learnable scaling factor (`self.scale[i]`) and an exponential function: of **exp(x * scale[i]) The use of the exponential function ensures that the predicted distances (LTRB) are positive.

The model returns lists of these three types of outputs for all FPN levels (P2, P3, P4, P5): `cls_outputs`, `reg_outputs`, and `ctr_outputs`.

## Target Assignment ( Training Targets Creation)
The target assignment process in FCOS determines which spatial locations (feature map points) on the Feature Pyramid Network (FPN) levels are responsible for predicting a given ground-truth bounding box. This assignment is crucial because FCOS is anchor-free, meaning targets must be defined directly on the feature map grid points.

Here is the mathematical intuition behind the assignment process and the resulting target tensors.

### 1. Target Assignment Intuition

Target assignment involves two main steps: **FPN Level Selection** (Area-based) and **Positive Sample Selection** (Center-based).

#### A. FPN Level Selection (Area-based)

The first assignment rule ensures that objects of different sizes are assigned to their appropriate FPN level, mitigating the need for multiple anchors per location. This is achieved by calculating the area of the ground-truth box `Area = (x_max - xmin) \(y_max - y_min)` and using thresholds:

*   Objects with `Area = 4096` are assigned to level **P2** (Stride `S=4`).
*   Objects with `4096 < Area = 16384` are assigned to level **P3** (Stride `S=8`).
*   Objects with `16384 <  Area = 65536` are assigned to level **P4** (Stride `S=16`).
*   Objects with `Area > 65536` are assigned to level **P5** (Stride `S=32`).
This process is managed by the `target_assignment` function, which returns the pyramid level (`p_tar`) and its corresponding stride (`stride_`) for each ground-truth box.

#### B. Positive Sample Selection (Center-based)

For a ground-truth box assigned to a specific level (P`i` with stride `S`), a feature map point is considered a positive sample if it meets two conditions, both applied at the feature map resolution:

**Condition 1: Inside the Bounding Box**
The feature map point must correspond to an image center point that falls within the original ground-truth bounding box coordinates `x_min, y_min, x_max, y_max`. The image center coordinates `x_img, y_img` corresponding to the grid point are calculated based on the stride `S`:
`x_img = (x_grid \ S) + S/2`
`y_img = (y_grid \ S) + S/2`
The point is "in box" if:
`in_box = (x_img > x_min and x_img < x_max) and (y_img > y_min and y_img < y_max)`
.

**Condition 2: Inside the Center Region (FCOS modification)**
To prevent multiple objects competing for central locations, FCOS only assigns positive labels to points near the center of the ground-truth box. This is defined using a center region "radius" calculation:
The center of the bounding box in feature-map coordinates is computed as:

x_center = ((x_min + x_max) / 2) / S  
y_center = ((y_min + y_max) / 2) / S  

The radius is derived from the bounding box size relative to the stride `S`, with a minimum radius of 1.0:

radius_x = max(1.0, (x_max - x_min) / S)  
radius_y = max(1.0, (y_max - y_min) / S)  

A grid location is considered **inside the center sampling radius** if both conditions are satisfied:

|x_grid − x_center| ≤ radius_x  
|y_grid − y_center| ≤ radius_y

.

A location is selected as a positive sample if `indices = in_box and rad`.

**Conflict Resolution (The "Farthest is Better" Rule):**
It is possible for multiple ground-truth boxes to claim the same location `x_grid,  y_grid`. FCOS resolves this conflict using the object area. A location is assigned to the ground-truth box that has the **smallest area** among all candidates that overlap that point.
The `assigned_area` tensor keeps track of the area of the object currently assigned to each location. A location update occurs only if the current box's area is smaller than the already assigned area (initialized to infinity).
update_mask is computed as:

update_mask = indices AND (box_area < assigned_area_nonzero)

The `final_mask` represents the coordinates that are updated by the current ground-truth box.

### 2. Target Tensors Creation

For every assigned positive location determined by `final_mask`, three types of target tensors are created:

#### A. Classification Targets (`cls_targets`)

For positive samples, the classification target tensor is updated to reflect the presence of an object of a specific class.
*   The class ID is used to index the correct channel in the classification target tensor, and a value of **1.0** is assigned to indicate a positive sample.
*   The classification targets for each level (P2-P5) are initialized as zero tensors with shape `(Size, Size, num_classes)`.

#### B. Regression Targets (`reg_targets`)

Regression targets represent the normalized distances from the center point (x_img, y_img) to the four sides of the assigned ground-truth bounding box (Left, Top, Right, Bottom), relative to the feature map stride `S`.

The normalized distances are computed as:

l = (x_img - x_min) / S  
t = (y_img - y_min) / S  
r = (x_max - x_img) / S  
b = (y_max - y_img) / S  

The regression target is stacked as:

ltrb = [l, t, r, b]

This 4-channel tensor is used as the supervision signal for the regression head during training.


#### C. Centerness Targets (`ctr_targets`)

The centerness target is calculated only for positive samples and is designed to penalize locations far from the center of the assigned bounding box. This target is a scalar value (between 0 and 1) derived from the regression targets (l, t, r, b):

Centerness is computed as the geometric mean of normalized left–right and top–bottom distances:

centerness = sqrt(
    (min(l, r) / (max(l, r) + 1e-6)) *
    (min(t, b) / (max(t, b) + 1e-6))
)

A value close to 1 indicates the location is near the center, while a value close to 0 indicates it is near the edge, resulting in a lower weight during centerness loss calculation. The centerness target tensor is initialized as zeros with shape `(Size, Size, 1)`.

## Loss Functions 
The FCOS model uses three distinct loss functions, applied simultaneously, covering classification, bounding box regression, and centerness prediction. These losses are normalized by the total number of positive locations across all FPN levels.

### 1. Classification Loss: Focal Loss

The classification loss is calculated using **Focal Loss** (L_cls), which addresses the imbalance between foreground and background samples.

The loss is defined based on the standard Binary Cross-Entropy (BCE) combined with a modulating factor derived from the prediction confidence (p_t).

**A. Binary Cross-Entropy (BCE) Calculation:**
The process begins by computing the binary cross-entropy (BCE) loss between the classification prediction logits and the binary class targets:

BCE = tf.nn.sigmoid_cross_entropy_with_logits(
    labels = cls_target,
    logits = cls_out
)


**B. Modulating Factor Components:**
1.  **Probability of Target (p_t):** This represents the model's confidence in the assigned ground-truth class is computed from the sigmoid-activated classification prediction:

p = tf.sigmoid(cls_out)

The target-aligned probability p_t is defined as:

- p_t = p, if target == 1.0
- p_t = 1.0 - p, if target == 0.0

In the implementation, p_t is computed using `tf.where` for numerical stability and vectorized execution.


2.  **Alpha Factor ($\alpha$):** This term weights positive and negative examples (with alpha = 0.25 by default):

alpha_factor is defined as:

- alpha_factor = alpha, if target == 1.0
- alpha_factor = 1.0 - alpha, if target == 0.0


3.  **Focal Weight:** This term reduces the loss contribution from easy (well-classified) examples using the focusing parameter gamma (gamma = 2.0 by default):

focal_weight = alpha_factor * (1.0 - p_t) ** gamma


**C. Focal Loss per Element:**
The element-wise classification loss is computed as:

cls_loss_element = focal_weight * BCE


**D. Total Classification Loss:**
The final classification loss is computed by summing the element-wise losses over all spatial locations and normalizing by the number of positive samples:

cls_loss = sum(cls_loss_element) / normalizer

where:

normalizer = max(1.0, num_pos)

Here, num_pos represents the total number of positive locations across all FPN levels.


### 2. Regression Loss: Generalized Intersection over Union (GIoU) Loss

The bounding box regression loss uses the **GIoU Loss** (L_reg). GIoU loss is calculated only over the locations identified as positive samples (foreground mask).

**A. Bounding Box Transformation:**
The predicted normalized LTRB distances (`reg_out`) and the target LTRB distances (`reg_target`) are converted back into absolute image-space bounding box coordinates using the feature map locations (`coords`) and the stride `S`.

For each feature map location with image-space center coordinates `(x, y)`, the bounding box is decoded as:

x_min = x - l * S  
y_min = y - t * S  
x_max = x + r * S  
y_max = y + b * S  

where `l`, `t`, `r`, and `b` are the normalized distances predicted by the regression head.

This decoding is applied independently for:
- predicted boxes (`pred_box`)
- target boxes (`target_box`)

**B. Calculation of IoU and Enclosing Area:**
1. **Intersection Area (`inter_area`):**  
   Computed as the overlapping area between the predicted foreground bounding box (`pred_box_fg`) and the target foreground bounding box (`target_box_fg`).

2. **Union Area (`union_area`):**  
   Computed as the sum of the areas of the predicted and target boxes minus the intersection area.

3. **Intersection over Union (IoU):**  

   IoU = inter_area / (union_area + 1e-6)

4.  **Smallest Enclosing Box Area (G):** This is the area of the smallest box that encloses both the prediction and the target box.

**C. GIoU Loss Calculation:**
The Generalized IoU (GIoU) metric is computed by adding a penalty term that accounts for the area of the smallest enclosing box `G`, which penalizes large gaps between the union area and the enclosing box.

GIoU is defined as:

GIoU = IoU - ((G - union_area) / (G + 1e-6))

The element-wise regression loss is then computed as one minus the clipped GIoU value:

reg_loss_element = 1.0 - tf.clip_by_value(GIoU, -1.0, 1.0)

### D. Total Regression Loss

The total regression loss is computed by summing the element-wise regression losses over all positive locations and normalizing by the total number of positive samples used for classification:

reg_loss = sum(reg_loss_element) / total_pos

Here, `total_pos` represents the total number of positive locations across all FPN levels.

### 3. Centerness Loss (Binary Cross-Entropy)

The centerness loss ensures that feature map locations closer to the center of the assigned ground-truth bounding box produce higher centerness scores. This loss is computed **only for positive samples** using binary cross-entropy (BCE).


#### A. Centerness Loss Calculation

The element-wise centerness loss is computed using binary cross-entropy between the centerness prediction logits and the centerness targets, restricted to positive locations only:

ctr_loss_element = tf.nn.sigmoid_cross_entropy_with_logits(
    labels = ctr_target,
    logits = ctr_pred
)


#### B. Total Centerness Loss

The total centerness loss is computed by summing the element-wise losses over positive locations and normalizing by the sum of centerness target values, which acts as a weighted normalizer:

ctr_normalizer = max(1.0, sum(ctr_target) + 1e-6)
ctr_loss = sum(ctr_loss_element) / ctr_normalizer


### Total Loss

The final training objective is defined as the sum of the classification, regression, and centerness losses:

total_loss = cls_loss + reg_loss + ctr_loss

Note: In some implementations, the centerness loss may be scaled by a coefficient (e.g., `coef * ctr_loss`), but the default formulation used here is a direct summation.


## LR Scheduler ( Cosine Decay Schema)
The FCOS model utilizes a custom **Learning Rate (LR) Scheduler** featuring a warm-up phase followed by a cosine decay, and a structured **Training Schema** that incorporates distinct optimizers for the backbone and heads, gradient accumulation, and mixed-precision training.


### Learning Rate Scheduler (Warm-up + Cosine Decay)

The learning rate scheduler is defined using the following parameters:

- Initial learning rate for warm-up (`lr_start`): implicitly set to `lr_max`
- Minimum learning rate (`lr_min`): 0.00000025
- Maximum learning rate (`lr_max`): 0.00005
- Total training steps (`total_steps`): epochs × counts  
  where `counts = 40000 // 4`
- Warm-up steps (`warm_up`): 5000
- Cosine decay steps (`T`): total_steps − warm_up

The `lr_scheduler` function computes the learning rate based on the current training step `t`.


#### 1. Warm-up Phase (t ≤ warm_up)

During the warm-up phase, the learning rate increases linearly from the starting value to the maximum learning rate:

lr = lr_start + ((lr_max - lr_start) / warm_up) * t


#### 2. Cosine Decay Phase (t > warm_up)

After the warm-up phase, cosine annealing is applied to gradually decay the learning rate from `lr_max` to `lr_min` over the remaining steps.

First, the step index is shifted:

t_prime = t - warm_up

The decay ratio is computed as:

decay = t_prime / T

The learning rate is then calculated using cosine decay:

lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(decay * pi))


### Training Schema of the Custom FCOS Model

The training process is defined by specific configurations regarding optimization, precision, and handling of gradients across the model components.

#### 1. Optimization and Learning Rates
The model uses the **AdamW** optimizer. Crucially the Training schema is based on Continue Learning. A common Approach used here as From the 40k chunks 10k chunks used per 30 epoches. Then the next chunk is training.
The Lr schduler with Warmup is the responsible to avoid the Catastropic Forgetting for the on going chunk.
For the Final Fine Tunining we used, 
the backbone and the prediction heads are optimized separately using distinct learning rates, a common strategy for transfer learning or fine-tuning:
- **Backbone Optimizer (`adamw_optimizer`):**  
  Uses a base learning rate of `0.000001` (lr = 1e-6).

- **Heads Optimizer (`adamw_optimizer`):**  
  Uses a higher learning rate of `0.00001` (lr = 1e-5).  
  This learning rate is **10× larger** than the backbone learning rate and is intended for fine-tuning the detection heads.

- **Weight Decay:**  
  Both optimizers ('adamw_optimizer` and `adamw_optimizer`) use a weight decay value of `5e-3`.


#### 2. Mixed Precision Training
The model is configured to use **mixed precision** (`mixed_float16`) globally, which helps speed up training by performing calculations using lower precision floating point numbers. Both the backbone and heads optimizers are wrapped in `mixed_precision.LossScaleOptimizer` to manage potential underflow issues caused by mixed precision.

#### 3. Gradient Accumulation

To simulate a larger effective batch size and stabilize gradient estimation, the model uses **gradient accumulation** over `acc_num = 8` steps.

- Gradients (`grads`) are computed for each mini-batch.
- If the batch counter (`counter`) is less than `acc_num`, gradients are accumulated into the `acc_grads` variables.
- Once `acc_num` (8) batches have been processed:
  - The accumulated gradients are retrieved
  - Gradients are unscaled (for mixed precision training)
  - Gradients are normalized by the accumulation count (`acc_counter`)
- The normalized gradients (`acc_norm_u`) are clipped using a global norm of `10.0` before being applied.
- This effectively simulates a larger batch size, following the intuition:

  Effective batch size = 8 × 4 = 32

#### 4. Differential Optimization for fine tuning on 40k records
The normalized gradients are applied separately to the backbone and the heads:
*   Gradients corresponding to variables belonging to the `FPN` layer (backbone) are stored in `bb_grads`.
*   Gradients corresponding to all other trainable variables (heads) are stored in `heads`.
*   The `adamw_optimizer` applies gradients only to the backbone variables, and the `adamw_optimizer` applies gradients only to the head variables.

#### 5. Training Fine-Tuning Process and Monitoring

The training process runs for a fixed number of epochs (`epochs = 10`) and follows a controlled fine-tuning strategy.

- For each batch, the following losses are computed:
  - Classification loss (`cls_loss`)
  - Regression loss (`reg_loss`)
  - Centerness loss (`ctr_loss`)

- The total loss is computed as:

  total_loss = cls_loss + reg_loss + (coef * ctr_loss)

  where the centerness loss coefficient is set to `coef = 1.0`.

- Before gradient computation, the total loss is scaled for mixed-precision training:

  total_loss = bb_optimizer.get_scaled_loss(total_loss)

- At the end of each epoch, the following metrics are logged using `mlflow`:
  - Classification loss
  - Regression loss
  - Centerness loss
  - Mean Average Precision (mAP)

- An early stopping mechanism monitors the validation loss (`val_cl`):
  - If the validation loss does not improve for 7 consecutive epochs (`pat >= 7`), training is stopped
  - The best-performing model checkpoint is saved automatically

## NMS Per class Scoring
The model utilizes a **Per-Class Non-Maximum Suppression (NMS)** schema for post-processing predictions and defines a specific approach for **Validation Data Split**, relying on a subset of the primary training dataset.

### Non-Maximum Suppression (NMS) Schema

The NMS process is applied after the model generates raw predictions to filter out redundant bounding boxes and obtain the final set of detections. The model uses a technique called **Per-Class NMS**.

#### 1. Prediction Generation

Raw model outputs from all FPN levels (classification logits, regression distances, and centerness logits) are first aggregated and processed.

- **Classification Probability (`cls_prob`):**  
  Computed by applying the sigmoid function to the classification logits:
  
  cls_prob = tf.nn.sigmoid(Class_pred)

- **Centerness Probability (`ctr_prob`):**  
  Computed by applying the sigmoid function to the centerness logits:
  
  ctr_prob = tf.nn.sigmoid(Ctr_pred)

- **Final Score:**  
  The confidence score for each location is computed as the product of classification probability and centerness probability:
  
  scores = cls_prob * ctr_prob

- **Location Filtering:**  
  Only predictions with scores greater than `0.001` are retained for further processing.

- **Bounding Box Decoding:**  
  The regression outputs (LTRB distances) are scaled by the corresponding FPN stride `S` and combined with the feature map coordinates to recover image-space bounding box coordinates:

  x_min = img_x - l  
  y_min = img_y - t  
  x_max = img_x + r  
  y_max = img_y + b  

- All decoded predictions are concatenated into:
  - `final_labels`
  - `final_scores`
  - `final_bboxes`

#### 2. Per-Class Non-Maximum Suppression (NMS)

The `per_class_nms` function applies Non-Maximum Suppression independently for each class.

- The function iterates over each class index from `0` to `num_classes - 1`.
- For each class:
  - Bounding boxes and scores belonging only to that class are selected.
  - `tf.image.non_max_suppression` is applied using the following parameters:
    - Maximum output size per class: `max_per_class = 100`
    - IoU threshold: `iou_threshold = 0.75`
    - Score threshold: `score_threshold = 0.05`
- The indices returned by the NMS operation are used to select the final boxes and scores for that class.
- The filtered predictions from all classes are combined to produce the final set of non-redundant detections.

### Score Interpretation

The score shown on each predicted bounding box represents the **classification probability**, not the overall bounding box confidence.

In FCOS, the final score used internally for ranking and Non-Maximum Suppression (NMS) is computed as:

score = cls_prob × centerness

This combined score reflects both:
- the likelihood of the object belonging to a class, and
- the spatial quality of the prediction.

However, for visualization purposes, only the **cube root of the (`score`)** is displayed. This value represents the model’s belief that the detected object belongs to the target class and is more intuitive for human interpretation.

The centerness term is used internally to suppress low-quality or off-center predictions and is **not a direct confidence measure**.


### Validation Data Split Schema

The model uses a subset of the main training data for validation, ensuring that these records are not used in the gradient calculation for training.

*   **Total Records:** The primary training dataset contains 40,000 records, of which 38,800 are usable for training.
*   **Validation Set Source:** The validation dataset (`val_dataset`) is constructed by taking the **first 1200 records** from the training TFRecord file (`ttfrecord_path`).
*   **Training Set Exclusion:** Correspondingly, the main training dataset (`train_dataset`) is constructed by **skipping the first 1200 records** from the same TFRecord file.
*   This setup ensures that the 1200 records used for validation are exclusive and not used during the main training loop, thus providing an unbiased measure of model performance.
*   The validation process measures Classification Loss, Regression Loss, Centerness Loss, and Mean Average Precision (mAP). The validation loss (Val_cls) is tracked, and an early stopping mechanism is implemented if the validation loss does not decrease after 7 epochs.

## Logs Monitoring
We monitor the logs of the Custom Fcos model using Mlflow uim which gives excellent graphs of logs statistics metrics and cross platform monitoring over wifi. 
This all makes it a better suits for a logs monitoring effficent and no memory spikes in the middle of the training.

The model was trained on 40,000 images using a curriculum approach. While the first two runs (20k images) focused on initial feature alignment and backbone stabilization but the graphs and data for the first two run in the curriculum training was lost due to mlflow data corrupted.

But the 3rd and 4th run along with the fine tuning of final phase has been given in the assests.

## Inferenece Pipeline
FCOS model inference uses Camera for the realtime detection and optimised the forward pass using **@tf.function**.
The following where the preprocessing done before feeding the model.
*   Capture the Frame from the Camera using "cv2.VideoCapture" command.
*   Convert the image to resize with padding if required.
*   Feed the image to the model.
*   Output is the Cls, Reg and Ctr which are the predictions given by the model.
*   The scoring method here used is **CTR x CLS**.
*   NMS over the prediction using **IOU Threshold as 0.3** and "**Score Threshold as 0.2** 
*   The resultant bboxes and scores are then applied to the image using cv2 then resize the image to the screen size and display.

## Results and Observations

- The custom FCOS model was successfully trained under strict memory constraints.
- Training remained stable across epochs with controlled classification, regression, and centerness losses.
- Mean Average Precision (mAP@0.75 ~ 0.375 - 0.395) showed gradual improvement during fine-tuning, indicating effective learning despite reduced model capacity and dataset size.
- The model demonstrates reliable person detection across varying scales and aspect ratios.

<img width="641" height="553" alt="Screenshot 2025-12-15 102946" src="https://github.com/user-attachments/assets/9e9b26f7-385d-42c9-81bb-f324a0507b19" />

*Qualitative inference results of the custom FCOS model trained on the COCO Person class, demonstrating robustness across indoor and outdoor scenes with crowded scenario.*

<img width="622" height="551" alt="Screenshot 2025-12-15 100831" src="https://github.com/user-attachments/assets/472c99aa-2492-493f-af99-4bf88f676e69" />

*Qualitative inference results of the custom FCOS model trained on the COCO Person class, demonstrating robustness across indoor and outdoor scenes.*

<img width="642" height="627" alt="image" src="https://github.com/user-attachments/assets/11e01fb9-6a0e-42a3-afce-8394f45ede05" />

*Qualitative inference results of the custom FCOS model trained on the COCO Person class, demonstrating robustness across indoor and outdoor scenes.*

## Design Trade-offs

This project prioritizes interpretability and training stability over raw performance. Several architectural and training simplifications
were made to ensure feasibility under limited hardware resources.


## Limitations and Trade-offs

- Training was restricted to a single class due to limited GPU memory.
- Batch size and input resolution were constrained to fit within available VRAM.
- The backbone and head depth were reduced compared to standard FCOS implementations.
- mAP performance is not directly comparable to large-scale pretrained models trained on the full COCO dataset.

## Future Work

- Extend training to multiple object classes.
- Increase input resolution and batch size on higher-memory GPUs.
- Explore pretrained backbone initialization.
- Optimize inference using TensorRT.

## How to run
The model code is simply one combined Code as CUSTOM_FCOS.py 
**Open CMD (through directory of the code) --> Type python -u CUSTOM_FCOS.py > logfile.txt 2>&1**

The above is the simpliest way to run the code in the windows terminals for the efficieny over Jupyter Lab web page memory usage and memory spikes
and Python IDLE storing the output logs and other variables in the memory for a long period gives periodic memory spikes and consumption.
This execution method was chosen specifically to ensure stability during long training sessions where each epoch may take several hours.

### Pretrained Weights

Pretrained weights for the final fine-tuned model are provided in the `weights/` directory.

To run inference using the pretrained model, ensure the weights path is correctly set in `inference_custom_fcos.py`.

