# Custom FCOS Objecct Detector (From Scratch - Tensorflow & Keras)
A lightweight, resource-aware implementation of the Fully Convolutional One Stage ( FCOS ) object detection framework
built completely from scratch without pretrained Backbone using TensorFlow.

## Motivation
This project was developed to understand and implement the FCOS object detection
architecture from first principles, without relying on prebuilt detection frameworks.

Due to limited system resources (RAM, VRAM, GPU power and long training times),
the model architecture and training strategy were carefully redesigned
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

## FCOS Model's Backbone
The Model uses Resnet Style -  Custom Backbone of 2.88 Million parameters along with the FPN generator.
The FCOS model utilizes a custom backbone incorporating bottleneck blocks and an FPN generator to extract multi-scale features, making specific use of activation functions and GroupNormalization layers throughout the process.

### Custom Backbone Architecture and Activation Functions

The custom backbone begins processing an input image shape of $$. The feature extraction relies heavily on the **bottleneck block** and specific activation functions:

1. Backbone Feature Extraction (C1, C2, C3, C4, C5):
The backbone uses the **Exponential Linear Unit (ELU)** activation function denoted mathematically as ELU(x) = { x ​if x>0 else α(ex−1) } for the stability of the Gradients and the smooth curve for the negative side which also gives smooth gradient flow.

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
*   **Structure:** A 1x1 Convolution layer that outputs **$N$ channels**, where $N$ is `self.num_classes`.
*   **Initialization:** It uses a specialized bias initialization (`bias\_initializer=tf.constant\_initializer(bias\_val)`), calculated using a prior probability $p=0.05$: $\text{bias\_val} = -\text{np.log}((1 - p) / p)$. This initialization helps stabilize training by biasing the model toward predicting negative (background) samples initially.

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

The first assignment rule ensures that objects of different sizes are assigned to their appropriate FPN level, mitigating the need for multiple anchors per location. This is achieved by calculating the area of the ground-truth box ($\text{Area} = (x_{\max} - x_{\min}) \times (y_{\max} - y_{\min})$) and using thresholds:

*   Objects with $\text{Area} \le 4096$ are assigned to level **P2** (Stride $S=4$).
*   Objects with $4096 < \text{Area} \le 16384$ are assigned to level **P3** (Stride $S=8$).
*   Objects with $16384 < \text{Area} \le 65536$ are assigned to level **P4** (Stride $S=16$).
*   Objects with $\text{Area} > 65536$ are assigned to level **P5** (Stride $S=32$).
This process is managed by the `target_assignment` function, which returns the pyramid level (`p_tar`) and its corresponding stride (`stride_`) for each ground-truth box.

#### B. Positive Sample Selection (Center-based)

For a ground-truth box assigned to a specific level (P$i$ with stride $S$), a feature map point $(x_{\text{grid}}, y_{\text{grid}})$ is considered a positive sample if it meets two conditions, both applied at the feature map resolution:

**Condition 1: Inside the Bounding Box**
The feature map point must correspond to an image center point that falls within the original ground-truth bounding box coordinates $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$. The image center coordinates $(\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}})$ corresponding to the grid point are calculated based on the stride $S$:
$$\mathbf{x}_{\text{img}} = (\mathbf{x}_{\text{grid}} \times S) + S/2$$
$$\mathbf{y}_{\text{img}} = (\mathbf{y}_{\text{grid}} \times S) + S/2$$
The point is "in box" if:
$$\text{in\_box} = (\mathbf{x}_{\text{img}} \ge x_{\min}) \land (\mathbf{x}_{\text{img}} \le x_{\max}) \land (\mathbf{y}_{\text{img}} \ge y_{\min}) \land (\mathbf{y}_{\text{img}} \le y_{\max})$$
.

**Condition 2: Inside the Center Region (FCOS modification)**
To prevent multiple objects competing for central locations, FCOS only assigns positive labels to points near the center of the ground-truth box. This is defined using a center region "radius" calculation:
The center coordinates on the feature grid are:
$$x_{\text{center}} = \frac{(x_{\min} + x_{\max}) / 2}{S}$$
$$y_{\text{center}} = \frac{(y_{\min} + y_{\max}) / 2}{S}$$
The radius is derived from the box size relative to the stride, ensuring a minimum radius of 1.0:
$$\text{radius}_{\text{x}} = \max\left(1.0, 1.0 \times \frac{x_{\max} - x_{\min}}{S}\right)$$
$$\text{radius}_{\text{y}} = \max\left(1.0, 1.0 \times \frac{y_{\max} - y_{\min}}{S}\right)$$
A grid point is "in radius" if:
$$\text{rad} = \left(|\mathbf{x}_{\text{grid}} - x_{\text{center}}| \le \text{radius}_{\text{x}}\right) \land \left(|\mathbf{y}_{\text{grid}} - y_{\text{center}}| \le \text{radius}_{\text{y}}\right)$$
.

A location is selected as a positive sample if $\text{indices} = \text{in\_box} \land \text{rad}$.

**Conflict Resolution (The "Farthest is Better" Rule):**
It is possible for multiple ground-truth boxes to claim the same location $(x_{\text{grid}}, y_{\text{grid}})$. FCOS resolves this conflict using the object area. A location is assigned to the ground-truth box that has the **smallest area** among all candidates that overlap that point.
The `assigned_area` tensor keeps track of the area of the object currently assigned to each location. A location update occurs only if the current box's area is smaller than the already assigned area (initialized to infinity).

$$\text{update\_mask} = \text{indices} \land (\text{box\_area} < \text{assigned\_area}_{\text{nonzero}})$$
The `final_mask` represents the coordinates that are updated by the current ground-truth box.

### 2. Target Tensors Creation

For every assigned positive location determined by `final_mask`, three types of target tensors are created:

#### A. Classification Targets (`cls_targets`)

For positive samples, the classification target tensor is updated to reflect the presence of an object of a specific class.
*   The class ID is used to index the correct channel in the classification target tensor, and a value of **1.0** is assigned to indicate a positive sample.
*   The classification targets for each level (P2-P5) are initialized as zero tensors with shape $(\text{Size}, \text{Size}, \text{num\_classes})$.

#### B. Regression Targets (`reg_targets`)

Regression targets represent the normalized distances from the center point $(\mathbf{x}_{\text{img}}, \mathbf{y}_{\text{img}})$ to the four sides of the assigned ground-truth box (Left, Top, Right, Bottom), relative to the feature map stride $S$.

The normalized distances are calculated as:
$$l = (\mathbf{x}_{\text{img}} - x_{\min}) / S$$
$$t = (\mathbf{y}_{\text{img}} - y_{\min}) / S$$
$$r = (x_{\max} - \mathbf{x}_{\text{img}}) / S$$
$$b = (y_{\max} - \mathbf{y}_{\text{img}}) / S$$
The target regression tensor is stacked as $\text{ltrb} = [l, t, r, b]$. This 4-channel tensor is what the regression head is trained to predict.

#### C. Centerness Targets (`ctr_targets`)

The centerness target is calculated only for positive samples and is designed to penalize locations far from the center of the assigned bounding box. This target is a scalar value (between 0 and 1) derived from the regression targets ($l, t, r, b$):

$$\text{Centerness} = \sqrt{\left(\frac{\min(l, r)}{\max(l, r) + 1e-6}\right) \times \left(\frac{\min(t, b)}{\max(t, b) + 1e-6}\right)}$$
.

A value close to 1 indicates the location is near the center, while a value close to 0 indicates it is near the edge, resulting in a lower weight during centerness loss calculation. The centerness target tensor is initialized as zeros with shape $(\text{Size}, \text{Size}, 1)$.

## Loss Functions 
The FCOS model uses three distinct loss functions, applied simultaneously, covering classification, bounding box regression, and centerness prediction. These losses are normalized by the total number of positive locations across all FPN levels.

### 1. Classification Loss: Focal Loss

The classification loss is calculated using **Focal Loss** ($\mathcal{L}_{\text{cls}}$), which addresses the imbalance between foreground and background samples.

The loss is defined based on the standard Binary Cross-Entropy (BCE) combined with a modulating factor derived from the prediction confidence ($p_t$).

**A. Binary Cross-Entropy (BCE) Calculation:**
The process begins by calculating the BCE between the prediction logits ($\text{cls\_out}$) and the binary class targets ($\text{cls\_target}$):
$$\text{BCE} = \text{tf.nn.sigmoid\_cross\_entropy\_with\_logits}(\text{labels}=\text{cls\_target}, \text{logits}=\text{cls\_out})$$.

**B. Modulating Factor Components:**
1.  **Probability of Target ($p_t$):** This represents the model's confidence in the assigned ground-truth class. It is based on the sigmoid-activated prediction ($\text{pred}$):
    $$p = \text{tf.sigmoid}(\text{cls\_out})$$.
    $$p_t = \begin{cases} \text{pred} & \text{if } \text{target} = 1.0 \\ 1.0 - \text{pred} & \text{if } \text{target} = 0.0 \end{cases}$$
    ($p_t$ is implemented using `tf.where`).

2.  **Alpha Factor ($\alpha$):** This weights positive and negative examples ($\alpha=0.25$ by default):
    $$\alpha_{\text{factor}} = \begin{cases} \alpha & \text{if } \text{target} = 1.0 \\ 1.0 - \alpha & \text{if } \text{target} = 0.0 \end{cases}$$.

3.  **Focal Weight:** This term reduces the loss contribution from easy (well-classified) examples using the focusing parameter $\gamma$ ($\gamma=2.0$ by default):
    $$\text{Focal\_Weight} = \alpha_{\text{factor}} \times (1.0 - p_t)^{\gamma}$$.

**C. Focal Loss per Element:**
$$\mathcal{L}_{\text{cls, element}} = \text{Focal\_Weight} \times \text{BCE}$$.

**D. Total Classification Loss:**
The final loss is the sum over all elements, normalized by the total number of positive samples ($\text{num\_pos}$):
$$\mathcal{L}_{\text{cls}} = \frac{\sum_{\text{all locations}} \mathcal{L}_{\text{cls, element}}}{\text{normalizer}}$$
where $\text{normalizer} = \max(1.0, \text{num\_pos})$. $\text{num\_pos}$ is the total number of positive locations across all FPN levels.

### 2. Regression Loss: Generalized Intersection over Union (GIoU) Loss

The bounding box regression loss uses the **GIoU Loss** ($\mathcal{L}_{\text{reg}}$). GIoU loss is calculated only over the locations identified as positive samples (foreground mask).

**A. Bounding Box Transformation:**
The predicted normalized LTRB distances ($\text{reg\_out}$) and the target LTRB distances ($\text{reg\_target}$) are converted back into absolute pixel coordinates $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ based on the feature map coordinates ($\text{coords}$) and the stride ($S$):

$$\text{Box} = [x - l \cdot S, y - t \cdot S, x + r \cdot S, y + b \cdot S]$$
where $l, t, r, b$ are the normalized distances, and $(x, y)$ are the center coordinates of the feature map point in image pixels. This is done separately for predicted boxes ($\text{pred\_box}$) and target boxes ($\text{target\_box}$).

**B. Calculation of IoU and Enclosing Area:**
1.  **Intersection Area ($\text{inter\_area}$):** Calculated as the overlap between the predicted bounding box ($\text{pred\_box}_{\text{fg}}$) and the target bounding box ($\text{target\_box}_{\text{fg}}$).
2.  **Union Area ($\text{union\_area}$):** Calculated as the sum of the areas of the predicted and target boxes minus the intersection area.
3.  **IoU:**
    $$\text{IoU} = \frac{\text{inter\_area}}{\text{union\_area} + 1e-6}$$.
4.  **Smallest Enclosing Box Area ($G$):** This is the area of the smallest box that encloses both the prediction and the target box.

**C. GIoU Loss Calculation:**
The GIoU metric is calculated first, where the second term penalizes large gaps between the union and the enclosing box area ($G$):
$$\text{GIoU} = \text{IoU} - \left(\frac{G - \text{union\_area}}{G + 1e-6}\right)$$.

The final GIoU Loss is defined as 1 minus the clipped GIoU value:
$$\mathcal{L}_{\text{reg, element}} = 1.0 - \text{tf.clip\_by\_value}(\text{GIoU}, -1.0, 1.0)$$.

**D. Total Regression Loss:**
The total regression loss is the sum of the element-wise losses over all positive locations, normalized by the total number of positive samples ($\text{total\_pos}$) found for classification:
$$\mathcal{L}_{\text{reg}} = \frac{\sum_{\text{positive locations}} \mathcal{L}_{\text{reg, element}}}{\text{total\_pos}}$$

### 3. Centerness Loss: Binary Cross-Entropy (BCE)

The Centerness loss ($\mathcal{L}_{\text{ctr}}$) ensures that locations closer to the center of the assigned ground-truth box produce higher centerness scores. This is calculated only for positive samples using standard BCE.

**A. Centerness Loss Calculation:**
The loss is computed using BCE between the centerness prediction logits ($\text{ctr\_pred}$) and the centerness targets ($\text{ctr\_target}$), restricted only to the positive coordinates ($\text{coord}$):
$$\mathcal{L}_{\text{ctr, element}} = \text{tf.nn.sigmoid\_cross\_entropy\_with\_logits}(\text{labels}=\text{ctr\_target}, \text{logits}=\text{ctr\_pred})$$.

**B. Total Centerness Loss:**
The total sum of this loss is normalized by the sum of the centerness target values of the positive samples ($\text{total\_num}$), which acts as a weighted normalizer:
$$\mathcal{L}_{\text{ctr}} = \frac{\sum_{\text{positive locations}} \mathcal{L}_{\text{ctr, element}}}{\max(1.0, \sum_{\text{positive locations}} \text{ctr\_target} + 1e-6)}$$.

### Total Loss

The final training objective is the sum of these three components:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{reg}} + \mathcal{L}_{\text{ctr}}$$. (Note: In some implementations, $\mathcal{L}_{\text{ctr}}$ may be multiplied by a coefficient, shown as $\text{coef} \times \text{ctr}$ in one instance, but the basic form is the sum).

## LR Scheduler ( Cosine Decay Schema)
The FCOS model utilizes a custom **Learning Rate (LR) Scheduler** featuring a warm-up phase followed by a cosine decay, and a structured **Training Schema** that incorporates distinct optimizers for the backbone and heads, gradient accumulation, and mixed-precision training.

### Learning Rate (LR) Scheduler

The custom FCOS model uses a learning rate scheduler designed to manage the LR across the training process, specifically incorporating a warm-up phase and a cosine decay component.

The scheduler relies on several defined parameters:
*   Initial LR for Cosine Scheduling: $n_{\text{start}}$ (implicitly set to $n_{\text{max}}$ in the implementation of the formula)
*   Minimum LR: $n_{\min} = 0.00000025$
*   Maximum LR: $n_{\max} = 0.00005$
*   Total training steps: $\text{total\_steps} = \text{epochs} \times \text{counts}$ (where $\text{counts} = 40000//4$)
*   Warm-up steps: $\text{warm} = 5000$
*   Cosine decay steps: $T = \text{total\_steps} - \text{warm}$

The `lr_scheduler` function operates based on the current step $t$:

**1. Warm-up Phase ($t \le \text{warm\_up}$):**
During the initial 5000 warm-up steps, the learning rate increases linearly from $n_{\text{start}}$ (implied initial minimum) up to $n_{\text{max}}$.
$$\text{lr} = n_{\text{start}} + \left(\frac{n_{\text{max}} - n_{\text{start}}}{\text{warm\_up}}\right) \times t$$

**2. Cosine Decay Phase ($t > \text{warm\_up}$):**
After the warm-up, the learning rate follows a cosine annealing curve, decaying from $n_{\text{max}}$ down to $n_{\min}$ over the remaining steps $T$.
First, the step counter is adjusted: $t' = t - \text{warm\_up}$.
The decay factor is calculated: $\text{decay} = t' / T$.
The LR is then computed using the cosine function:
$$\text{lr} = n_{\min} + \left(0.5 \times (n_{\text{max}} - n_{\min})\right) \times \left(1 + \cos(\text{decay} \times \pi)\right)$$



### Training Schema of the Custom FCOS Model

The training process is defined by specific configurations regarding optimization, precision, and handling of gradients across the model components.

#### 1. Optimization and Learning Rates
The model uses the **AdamW** optimizer. Crucially the Training schema is based on Continue Learning. A common Approach used here as From the 40k chunks 10k chunks used per 30 epoches. Then the next chunk is training.
The Lr schduler with Warmup is the responsible to avoid the Catastropic Forgetting for the on going chunk.
For the Final Fine Tunining we used, 
the backbone and the prediction heads are optimized separately using distinct learning rates, a common strategy for transfer learning or fine-tuning:
*   **Backbone Optimizer (`bb_optimizer`):** Uses a base learning rate of $0.000001$ ($\text{lr} = 1\times10^{-6}$).
*   **Heads Optimizer (`heads_optimizer`):** Uses a higher learning rate of $0.00001$ ($\text{lr} = 1\times10^{-5}$). This LR is 10 times the backbone LR, intended for fine-tuning the heads.
*   Both optimizers (`bb_optimizer` and `heads_optimizer`) utilize a weight decay of $5\times10^{-3}$.

#### 2. Mixed Precision Training
The model is configured to use **mixed precision** (`mixed_float16`) globally, which helps speed up training by performing calculations using lower precision floating point numbers. Both the backbone and heads optimizers are wrapped in `mixed_precision.LossScaleOptimizer` to manage potential underflow issues caused by mixed precision.

#### 3. Gradient Accumulation
To simulate a larger batch size or stabilize gradient estimation, the model uses **gradient accumulation** over $\text{acc\_num} = 8$ steps.
*   Gradients (`grads`) are calculated for the current batch.
*   If the batch counter (`counter`) is less than 8, the gradients are accumulated in `acc_grads` variables.
*   Once $\text{acc\_num}$ (8) batches have been processed, the accumulated gradients are retrieved, unscaled, and normalized by the accumulation count (`acc\_counter`).
*   The normalized gradients (`acc\_norm\_u`) are then clipped by norm $10.0$.
*   This shows the larger batch like intuition (`Batches = 8 x 4 = 32`)

#### 4. Differential Optimization for fine tuning on 40k records
The normalized gradients are applied separately to the backbone and the heads:
*   Gradients corresponding to variables belonging to the `FPN` layer (backbone) are stored in `bb_grads`.
*   Gradients corresponding to all other trainable variables (heads) are stored in `heads`.
*   The `bb_optimizer` applies gradients only to the backbone variables, and the `heads_optimizer` applies gradients only to the head variables.

#### 5. Training Fine tuning Process and Monitoring
The training iterates through a defined number of epochs (`epochs = 10`).
*   Losses ($\mathcal{L}_{\text{cls}}, \mathcal{L}_{\text{reg}}, \mathcal{L}_{\text{ctr}}$) are calculated for each batch. The total loss includes a coefficient ($\text{coef}=1.0$) for the centerness loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{reg}} + (\text{coef} \times \mathcal{L}_{\text{ctr}})$.
*   The losses are scaled before gradient calculation (`total_loss = bb_optimizer.get_scaled_loss(total_loss)`).
*   Metrics (Classification Loss, Regression Loss, Centerness Loss, and mean Average Precision (mAP)) are logged using `mlflow` at the end of each epoch.
*   The model includes an early stopping mechanism that checks the validation loss (`val\_cl`). If the validation loss does not decrease after 7 epochs (`pat >= 7`), training is halted, and a checkpoint is saved.

## NMS Per class Scoring
The model utilizes a **Per-Class Non-Maximum Suppression (NMS)** schema for post-processing predictions and defines a specific approach for **Validation Data Split**, relying on a subset of the primary training dataset.

### Non-Maximum Suppression (NMS) Schema

The NMS process is applied after the model generates raw predictions to filter out redundant bounding boxes and obtain the final set of detections. The model uses a technique called **Per-Class NMS**.

#### 1. Prediction Generation
First, raw model outputs (classification logits, regression distances, and centerness logits) from all FPN levels are aggregated:
*   **Classification Probability ($\text{cls\_prob}$):** Calculated by applying the sigmoid function to the classification output logits ($\text{cls\_prob} = \text{tf.nn.sigmoid}(\text{Class\_pred})$).
*   **Centerness Probability ($\text{ctr\_prob}$):** Calculated by applying the sigmoid function to the centerness output logits ($\text{ctr\_prob} = \text{tf.nn.sigmoid}(\text{Ctr\_pred})$).
*   **Final Score:** The combined score for a location and a class is the product of the classification probability and the centerness probability ($\text{scores} = \text{cls\_prob} \times \text{ctr\_prob}$).
*   **Location Filtering:** Only predictions with scores greater than $0.001$ are kept for further processing.
*   **Bounding Box Calculation:** Regression outputs (LTRB distances) are scaled by the FPN stride ($S$) and used with the feature map coordinates to determine the final bounding box coordinates ($\text{img\_y} - t, \text{img\_x} - l, \text{img\_y} + b, \text{img\_x} + r$).
*   All predictions are concatenated into `final_labels`, `final_scores`, and `final_bboxes`.

#### 2. Per-Class NMS Implementation
The `per_class_nms` function executes the Non-Maximum Suppression:
*   It iterates through each class $c$ (from 0 to $\text{num\_classes}-1$).
*   For each class, it filters the boxes and scores corresponding only to that class.
*   The `tf.image.non_max_suppression` function is applied to these class-specific boxes and scores using the following hyperparameters:
    *   **Maximum Output Size:** $\text{max\_per\_class} = 100$.
    *   **IoU Threshold:** $\text{iou\_threshold} = 0.75$.
    *   **Score Threshold:** $\text{score\_threshold} = 0.05$.
*   The selected indices from the NMS operation are used to retrieve the final boxes and scores for that class.
*   The results are combined across all classes to yield the final, non-redundant predictions.

### Validation Data Split Schema

The model uses a subset of the main training data for validation, ensuring that these records are not used in the gradient calculation for training.

*   **Total Records:** The primary training dataset contains 40,000 records, of which 38,800 are usable for training.
*   **Validation Set Source:** The validation dataset (`val_dataset`) is constructed by taking the **first 1200 records** from the training TFRecord file (`ttfrecord_path`).
*   **Training Set Exclusion:** Correspondingly, the main training dataset (`train_dataset`) is constructed by **skipping the first 1200 records** from the same TFRecord file.
*   This setup ensures that the $1200$ records used for validation are exclusive and not used during the main training loop, thus providing an unbiased measure of model performance.
*   The validation process measures Classification Loss, Regression Loss, Centerness Loss, and Mean Average Precision (mAP). The validation loss ($\text{val\_cl}$) is tracked, and an early stopping mechanism is implemented if the validation loss does not decrease after 7 epochs.

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
Open CMD (through directory of the code) --> Type **python -u CUSTOM_FCOS.py > logfile.txt 2>&1
The above is the simpliest way to run the code in the windows terminals for the efficieny over Jupyter Lab web page memory usage and memory spikes
and Python IDLE storing the output logs and other variables in the memory for a long period gives periodic memory spikes and consumption.
This execution method was chosen specifically to ensure stability during long training sessions where each epoch may take several hours.

