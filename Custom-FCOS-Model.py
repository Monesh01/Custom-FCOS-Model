import os
import glob
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import mlflow

mixed_precision.set_global_policy("mixed_float16")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Limit TensorFlow to only allocate 3000 MB (3 GB) of GPU memory
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3600)]
        )
        print("GPU memory limit set to 3600 MB")
    except RuntimeError as e:
        print("Error setting GPU memory limit:", e)

print("GPUs:", tf.config.list_physical_devices('GPU'))

def build_backbone(filters=32, training = False):
    inputs = layers.Input(shape=[512, 512, 3])
    print("Input layer created")

    
    def bottleneck_block(x, filters, downsample=False):
        
        shortcut = x
        strides = 2 if downsample else 1
        
        # 1x1 Conv
        x = layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False)(x)
        x = tfa.layers.GroupNormalization(groups = 32, axis = -1)(x)
        x = layers.Activation(tf.nn.elu)(x)
        #x = layers.PReLU(shared_axes=[1, 2])(x)
        
        # 3x3 Conv
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = tfa.layers.GroupNormalization(groups = 32, axis = -1)(x)
        x = layers.Activation(tf.nn.elu)(x)

        '''
        # 3 x 3 conv additional
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = tfa.layers.GroupNormalization(groups = 32, axis = -1)(x)
        x = layers.Activation(tf.nn.elu)(x)
        '''

        
       # 1x1 Conv (Expansion)
        x = layers.Conv2D(filters * 2, 1, padding='same', use_bias=False)(x)
        
        # Shortcut Path
        if downsample or shortcut.shape[-1] != filters * 2:
            shortcut = layers.Conv2D(filters * 2, 1, strides=strides, padding='same', use_bias=False)(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = tfa.layers.GroupNormalization(groups = 32, axis = -1)(x)
        x = layers.Activation(tf.nn.elu)(x)
        x = layers.Conv2D(filters * 2, 3, padding='same', use_bias=False)(x)
        x = tfa.layers.GroupNormalization(groups = 32, axis = -1)(x)
        x = layers.Activation(tf.nn.elu)(x)
        return x


    # Lateral (1x1) Convs: To reduce C2, C3, C4 to 28 channels
    lat_convs = [
        layers.Conv2D(128, 1, padding='same', use_bias=False, name=f'lat_conv_{i}')
        for i in range(4) # Corresponds to C2, C3, C4, C5
    ]
    # Lateral BN layers for M1
    lat_bns = [
        tfa.layers.GroupNormalization(groups = 32, axis = -1, name=f'lat_bn_{i}')
        for i in range(4)
    ]
    
    
    smooth_convs = [layers.Conv2D(128, 3, padding='same', use_bias=False, name=f'smooth_conv_{i}')
        for i in range(4)
    ]
    
    smooth_bns = [
        tfa.layers.GroupNormalization(groups = 32, axis = -1, name=f'smooth_bn_{i}')
        for i in range(4)
    ]
    

    fuse_bns = [
        tfa.layers.GroupNormalization(groups = 32, axis = -1)
        for i in range(3) 
    ]

    
    def FPN_generator(C, M_in=None, idx=0):
        m1 = lat_convs[idx](C)
        m1 = lat_bns[idx](m1)
        m1 = layers.LeakyReLU(alpha = 0.01)(m1)
        

        if M_in is not None:
            m = layers.Add()([m1, layers.UpSampling2D(size=(2, 2))(M_in)])
            m = fuse_bns[idx-1](m) 
            m = layers.LeakyReLU(alpha = 0.01)(m)
        else:
            m = m1
        p = smooth_convs[idx](m)
        p = smooth_bns[idx](p)
        p = layers.LeakyReLU(alpha = 0.01)(p)
        
        return p, m

    x = layers.Conv2D(filters, kernel_size=5, strides=2, padding='same', use_bias=False)(inputs)
    x = tfa.layers.GroupNormalization(groups = 16, axis = -1)(x)
    x = layers.Activation(tf.nn.elu)(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)
    c1 = x


    c2 = bottleneck_block(c1, filters = 32)
    c2 = bottleneck_block(c2, filters = 32)
    c3 = bottleneck_block(c2, filters = 64, downsample = True)
    c3 = bottleneck_block(c3, filters = 64, downsample = False)
    c4 = bottleneck_block(c3, filters = 128, downsample = True)
    c5 = bottleneck_block(c4, filters = 128, downsample = True)

    p5, m4 = FPN_generator(c5, idx = 0)   
    p4, m3 = FPN_generator(c4, m4, idx = 1)
    p3, m2 = FPN_generator(c3, m3, idx = 2)
    p2, m1 = FPN_generator(c2, m2, idx = 3) 

    return models.Model(inputs=inputs, outputs=[p2, p3, p4, p5], name='FPN')

class FCOSModel(tf.keras.Model):
    def __init__(self, num_classes):
        super(FCOSModel, self).__init__()
        self.num_classes = num_classes
        self.backbone = build_backbone()


        def create_head_tower():
            fill = 64
            return tf.keras.Sequential([
                layers.Conv2D(fill, 3, padding='same', use_bias=False),
                tfa.layers.GroupNormalization(groups=16, axis=-1),
                layers.Activation(tf.keras.activations.swish),
                layers.Conv2D(fill, 3, padding='same', use_bias = False),
                tfa.layers.GroupNormalization(groups=16, axis=-1),
                layers.Activation(tf.keras.activations.swish),
                layers.Conv2D(fill, 3, padding='same', use_bias=False),
                tfa.layers.GroupNormalization(groups=16, axis=-1),
                layers.Activation(tf.keras.activations.swish),
                layers.Conv2D(fill, 3, padding='same', use_bias=False),
                tfa.layers.GroupNormalization(groups=16, axis=-1),
                layers.Activation(tf.keras.activations.swish),
            ])
            
        self.scale = [tf.Variable(0.1, trainable=True, dtype = tf.float32) for _ in range(4)]

        self.cls_tower = create_head_tower()
        self.reg_tower = create_head_tower()
    
        p = 0.05
        bias_val = -np.log((1 - p) / p)
        self.cls_head = layers.Conv2D(self.num_classes, 1, 
                                      padding='same', 
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer=tf.constant_initializer(bias_val),
                                      activation=None,
                                      name="Classification")


        init = -np.log(100)
        self.reg_head = layers.Conv2D(4, 1, 
                                      padding='same', 
                                      activation= None, 
                                      name="Regression",
                                      bias_initializer=tf.constant_initializer(init))

        
        self.ctr_head = layers.Conv2D(1, 1, 
                                      padding='same', 
                                      activation = None, name="Centerness", 
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer=tf.constant_initializer(-2.0))
        



    def call(self, inputs, training=False): 

        features = self.backbone(inputs)
        
        cls_outputs, reg_outputs, ctr_outputs = [], [], []
        i = 0
        for f in features:
            
            cls_tower_output = self.cls_tower(f)
            raw_feat = self.reg_tower(f)
            raw_out = self.reg_head(raw_feat)
            reg_tower_output = tf.exp(raw_out * tf.cast(self.scale[i], dtype = raw_out.dtype))
            i += 1

            
            cls_outputs.append(self.cls_head(cls_tower_output))
            reg_outputs.append(reg_tower_output)
            ctr_outputs.append(self.ctr_head(raw_feat))


        return cls_outputs, reg_outputs, ctr_outputs


    def gen_tar(self, gt_boxes, num_classes, batch_size):
        batch_cls_p2, batch_cls_p3, batch_cls_p4, batch_cls_p5 = [], [], [], []
        batch_reg_p2, batch_reg_p3, batch_reg_p4, batch_reg_p5 = [], [], [], []
        batch_ctr_p2, batch_ctr_p3, batch_ctr_p4, batch_ctr_p5 = [], [], [], []



        def target_assignment(gt_boxes_img):
            p_tar = []     
            stride_ = []   
        
            for box in gt_boxes_img:
                _, x_min, y_min, x_max, y_max = box
                width = float(x_max - x_min)
                height = float(y_max - y_min)
                area = width * height


                if tf.less_equal(area, 4096):
                    stride_.append(4)
                    p_tar.append("p2")
                elif tf.less_equal(area, 16384):
                    stride_.append(8)
                    p_tar.append("p3")
                elif tf.less_equal(area, 65536):
                    stride_.append(16)
                    p_tar.append("p4")
                else:
                    stride_.append(32)
                    p_tar.append("p5")
            return p_tar, stride_
        
        def center_assignment(p, stride, gt_box, cls_targets, reg_targets, ctr_targets, assigned_area, num_classes):
    
            size = int(512/stride)
            x_coord, y_coord = tf.range(size, dtype = tf.float32), tf.range(size, dtype = tf.float32)
            x_grid, y_grid = tf.meshgrid(x_coord, y_coord)
            class_id, x_min, y_min, x_max, y_max  = gt_box
            x_img = (x_grid * stride) + stride / 2
            y_img = (y_grid * stride) + stride/2
            x_center = ((x_min + x_max) / 2)/ stride
            y_center = ((y_min + y_max) / 2)/ stride
            radius_x = tf.maximum(1.0, 1.0 * (x_max - x_min) / tf.cast(stride, tf.float32))
            radius_y = tf.maximum(1.0, 1.0 * (y_max - y_min) / tf.cast(stride, tf.float32))
            in_box_x = (x_img >= x_min) & (x_img <= x_max)
            in_box_y = (y_img >= y_min) & (y_img <= y_max)
            in_box = tf.logical_and(in_box_x, in_box_y)
            rad = tf.logical_and((tf.abs(x_grid - x_center ) <= radius_x), (tf.abs( y_grid - y_center )<=radius_y))
            indices = tf.logical_and(in_box, rad)
        
            l = (x_img - x_min) / stride
            t = (y_img - y_min) / stride
            r = (x_max - x_img) / stride
            b = (y_max - y_img) / stride
        
            area = (x_max - x_min) * (y_max - y_min)
            box_area = tf.fill((size, size), area)
            box_area = tf.cast(box_area, dtype = tf.float32)
        
            assigned_area_nonzero = tf.where(assigned_area == 0, tf.constant(float('inf')), assigned_area)
            
            update_mask = tf.logical_and(indices, box_area < assigned_area_nonzero)
        
            final_mask = tf.logical_and(indices, update_mask)
            
            area = tf.boolean_mask(box_area, final_mask)
            indices = tf.where(final_mask)
            indices = tf.cast(indices, tf.int32)
            num_positives = tf.shape(indices)[0]
            cls_ids = tf.cast(class_id, tf.int32) - 1
            cls_index = tf.fill((num_positives, 1), cls_ids)
            cls_indices = tf.concat([indices, cls_index], axis = 1)
            cls_updates = tf.ones(num_positives, dtype = tf.float32)
            '''
            cls = tf.cast(cls, dtype = tf.int32)
            cls = tf.boolean_mask(cls, final_mask)
            '''
            
            ltrb = tf.stack([l, t, r, b], axis = 2)
            reg = tf.boolean_mask(ltrb, final_mask)
            l, t, r, b = tf.unstack(reg, axis = 1)
            centerness = tf.sqrt((tf.minimum(l, r) / (tf.maximum(l, r) + 1e-6)) * (tf.minimum(t, b) / (tf.maximum(t, b) + 1e-6)))
            centerness = tf.expand_dims(centerness, axis = -1)
        
            
            cls_targets = tf.tensor_scatter_nd_update(cls_targets, cls_indices, cls_updates)
            reg_targets = tf.tensor_scatter_nd_update(reg_targets, indices, reg)
            ctr_targets = tf.tensor_scatter_nd_update(ctr_targets, indices, centerness)
            assigned_area = tf.tensor_scatter_nd_update(assigned_area, indices, area)
            #tf.print("Number of positives assigned:", tf.reduce_sum(tf.cast(cls_targets > 0, tf.int32)))
            return cls_targets, reg_targets, ctr_targets, assigned_area
        
            
        for img in range(batch_size):
        
            size_p2 = int(512/4)  #128
            cls_targets_p2 = tf.zeros((size_p2, size_p2, num_classes), dtype=tf.float32)
            reg_targets_p2 = tf.zeros((size_p2, size_p2, 4), dtype=tf.float32)
            ctr_targets_p2 = tf.zeros((size_p2, size_p2, 1), dtype=tf.float32)
            assigned_area_p2 = tf.cast(tf.fill([size_p2, size_p2], float('inf')), dtype=tf.float32)
            
        
            size_p3 = int(512/8) #64
            cls_targets_p3 = tf.zeros((size_p3, size_p3, num_classes), dtype=tf.float32)
            reg_targets_p3 = tf.zeros((size_p3, size_p3, 4), dtype=tf.float32)
            ctr_targets_p3 = tf.zeros((size_p3, size_p3, 1), dtype=tf.float32)
            assigned_area_p3 = tf.cast(tf.fill([size_p3, size_p3], float('inf')), dtype=tf.float32)
        
            size_p4 = int(512/16) #32
            cls_targets_p4 = tf.zeros((size_p4, size_p4, num_classes), dtype=tf.float32)
            reg_targets_p4 = tf.zeros((size_p4, size_p4, 4), dtype=tf.float32)
            ctr_targets_p4 = tf.zeros((size_p4, size_p4, 1), dtype=tf.float32)
            assigned_area_p4 = tf.cast(tf.fill([size_p4, size_p4], float('inf')), dtype=tf.float32)


            size_p5 = int(512/32) #16
            cls_targets_p5 = tf.zeros((size_p5, size_p5, num_classes), dtype=tf.float32)
            reg_targets_p5 = tf.zeros((size_p5, size_p5, 4), dtype=tf.float32)
            ctr_targets_p5 = tf.zeros((size_p5, size_p5, 1), dtype=tf.float32)
            assigned_area_p5 = tf.cast(tf.fill([size_p5, size_p5], float('inf')), dtype=tf.float32)
        
            
            p_tar, stride_ = target_assignment(gt_boxes[img])

            for p, st, gt_box in zip(p_tar, stride_, gt_boxes[img]):
                
                match st:
                    case 4:
                        cls_targets_p2, reg_targets_p2, ctr_targets_p2, assigned_area_p2 = center_assignment(p, st, gt_box, cls_targets_p2, reg_targets_p2, ctr_targets_p2, assigned_area_p2, num_classes)
                    case 8:
                        cls_targets_p3, reg_targets_p3, ctr_targets_p3, assigned_area_p3 = center_assignment(p, st, gt_box, cls_targets_p3, reg_targets_p3, ctr_targets_p3, assigned_area_p3, num_classes)
                    case 16:
                        cls_targets_p4, reg_targets_p4, ctr_targets_p4, assigned_area_p4 = center_assignment(p, st, gt_box, cls_targets_p4, reg_targets_p4, ctr_targets_p4, assigned_area_p4, num_classes)
                    case 32:
                        cls_targets_p5, reg_targets_p5, ctr_targets_p5, assigned_area_p5 = center_assignment(p, st, gt_box, cls_targets_p5, reg_targets_p5, ctr_targets_p5, assigned_area_p5, num_classes)
            batch_cls_p2.append(cls_targets_p2)
            batch_cls_p3.append(cls_targets_p3)
            batch_cls_p4.append(cls_targets_p4)
            batch_cls_p5.append(cls_targets_p5)
        
        
            batch_reg_p2.append(reg_targets_p2)
            batch_reg_p3.append(reg_targets_p3)
            batch_reg_p4.append(reg_targets_p4)
            batch_reg_p5.append(reg_targets_p5)
            
            batch_ctr_p2.append(ctr_targets_p2)
            batch_ctr_p3.append(ctr_targets_p3)
            batch_ctr_p4.append(ctr_targets_p4)
            batch_ctr_p5.append(ctr_targets_p5)          
        
        cls_p2 = tf.stack(batch_cls_p2)
        cls_p3 = tf.stack(batch_cls_p3)
        cls_p4 = tf.stack(batch_cls_p4)
        cls_p5 = tf.stack(batch_cls_p5)
        
        
        reg_p2 = tf.stack(batch_reg_p2)
        reg_p3 = tf.stack(batch_reg_p3)
        reg_p4 = tf.stack(batch_reg_p4)
        reg_p5 = tf.stack(batch_reg_p5)
        
        ctr_p2 = tf.stack(batch_ctr_p2)
        ctr_p3 = tf.stack(batch_ctr_p3)
        ctr_p4 = tf.stack(batch_ctr_p4)
        ctr_p5 = tf.stack(batch_ctr_p5)  


        
        cls_stack = [cls_p2, cls_p3, cls_p4, cls_p5]
        reg_stack = [reg_p2, reg_p3, reg_p4, reg_p5]
        ctr_stack = [ctr_p2, ctr_p3, ctr_p4, ctr_p5]

        return cls_stack, reg_stack, ctr_stack


    def losses(self, cls_out, reg_out, ctr_out, cls_stack, reg_stack, ctr_stack, batch_size, epoch):
        STRIDES = [4.0, 8.0, 16.0, 32.0] 

        def focal_loss(cls_out, cls_target, num_classes, alpha=0.25, gamma=2.0, eps=1e-5, normalize_pos=True):
            
            target = tf.cast(cls_target, tf.float32)
            pred = tf.sigmoid(cls_out)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=cls_out)
            pred = tf.clip_by_value(pred, eps, 1.0 - eps)  
            pt = tf.where(tf.equal(target, 1.0), pred, 1.0 - pred)
            alpha_factor = tf.where(tf.equal(target, 1.0),
                                    tf.constant(alpha, dtype=tf.float32),
                                    tf.constant(1.0 - alpha, dtype=tf.float32))
        

            focal_weight = alpha_factor * tf.pow(1.0 - pt, gamma) 
        
            loss_per_element = focal_weight * bce  
        

            loss = tf.reduce_sum(loss_per_element)

            num_pos = tf.reduce_sum(tf.cast(tf.greater(target, 0), tf.float32))  # scalar
            normalizer = tf.maximum(1.0, num_pos)  
            loss = loss 
            return loss, normalizer

        '''
        def iou_loss(reg_out, reg_target, cls_target, stride):  
            
            if cls_target.shape[-1] == 1:
                foreground_mask = cls_target[..., 0] > 0
            else:
                foreground_mask = tf.reduce_max(cls_target, axis=-1) > 0
        
            H, W = tf.shape(reg_target)[0], tf.shape(reg_target)[1]
            y_coords,x_coords = tf.meshgrid(tf.range(H), tf.range(W))
        
            # Scale coordinates to absolute pixel locations
            x_coords = tf.cast(x_coords, tf.float32) * stride + stride / 2
            y_coords = tf.cast(y_coords, tf.float32) * stride + stride / 2
        
            coords = tf.stack([x_coords, y_coords], axis=-1)  # [H, W, 2]
        
            if tf.reduce_sum(tf.cast(foreground_mask, tf.int32)) == 0:
                return tf.constant(0.0, dtype=tf.float32)

        

            def pred_ltrb_to_box(ltrb, coords):
                
                x = coords[..., 0]
                y = coords[..., 1]

                l, t, r, b = tf.unstack(tf.cast(ltrb, dtype = tf.float32), axis=-1)
                l = l * stride
                t = t * stride
                r = r * stride
                b = b * stride
                #tf.print(f"Prediction: {l, t, r, b}")
                x_min = x - l
                y_min = y - t
                x_max = x + r
                y_max = y + b
                return tf.stack([x_min, y_min, x_max, y_max], axis=-1)



            def tar_ltrb_to_box(ltrb, coords):
                
                x = coords[..., 0]
                y = coords[..., 1]
                l, t, r, b = tf.unstack(tf.cast(ltrb, dtype = tf.float32), axis=-1)
                l = l * stride
                t = t * stride
                r = r * stride
                b = b * stride
                #tf.print(f"Target :{l, t, r, b}")
                x_min = x - l
                y_min = y - t
                x_max = x + r
                y_max = y + b
                return tf.stack([x_min, y_min, x_max, y_max], axis=-1)

                
        
            pred_box = pred_ltrb_to_box(reg_out, coords)
            target_box = tar_ltrb_to_box(reg_target, coords)

            pred_box_fg = tf.boolean_mask(pred_box, foreground_mask)
            target_box_fg = tf.boolean_mask(target_box, foreground_mask)
        
            x_left = tf.maximum(pred_box_fg[:, 0], target_box_fg[:, 0])
            y_top = tf.maximum(pred_box_fg[:, 1], target_box_fg[:, 1])
            x_right = tf.minimum(pred_box_fg[:, 2], target_box_fg[:, 2])
            y_bottom = tf.minimum(pred_box_fg[:, 3], target_box_fg[:, 3])

            x_left_e = tf.minimum(pred_box_fg[:, 0], target_box_fg[:, 0])
            y_top_e  = tf.minimum(pred_box_fg[:, 1], target_box_fg[:, 1])
            x_right_e = tf.maximum(pred_box_fg[:, 2], target_box_fg[:, 2])
            y_bottom_e = tf.maximum(pred_box_fg[:, 3], target_box_fg[:, 3])


            g = tf.maximum(0.0, x_right_e - x_left_e) * tf.maximum(0.0, y_bottom_e - y_top_e)
            eps = 1e-6
            g = tf.maximum(g, eps)
            inter_area = tf.maximum(0.0, x_right - x_left) * tf.maximum(0.0, y_bottom - y_top)
            pred_area = (pred_box_fg[:, 2] - pred_box_fg[:, 0]) * (pred_box_fg[:, 3] - pred_box_fg[:, 1])
            target_area = (target_box_fg[:, 2] - target_box_fg[:, 0]) * (target_box_fg[:, 3] - target_box_fg[:, 1])
            union_area = pred_area + target_area - inter_area

            pred_area = tf.maximum(pred_area, eps)
            target_area = tf.maximum(target_area, eps)

            iou = inter_area / (union_area + 1e-6)
            giou = iou - ((g - union_area)/ (g + 1e-6))
            giou_clipped = tf.clip_by_value(giou, -1.0, 1.0)
            loss = 1.0 - giou_clipped
            return tf.reduce_sum(loss)


        '''
        def iou_loss(reg_out, reg_target, cls_tar, stride, feature_shape, H, W):  
            foreground_mask = tf.reduce_any(cls_tar > 0, axis = -1)
            y_coords, x_coords = tf.meshgrid(tf.range(H), tf.range(W))
        
            x_coords = tf.cast(x_coords, tf.float32) * stride + stride / 2
            y_coords = tf.cast(y_coords, tf.float32) * stride + stride / 2
        
            coords = tf.stack([x_coords, y_coords], axis=-1)  # [H, W, 2]
            coords = tf.reshape(coords, [-1, 2])
            B = feature_shape 
            coords = tf.tile(coords, [B, 1])
            
            if tf.reduce_sum(tf.cast(foreground_mask, tf.int32)) == 0:
                return tf.constant(0.0, dtype=tf.float32)

        

            def pred_ltrb_to_box(ltrb, coords):
                
                x = coords[..., 0]
                y = coords[..., 1]

                l, t, r, b = tf.unstack(tf.cast(ltrb, dtype = tf.float32), axis=-1)
                l = l * stride
                t = t * stride
                r = r * stride
                b = b * stride
                #tf.print(f"Prediction: {l, t, r, b}")

                x_min = x - l
                y_min = y - t
                x_max = x + r
                y_max = y + b
                return tf.stack([x_min, y_min, x_max, y_max], axis=-1)



            def tar_ltrb_to_box(ltrb, coords):
                
                x = coords[..., 0]
                y = coords[..., 1]
                l, t, r, b = tf.unstack(tf.cast(ltrb, dtype = tf.float32), axis=-1)
                l = l * stride
                t = t * stride
                r = r * stride
                b = b * stride
                #tf.print(f"Target :{l, t, r, b}")
                x_min = x - l
                y_min = y - t
                x_max = x + r
                y_max = y + b
                return tf.stack([x_min, y_min, x_max, y_max], axis=-1)

                
        
            pred_box = pred_ltrb_to_box(reg_out, coords)
            target_box = tar_ltrb_to_box(reg_target, coords)

            pred_box_fg = tf.boolean_mask(pred_box, foreground_mask)
            target_box_fg = tf.boolean_mask(target_box, foreground_mask)
        
            x_left = tf.maximum(pred_box_fg[:, 0], target_box_fg[:, 0])
            y_top = tf.maximum(pred_box_fg[:, 1], target_box_fg[:, 1])
            x_right = tf.minimum(pred_box_fg[:, 2], target_box_fg[:, 2])
            y_bottom = tf.minimum(pred_box_fg[:, 3], target_box_fg[:, 3])

            x_left_e = tf.minimum(pred_box_fg[:, 0], target_box_fg[:, 0])
            y_top_e  = tf.minimum(pred_box_fg[:, 1], target_box_fg[:, 1])
            x_right_e = tf.maximum(pred_box_fg[:, 2], target_box_fg[:, 2])
            y_bottom_e = tf.maximum(pred_box_fg[:, 3], target_box_fg[:, 3])


            g = tf.maximum(0.0, x_right_e - x_left_e) * tf.maximum(0.0, y_bottom_e - y_top_e)
            eps = 1e-6
            g = tf.maximum(g, eps)
            inter_area = tf.maximum(0.0, x_right - x_left) * tf.maximum(0.0, y_bottom - y_top)
            pred_area = (pred_box_fg[:, 2] - pred_box_fg[:, 0]) * (pred_box_fg[:, 3] - pred_box_fg[:, 1])
            target_area = (target_box_fg[:, 2] - target_box_fg[:, 0]) * (target_box_fg[:, 3] - target_box_fg[:, 1])
            union_area = pred_area + target_area - inter_area

            pred_area = tf.maximum(pred_area, eps)
            target_area = tf.maximum(target_area, eps)

            iou = inter_area / (union_area + 1e-6)
            giou = iou - ((g - union_area)/ (g + 1e-6))
            giou_clipped = tf.clip_by_value(giou, -1.0, 1.0)
            loss = 1.0 - giou_clipped
            return tf.reduce_sum(loss)

        def bce_loss(ctr_pred, ctr_target, cls_target, epochs, from_logits=True):
            coord = tf.where(ctr_target > 0)
        
            if tf.shape(coord)[0] == 0:
                return tf.constant(0.0), tf.constant(0.0)       #, tf.constant(0.0)
        
            ctr_pred = tf.gather_nd(ctr_pred, coord)
            ctr_target = tf.gather_nd(ctr_target, coord)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=ctr_target, logits=ctr_pred)
            return tf.reduce_sum(bce), (tf.reduce_sum(ctr_target) + 1e-6)       #tf.cast(tf.shape(coord)[0], tf.float32)  


        
        num_classes = self.num_classes
        '''
        cls_loss, reg_loss, ctr_loss, num_tot = [], [], [], []
        for i in range(batch_size):
            for level in range(4):
                cls_target = tf.gather(cls_stack[level],i)
                cls_pred = tf.gather(cls_out[level],i)
                loss, num = focal_loss(cls_pred, cls_target, num_classes, batch_size)
                cls_loss.append(loss)
                num_tot.append(num)
        
        total_cls_loss = tf.reduce_sum(tf.stack(cls_loss))
        total_pos = tf.maximum(tf.reduce_sum(tf.stack(num_tot)), 1.0) # Denominator
        cls = total_cls_loss / total_pos
        print(f"Num of pos center : {total_pos}")
        print(f"Classification Loss : {cls}")

        
        reg_loss_list = []
        
        for i in range(batch_size):
            for level in range(4):
                reg_target = tf.gather(reg_stack[level], i)
                reg_pred = tf.gather(reg_out[level], i)
                cls_target = tf.gather(cls_stack[level], i)
                stride = STRIDES[level]
                loss = iou_loss(reg_pred, reg_target, cls_target, stride)
        
                loss = tf.where(tf.math.is_finite(loss), loss, 0.0)
        
                reg_loss_list.append(loss)
        total_reg_loss = tf.reduce_sum(tf.stack(reg_loss_list))
        reg = total_reg_loss / total_pos
        print(f"Regression Loss : {reg}")


        ctr_num = []
        for i in range(batch_size):
            for level in range(4):
                #ctr_target = tf.gather(ctr_stack[level], i) 
                ctr_target = tf.gather(ctr_stack[level], i)
                ctr_pred = tf.gather(ctr_out[level], i)     
                cls_target = tf.gather(cls_stack[level], i)  
                loss, num = bce_loss(ctr_pred, ctr_target, cls_target, epoch)
                
                loss = tf.where(tf.math.is_finite(loss), loss, 0.0)
                num = tf.where(tf.math.is_finite(num), num, 0.0)
                ctr_loss.append(loss)
                ctr_num.append(num)
        ctr = tf.reduce_sum(tf.stack(ctr_loss))
        total_num = tf.reduce_sum(tf.stack(ctr_num))
        tot_num = tf.maximum(total_num, 1.0) 
        print(f"Centerness Loss before Normalization: {ctr}")
        ctr = ctr / tot_num
        print(f"Num of target weighted : {tot_num}")
        print(f"Centerness Loss : {ctr}")
        '''
        num_classes = self.num_classes
        cls_target = tf.concat([tf.reshape(i, [-1, 1]) for i in cls_stack], axis=0)
        cls_pred = tf.concat([tf.reshape(i, [-1, 1]) for i in cls_out], axis=0)
        cls_loss, num = focal_loss(cls_pred, cls_target, num_classes)
        total_pos = tf.maximum(num, 1.0) # Denominator
        cls = cls_loss / total_pos
        print(f"Num of pos center : {total_pos}")
        print(f"Classification Loss : {cls}")



        
        reg_loss_list = []
        
        for level in range(4):
            feature_shape = tf.shape(reg_stack[level])[0]
            H = tf.shape(reg_stack[level])[1]
            W = tf.shape(reg_stack[level])[2]
            reg_target = tf.reshape(reg_stack[level], [-1, 4])
            reg_pred = tf.reshape(reg_out[level], [-1, 4])
            cls_tar = tf.reshape(cls_stack[level], [-1, 1])
            stride = STRIDES[level]
            loss = iou_loss(reg_pred, reg_target, cls_tar, stride, feature_shape, H, W)
        
            loss = tf.where(tf.math.is_finite(loss), loss, 0.0)
        
            reg_loss_list.append(loss)
        total_reg_loss = tf.reduce_sum(tf.stack(reg_loss_list))
        reg = total_reg_loss / total_pos
        print(f"Regression Loss : {reg}")


        '''
        #ctr_target = tf.gather(ctr_stack[level], i) 
        ctr_target = tf.concat([tf.reshape(i, [-1, 1]) for i in ctr_stack], axis = 0)
        ctr_pred = tf.concat([tf.reshape(i, [-1, 1]) for i in ctr_out], axis = 0)  
        ctr_loss, num = bce_loss(ctr_pred, ctr_target, cls_target, epoch)
        ctr_loss = tf.where(tf.math.is_finite(ctr_loss), ctr_loss, 0.0)
        num = tf.where(tf.math.is_finite(num), num, 0.0)
        num = tf.maximum(num, 1.0) 
        tf.print(f"Centerness Loss before Normalization: {ctr_loss}")
        ctr = ctr_loss / num
        tf.print(f"Num of target weighted : {num}")
        tf.print(f"Centerness Loss : {ctr}")
        '''

        #ctr_target = tf.gather(ctr_stack[level], i) 
        ctr_target = tf.concat([tf.reshape(i, [-1, 1]) for i in ctr_stack], axis = 0)
        ctr_pred = tf.concat([tf.reshape(i, [-1, 1]) for i in ctr_out], axis = 0)  
        top_val, top_index = tf.math.top_k(tf.reshape(ctr_target, [-1]), k = 10)
        print(f"Top 10 High quality target values : {top_val}")
        print(f"Top 10 High quality predicted : {tf.sigmoid(tf.gather(tf.reshape(ctr_pred, [-1]), top_index))}")
        ctr_loss, num = bce_loss(ctr_pred, ctr_target, cls_target, epoch)
        ctr_loss = tf.where(tf.math.is_finite(ctr_loss), ctr_loss, 0.0)
        num = tf.where(tf.math.is_finite(num), num, 0.0)
        num = tf.maximum(num, 1.0) 
        tf.print(f"Centerness Loss before Normalization: {ctr_loss}")
        ctr = ctr_loss / num
        print(f"Num of target weighted : {num}")
        print(f"Centerness Loss : {ctr}")

        return cls, reg, ctr



def augment(img):
    img = tf.cond(tf.random.uniform([]) > 0.5,
    lambda: tf.image.random_brightness(img, max_delta=0.1),
    lambda: img)
    img = tf.cond(tf.random.uniform([]) > 0.5,
    lambda: img + tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.01),
    lambda: img)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img
    
def parse_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'annot': tf.io.VarLenFeature(tf.float32),
        'len': tf.io.VarLenFeature(tf.int64),
    }

    parsed = tf.io.parse_single_example(example, feature_description)

    image = tf.io.decode_png(parsed['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  
    image = tf.image.resize(image, (512, 512))


    flat_values = tf.sparse.to_dense(parsed['annot'])
    row_lengths = tf.sparse.to_dense(parsed['len'])

    annot = tf.RaggedTensor.from_row_lengths(flat_values, row_lengths)
    annot = tf.cast(annot, tf.float32)       

    return augment(image), annot
    

def parse_tfrecord_val(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'annot': tf.io.VarLenFeature(tf.float32),
        'len': tf.io.VarLenFeature(tf.int64),
    }

    parsed = tf.io.parse_single_example(example, feature_description)

    image = tf.io.decode_png(parsed['image'], channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  
    image = tf.image.resize(image, (512, 512))


    flat_values = tf.sparse.to_dense(parsed['annot'])
    row_lengths = tf.sparse.to_dense(parsed['len'])
    print("No issue before parsing the annot into ragged")

    annot = tf.RaggedTensor.from_row_lengths(flat_values, row_lengths)
    annot = tf.cast(annot, tf.float32)       

    return image, annot

ttfrecord_path = r"C:\Users\Monesh\FCOS\tfrecords\train.tfrecord"
batch_size = 4
train_dataset = (
    tf.data.TFRecordDataset(ttfrecord_path)                          # Totally 40k records, 38800 usable for training
    .skip(1200)                                                      ## Skip 1200 for validation & .take(10000) per chunk and use full 38800 for fine tuning
    .shuffle(512, reshuffle_each_iteration=True)
    .map(parse_tfrecord, num_parallel_calls = 2)
    # .map(lambda img, annot : (augment(img), annot), num_parallel_calls = 2)
    .batch(batch_size, drop_remainder=True)
    .prefetch(1)
)
vtfrecord_path = r"C:\Users\Monesh\FCOS in Python\validation.tfrecord"
val_batch_size = 4
val_dataset = (
    tf.data.TFRecordDataset(ttfrecord_path)                          ## From 40k records of training dataset using 1200 for the validation 
    .take(1200)
    .map(parse_tfrecord_val, num_parallel_calls = 2)
    .batch(val_batch_size, drop_remainder=True)
    .prefetch(1)
)


num_classes = 1
with tf.device('/GPU:0'):
    model = FCOSModel(num_classes = num_classes)
    lr = 0.000001
    bb_opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=5e-3)  ## Optimizer for backbone for fine tuning
    bb_optimizer = mixed_precision.LossScaleOptimizer(bb_opt)
    lr = 0.00001
    head_opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=5e-3)   ## Optimizer for heads for fine tuning 10x backbone lr 
    heads_optimizer = mixed_precision.LossScaleOptimizer(head_opt)

    '''
    lr = 0.0000025                                  ## Starting lr for Cosine scheduling
    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=5e-3)   ## Optimizer for heads for fine tuning 10x backbone lr 
    optimizer = mixed_precision.LossScaleOptimizer(opt)
    
    '''
    # Simulate FPN outputs for each level
    dummy_features = [
        tf.random.uniform((1, 128, 128, 128)),
        tf.random.uniform((1, 64, 64, 128)),
        tf.random.uniform((1, 32, 32, 128)),
    ]
    
    # Pass through towers to force build
    for f in dummy_features:
        _ = model.cls_tower(f)
        _ = model.reg_tower(f)
    

    dummy_img = tf.zeros([4, 512, 512, 3], dtype=tf.float32)
    

    
    dummy_gt = tf.ragged.constant([
        [[1.0, 50.0, 50.0, 300.0, 300.0]],
        [[1.0, 100.0, 100.0, 300.0, 300.0]],
        [[1.0, 20.0, 20.0, 50.0, 50.0]],
        [[1.0, 50.0, 50.0, 250.0, 250.0]],
        [[1.0, 50.0, 50.0, 300.0, 300.0]],
        [[1.0, 50.0, 50.0, 200.0, 200.0]],
        [[1.0, 50.0, 50.0, 150.0, 150.0]],
        [[1.0, 50.0, 50.0, 100.0, 100.0]]
    ], dtype=tf.float32)
    

    dense_gt = dummy_gt.to_tensor(default_value=-1.0)
    
    # 4. Force model build with dummy input
    _ = model(dummy_img, training=True)
    
    print("--- Building all model and optimizer variables on first batch ---")
    

    with tf.GradientTape(watch_accessed_variables=False) as tape:

        cls_preds, reg_preds, ctr_preds = model(dummy_img, training=True)
        cls_preds = [tf.cast(x, tf.float32) for x in cls_preds]
        reg_preds = [tf.cast(x, tf.float32) for x in reg_preds]
        ctr_preds = [tf.cast(x, tf.float32) for x in ctr_preds]
    

        cls_targets, reg_targets, ctr_targets = model.gen_tar(dummy_gt, num_classes, batch_size = 4)

        cls_targets = [tf.cast(x, tf.int32) for x in cls_targets]
        reg_targets = [tf.cast(x, tf.float32) for x in reg_targets]
        ctr_targets = [tf.cast(x, tf.float32) for x in ctr_targets]
    
        # Loss computation
        cls_loss, reg_loss, ctr_loss = model.losses(
            cls_preds, reg_preds, ctr_preds,
            cls_targets, reg_targets, ctr_targets,
            batch_size = 4, epoch = 1
        )
        total_loss = cls_loss + reg_loss + ctr_loss
        total_loss = bb_optimizer.get_scaled_loss(total_loss)
    
        grads = tape.gradient(total_loss, model.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]
        grads = [tf.clip_by_norm(g, clip_norm=10.0) for g in grads]
        grads = bb_optimizer.get_unscaled_gradients(grads)
        def _unwrap_var(v):

            return getattr(v, "variable", getattr(v, "_variable", v))
                
        vars_unwrapped = [_unwrap_var(v) for v in model.trainable_variables]
                
    
        grads_and_vars = [(g, v) for g, v in zip(grads, vars_unwrapped) if g is not None]
    
    print("--- Variable build complete. Starting main training. ---")


# model.load_weights(f"C:/Users/Monesh/FCOS/Saved_weights/Model_dec_10th first buffer 512 epoches 6 plus epoch_10")
# model.load_weights(f"C:/Users/Monesh/FCOS/Saved_weights/Model_dec_14th Third chunk buffer 512 plus map 0.75 epoch_2")
# model.load_weights(f"C:/Users/Monesh/FCOS/Saved_weights/Model_dec_14th 4th chunk buffer 512 plus map 0.75 epoch_{epoch}") #<-- Its 6th epoch is the optimal one 
model.load_weights(f"C:/Users/Monesh/FCOS/Saved_weights/Model_dec_15th Final fine tuning epoch_1")      ## <---- Final weights
counts = 40000//4


def safe_squeeze_last(x):
    if x.shape.rank is not None and x.shape[-1] == 1:
        return tf.squeeze(x, axis=-1)
    return x


def mAP(Class_pred, Reg_pred, Ctr_pred, annot, strides, num_classes, img_size = 512):
    def pred_coords(Class_pred, Reg_pred, Ctr_pred, strides, img_size=512):
        cls_all, score_all, box_all = [], [], []
    
        for i in range(len(strides)):
            stride = strides[i]
    
            cls_prob = tf.nn.sigmoid(Class_pred[i])                      # (H,W,C)
            ctr_prob = tf.nn.sigmoid(safe_squeeze_last(Ctr_pred[i]))     # (H,W)
            reg = Reg_pred[i]                                            # (H,W,4)
    
            H = tf.shape(cls_prob)[0]
            W = tf.shape(cls_prob)[1]
            C = tf.shape(cls_prob)[-1]
    
            y_grid, x_grid = tf.meshgrid(tf.range(H), tf.range(W), indexing="ij")
    
            ctr_prob = ctr_prob[..., None]
    
            scores = cls_prob * ctr_prob                                 # (H,W,C)
    

            scores = tf.reshape(scores, [-1, num_classes])                         # (N,C)
            reg = tf.reshape(reg, [-1, 4])
            x_grid = tf.reshape(x_grid, [-1])
            y_grid = tf.reshape(y_grid, [-1])
    

            mask = scores > 0.001
    
            idx = tf.where(mask)                                          # (K,2)
            loc_idx = idx[:, 0]
            cls_idx = idx[:, 1]
    
            score = tf.gather_nd(scores, idx)
            reg = tf.gather(reg, loc_idx)
            x = tf.gather(x_grid, loc_idx)
            y = tf.gather(y_grid, loc_idx)
    
            stride_f = tf.cast(stride, tf.float32)
    
            img_x = tf.cast(x, tf.float32) * stride_f + stride_f / 2
            img_y = tf.cast(y, tf.float32) * stride_f + stride_f / 2
    
            l, t, r, b = tf.unstack(reg * stride_f, axis=-1)
    
            box = tf.stack(
                [img_y - t, img_x - l, img_y + b, img_x + r],
                axis=-1
            )
    
            cls_all.append(cls_idx)
            score_all.append(score)
            box_all.append(box)
    
        return (
            tf.concat(cls_all, 0),
            tf.concat(score_all, 0),
            tf.concat(box_all, 0)
        )


    def tar_coords(annot):
        Class_id = tf.cast(annot[..., 0], tf.int32) - 1
        bbox = tf.stack([annot[..., 2], annot[..., 1], annot[..., 4], annot[..., 3]], axis = -1)
        return Class_id, bbox


    final_labels, final_scores, final_bboxes = pred_coords(Class_pred, Reg_pred, Ctr_pred, strides,img_size)
    tar_labels, tar_boxes = tar_coords(annot)


    if tf.size(final_scores) == 0:
        return 0.0
    def per_class_nms(boxes, scores, labels, num_classes, iou_thresh, score_thresh, max_per_class=100):
        final_boxes = []
        final_scores = []
        final_labels = []
    
        for c in range(num_classes):
            cls_mask = labels == c
    
            if not np.any(cls_mask):
                continue
    
            boxes_c = boxes[cls_mask]
            scores_c = scores[cls_mask]
    
            selected = tf.image.non_max_suppression(
                boxes_c,
                scores_c,
                max_output_size=max_per_class,
                iou_threshold=iou_thresh,
                score_threshold=score_thresh
            )
    
            boxes_keep = tf.gather(boxes_c, selected).numpy()
            scores_keep = tf.gather(scores_c, selected).numpy()
            labels_keep = np.full(len(boxes_keep), c, dtype=np.int32)
    
            final_boxes.append(boxes_keep)
            final_scores.append(scores_keep)
            final_labels.append(labels_keep)
    
        if len(final_boxes) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=np.int32)
    
        return (
            np.concatenate(final_boxes, axis=0),
            np.concatenate(final_scores, axis=0),
            np.concatenate(final_labels, axis=0)
        )
    iou_thresh = 0.75

    pred_boxes, pred_scores, pred_labels = per_class_nms(
    final_bboxes.numpy(),
    final_scores.numpy(),
    final_labels.numpy(),
    num_classes=num_classes,
    iou_thresh=iou_thresh,
    score_thresh=0.05,
    max_per_class=100
    )
    
    preds = [(s, l, b) for s, l, b in zip(pred_scores, pred_labels, pred_boxes)]
    
    tar_labels = tar_labels.numpy() if hasattr(tar_labels, 'numpy') else tar_labels
    tar_boxes = tar_boxes.numpy() if hasattr(tar_boxes, 'numpy') else tar_boxes
    targets = [(l, b) for l, b in zip(tar_labels, tar_boxes)]


    def iou_matrix(boxes1, boxes2):
        y1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        x1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        y2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        x2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area1 = (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 2] - boxes1[:, 0])
        area2 = (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 2] - boxes2[:, 0])
        union = area1[:, None] + area2[None, :] - inter
        return inter / np.clip(union, 1e-6, None)
    def compute_ap_single_class(preds, gt_boxes, iou_thresh=0.45):
    
        if len(gt_boxes) == 0:
            return 0.0
        if len(preds) == 0:
            return 0.0
    
        # sort predictions by score
        preds = sorted(preds, key=lambda x: x[0], reverse=True)
    
        pred_boxes = np.stack([p[1] for p in preds])
        gt_boxes   = np.stack(gt_boxes)
    
        ious = iou_matrix(pred_boxes, gt_boxes)
    
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        gt_used = np.zeros(len(gt_boxes), dtype=bool)
    
        for i in range(len(preds)):
            valid = np.where(~gt_used)[0]
            if len(valid) == 0:
                fp[i] = 1
                continue
    
            best_idx = valid[np.argmax(ious[i, valid])]
            best_iou = ious[i, best_idx]
    
            if best_iou >= iou_thresh:
                tp[i] = 1
                gt_used[best_idx] = True
            else:
                fp[i] = 1
    
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
    
        recall = tp_cum / (len(gt_boxes) + 1e-6)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    
        recall = np.concatenate(([0.0], recall))
        precision = np.concatenate(([1.0], precision))

        return np.trapz(precision, recall)

    def compute_map(preds, gts, num_classes, iou_thresh=iou_thresh):
        if len(gts) == 0 or len(preds) == 0:
            return 0.0
    
        APs = []
    
        for c in range(0, num_classes):  # classes start from 0
            preds_c = [(s, b) for s, l, b in preds if l == c]
            gts_c   = [b for l, b in gts if l == c]
    
            AP_c = compute_ap_single_class(preds_c, gts_c, iou_thresh)
            APs.append(AP_c)
        return float(np.mean(APs))
    return compute_map(preds, targets, num_classes, iou_thresh)


best_loss = float('inf')
n_min = 0.00000025
n_max = 0.000005

        
steps = 0
cls_log = []
reg_log = []
ctr_log = []
batch_log = []
epoch_log = []
epochs = 10
pat = 0
val_best = 100000
total_steps = epochs * counts
warm = 5000
T = total_steps - warm

coef = 1.0
acc_num = 8
counter = 0
val_counter = 0
print("Training Started")
acc_grads = [tf.Variable(tf.zeros_like(v), trainable=False) for v in model.trainable_variables]
acc_counter = [tf.Variable(0.0, trainable=False) for v in model.trainable_variables]


def _unwrap_var(v):
    return getattr(v, "variable", getattr(v, "_variable", v))


def lr_scheduler(n_start, n_min, n_max, t, T, warm_up):
    if t <= warm_up:
        lr = n_start + ((n_max - n_start)*(t / warm_up))
    else:
        t -= warm_up
        decay = t / T
        lr = n_min + ((0.5*(n_max - n_min))*(1 + tf.math.cos(tf.constant(decay * np.pi))))
    return lr

def snapshot_vars(stage):
    return set(v.name for v in model.trainable_variables), stage


vars_unwrapped = [_unwrap_var(v) for v in model.trainable_variables]
bb_unwrapped = set(_unwrap_var(v).ref() for v in model.get_layer("FPN").trainable_variables)

import gc
from tqdm import tqdm
with mlflow.start_run(run_name = "Custom FCOS Implementation"):
    with tf.device('/GPU:0'):
        mlflow.log_param("Optimizer", "AdamW")
        mlflow.log_param("Number of Objects", 1)
        mlflow.log_param("No of Epoch", epochs)
        mlflow.log_param("Warm Up steps", warm)
        mlflow.log_param("Training Steps", T)

        for epoch in range(1, epochs + 1):
            
            epoch_time = datetime.now()
            v_loss = 0.0
            batch = 1
            p = 0
            epoch_loss = 0.0
            cls_bat, reg_bat, ctr_bat = [], [], []
            
            
            for inputs, gt_boxes in tqdm(train_dataset):
                
                print(f"Epoch : {epoch}, Batch :{batch}")
                batch += 1

                batch_time = datetime.now()
                before, stage_before = snapshot_vars("Before model call")
        
        
                with tf.GradientTape() as tape:
                    fors_time = datetime.now()
    
                    cls_preds, reg_preds, ctr_preds = model(inputs, training=True)
                    for_time = datetime.now()
                    print(f"Time Taken for Forward Pass {epoch} : {((for_time - fors_time).total_seconds())} seconds")
                    cls_preds = [tf.cast(x, tf.float32) for x in cls_preds]
                    reg_preds = [tf.cast(x, tf.float32) for x in reg_preds]
                    ctr_preds = [tf.cast(x, tf.float32) for x in ctr_preds]
    
                    cls_targets, reg_targets, ctr_targets = model.gen_tar(gt_boxes, num_classes, batch_size)
                    cls_targets = [[tf.cast(x, tf.int32) for x in level] for level in cls_targets]
                    reg_targets = [[tf.cast(x, tf.float32) for x in level] for level in reg_targets]
                    ctr_targets = [[tf.cast(x, tf.float32) for x in level] for level in ctr_targets]

                    
                    cls, reg, ctr = model.losses(cls_preds, reg_preds, ctr_preds, cls_targets, reg_targets, ctr_targets, batch_size, epoch)
                    cls_bat.append(cls)
                    reg_bat.append(reg)
                    ctr_bat.append(ctr)
                    total = cls + reg + (coef * ctr)
                    total_loss = bb_optimizer.get_scaled_loss(total)
                    
                #after, stage_after = snapshot_vars("After model call")

                grads = tape.gradient(total_loss, model.trainable_variables)
                grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, model.trainable_variables)]


                
                
                if counter < acc_num:
                    
                    for i, g in enumerate(grads):
                        if g is not None:
                            acc_grads[i].assign_add(g)
                            acc_counter[i].assign_add(1.0)
                        
                    counter += 1
                else:
                    acc_unscaled = bb_optimizer.get_unscaled_gradients(acc_grads)
                    acc_norm = [tf.math.divide_no_nan(grad_sum, count) for grad_sum, count in zip(acc_unscaled, acc_counter)]
                    acc_norm_u = [tf.clip_by_norm(g, clip_norm=10.0) for g in acc_norm]
                    #grads_and_vars = [(g, v) for g, v in zip(acc_norm, vars_unwrapped) if g is not None]
                    bb_grads = []
                    heads = []
                    

                    for g,v in zip(acc_norm_u, vars_unwrapped):
                        if g is None:
                            continue
                        elif v.ref() in bb_unwrapped:
                            bb_grads.append((g, v))
                        else:
                            heads.append((g, v))
                    bb_optimizer.apply_gradients(bb_grads)
                    heads_optimizer.apply_gradients(heads)
                    for i in range(len(acc_grads)):
                        acc_grads[i].assign(tf.zeros_like(acc_grads[i]))
                        acc_counter[i].assign(0.0)
                    counter = 0

                
                steps += 1
                #new_lr = lr_scheduler(lr, n_min, n_max, steps, T, warm)
                #optimizer.learning_rate.assign(new_lr)
                ## Here the Lr scheduler used during chunks training in the curriculum basis 
            mlflow.log_metric("Classification Loss", np.mean(cls_bat), step = epoch)
            mlflow.log_metric("Regression Loss", np.mean(reg_bat), step = epoch)
            mlflow.log_metric("Centerness Loss", np.mean(ctr_bat), step = epoch)
            #mlflow.log_metric("Total_loss", np.mean(total_bat), step = steps)
            #model.save_weights(f"Saved_weights/Model_nov_30th_night_acc_second 5k_epoches_{epoch}")
            #model.save_weights(f"C:/Users/Monesh/FCOS/Saved_weights/Model_dec_14th 4th chunk buffer 512 plus map 0.75 epoch_{epoch}")
            model.save_weights(f"C:/Users/Monesh/FCOS/Saved_weights/Model_dec_15th night Final fine tuning epoch_{epoch}")
            cur_time = datetime.now()

                
            mlflow.log_metric("Epoch Loss", np.mean([np.mean(cls_bat), np.mean(reg_bat), np.mean(ctr_bat)]), step = epoch)

            cur_time = datetime.now()
            print(f"Time Taken for Epoch {epoch} : {((cur_time - epoch_time).total_seconds())/60} minutes")
            print(f"\nEnd of the epoch {epoch}...........")
            print("-----------------------------------------------------------------")
            
            v_loss = 0.0
            ap = 0.0
            val_cls, val_reg, val_ctr, val_loss, meanAP = [], [], [], [], []
            val_st = datetime.now()
            for img, bbox in tqdm(val_dataset):
                
                A_, B_, C_ = model(img, training=False)
                A_ = [tf.cast(x, tf.float32) for x in A_]
                B_ = [tf.cast(x, tf.float32) for x in B_]
                C_ = [tf.cast(x, tf.float32) for x in C_]
                a_, b_, c_ = model.gen_tar(bbox, 1, val_batch_size)
                a_ = [tf.cast(x, tf.int32) for x in a_]
                b_ = [tf.cast(x, tf.float32) for x in b_]
                c_ = [tf.cast(x, tf.float32) for x in c_]
                cls_, reg_, ctr_ = model.losses(A_, B_, C_, a_, b_, c_, val_batch_size, 1)
                a_ = [tf.cast(x, tf.float32) for x in a_]
                strides = [4.0, 8.0, 16.0, 32.0]
                aps = []
                for i in range(val_batch_size):
                    annot = bbox[i].merge_dims(0, 1)
                    annot = tf.reshape(annot, [-1, 5])
                    all_cls, all_reg, all_ctr = [], [], []
                    for j in range(len(strides)):
                        all_cls.append(A_[j][i])
                        all_reg.append(B_[j][i])
                        all_ctr.append(C_[j][i])
                    ap = mAP(all_cls, all_reg, all_ctr, annot, strides = strides, num_classes = 1)
                    aps.append(ap)
                meanAP.append(np.mean(aps))
                val_cls.append(cls_)
                val_reg.append(reg_)
                val_ctr.append(ctr_)
            val_cur = datetime.now()

            mlflow.log_metric("Validation Classification Loss", np.mean(val_cls), step = epoch)
            mlflow.log_metric("Validation Regression Loss", np.mean(val_reg), step = epoch)
            mlflow.log_metric("Validation Centerness Loss", np.mean(val_ctr), step = epoch)
            mlflow.log_metric("MAP", np.mean(meanAP), step = epoch)            
            mlflow.log_metric("Validation Loss", np.mean([np.mean(val_cls), np.mean(val_reg), np.mean(val_ctr)]), step = epoch)
            print(f"MAP : {np.mean(meanAP)}")
            val_cl = np.mean([np.mean(val_cls), np.mean(val_reg), np.mean(val_ctr)])
            if val_cl < val_best:
                val_best = val_cl
                pat = 0
            else:
                pat += 1
            if pat >= 7:
                print("Validation loss explodes")
                model.save_weights(f"C:/Users/Monesh/FCOS/Saved_weights/Model_dec 10th early checkpoint_epoch_{epoch}")
                break

            print(f"Validation time : {(val_cur - val_st).total_seconds()}")
            print("End of Validation")
            print("-----------------------------------------------------------------")
            gc.collect()

model.save_weights(f"C:/Users/Monesh/FCOS/Saved_weights/Model_dec_15th final epochs") 
