from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, \
                 mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super().__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + num_classes
        self.threshold          = 4

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_Mixup(image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)

        if len(box) != 0:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        y_true = self.get_target(box)
        return image, box, y_true

    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x > 0.5 and sub_y < 0.5:
            return [[0, 0], [1, 0], [0, -1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        else: 
            return [[0, 0], [-1, 0], [0, -1]]
        
    def get_target(self, targets):
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)

        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]
        y_true      = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]

        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            # 将anchor和target都转换到当前特征层
            anchors = np.array(self.anchors) / {0:32, 1:16, 2:8}[l]

            in_h, in_w = grid_shapes[l]

            batch_target = np.zeros_like(targets)
            batch_target[:, [0,2]] = targets[:, [0,2]] * in_w
            batch_target[:, [1,3]] = targets[:, [1,3]] * in_h
            batch_target[:, 4] = targets[:, 4]
            #-------------------------------------------------------#
            #   wh                          : num_true_box, 2
            #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
            #   anchors                     : 9, 2
            #   np.expand_dims(anchors, 0)  : 1, 9, 2
            #   
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : num_true_box, 9
            #-------------------------------------------------------#
            ratio_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratio_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratio = np.concatenate([ratio_of_anchors_gt, ratio_of_gt_anchors], axis=-1)
            max_ratio = np.max(ratio, axis=-1)

            for t, ratio in enumerate(max):
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True       
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    #----------------------------------------#
                    #   获得真实框属于哪个网格点
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #----------------------------------------#
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))
                    # 获得靠的最近的两个网格点，加上自身的偏移点
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[l][k, local_i, local_j] != 0:
                            if box_best_ratio[l][k, local_i, local_j] > ratio[mask]: 
                                y_true[l][k, local_i, local_j, :] = 0        # 使用ratio最小的，重置数据
                            else:
                                continue

                        #----------------------------------------#
                        #   取出真实框的种类
                        #----------------------------------------#
                        c = int(batch_target[t, 4])

                        #----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        #----------------------------------------#
                        y_true[l][k, local_i, local_j, 0:4] = batch_target[t, 0:4]
                        y_true[l][k, local_i, local_j, 4] = 1
                        y_true[l][k, local_i, local_j, c+5] = 1
                        #----------------------------------------#
                        #   获得当前先验框最好的比例
                        #----------------------------------------#
                        box_best_ratio[l][k, local_i, local_j] = ratio[mask]

        return y_true



    def get_random_data_with_Mixup(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()

        image = Image.open(line[0])
        image = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih = image.size
        h, w = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = iw * scale
            nh = ih * scale
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box) > 0:
                box[:, [0, 2]] = box[:[0,2]] * nw / iw + dx
                box[:, [1, 3]] = box[:[0,2]] * nh / ih + dy
                # discard invalid box
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 3][box[:, 3] >w] = w
                box[:, 4][box[:, 4] >h] = h

            return image_data, box
        
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box










    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.4, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape



    def rand(self, a=0, b=1):
        return np.random.rand() * (b-a) + a
    
    def yolo_dataset_collate(batch):
        images = []
        boxes = []
        y_trues = [[] for _ in batch[0][2]]
        for image, box, y_true in batch:
            images.append(image)
            boxes.append(box)
            for i, sub_y_true in enumerate(y_true):
                y_trues[i].append(sub_y_true)

        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        boxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in boxes]
        y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
        return images, boxes, y_trues
















