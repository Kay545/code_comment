import cv2
import copy
import math 
import numpy as np
import random 

class ImageEnhancement(object):
    def __init__(self,opt):
        self.opt = opt 
        self.input_w,self.input_h = self.opt.input_size 
    


    def __get_transform_matrix(self,img_shape, new_shape, degrees, scale, shear, translate):
        new_height, new_width = new_shape
        # Center
        C = np.eye(3)
        C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y transla ion (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
        return M, s


    def random_affine(self,img,labels,degrees=0.0,translate=0.1,scale=0.5,shear=0.0):
        '''随机进行仿射变换'''
        n = len(labels)
        
        M, s = self.__get_transform_matrix(img.shape[:2], (self.input_h,self.input_w), degrees, scale, shear, translate)
        if (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(self.input_w,self.input_h), borderValue=(114, 114, 114))
        # Transform label coordinates
        if n:
            new = np.zeros((n, 4))

            xy = np.ones((n * 4, 3))
            xy[:, :2] = labels[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, self.input_w)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, self.input_h)

            # filter candidates
            i = self.__box_candidates(labels[:, 0:4].T * s, new.T, area_thr=0.1)
            labels = labels[i]
            labels[:, 0:4] = new[i]
        return img,labels 
            
    def __box_candidates(self,box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
    
    def __augment_hsv(self,im, hgain=0.5, sgain=0.5, vgain=0.5):
        '''hsv 色域变换'''
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            # 分离通道
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            # 随机调整hsv之后重新组合通道
            # cv2.LUT(x, table)以x中的值为索引取根据table中的值
            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            # 将hsv格式转为BGR格式
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        return im_hsv 

    def __image_blur_option(self, image, boxes, min_w_h):
        '''高斯模糊'''
        # 设置模糊模式，blur_select :1--->背景模糊  blur_select:2 ---->整幅图模糊
        blur_select = random.randint(1, 2)
        if blur_select == 1:  # 背景模糊
            moment_image = copy.deepcopy(image)
            random_size = random.randint(2, 5)
            ksize = (random_size // 2) * 2 + 1
            # 先模糊整幅图像
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
            # 再将原图annotations的box区域填充回去
            for box in boxes:
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                image[ymin:ymax, xmin:xmax, :] = moment_image[ymin:ymax, xmin:xmax, :]
        else:
            # 对整幅图像进行blur模糊处理
            # 生成一个随机数，用于随机的kernel
            random_size = random.randint(2, 7)
            #如过图像中存在目标小于40的情况，则不进行模糊处理
            if min_w_h < 40:
                return image
            ksize = (random_size // 2) * 2 + 1
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image

    def __image_gray_transform(self, image):
        '''灰度变换'''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 进行灰度变换之后变成单通道，需要扩充通道到3通道
        image = np.expand_dims(image, axis=2)
        image = np.concatenate((image, image, image), axis=-1)
        return image

    def __image_gaussian_noise_transform(self, image):
        '''高斯噪声'''
        mean, var = 0, 0.001
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        image = np.clip(out, low_clip, 1.0)
        image = np.uint8(image * 255)
        return image


    def __image_erase_transform(self, image, boxes):
        '''随机擦除'''
        h, w, c = image.shape
        np.random.shuffle(boxes)
        box_number = len(boxes)
        #图像中最多有50%的box进行随机擦除处理
        erase_max_number = box_number // 2 if box_number >= 2 else 1
        # 初始化erase操作的个数
        erase_count = 0
        for box in boxes:
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            box_width = xmax - xmin
            box_height = ymax - ymin
            # 如果标注框的宽高都大于50个像素值则进行随机擦除处理
            if box_width > 100 and box_height > 200:
                erase_select = np.random.random() < 0.9
                if erase_count < erase_max_number and erase_select:
                    # 选取随机擦除区域的宽高
                    erase_w = random.randint(box_width // 5, box_width // 3)
                    erase_h = random.randint(box_height // 5, box_height // 3)
                    # 选取随机擦除的位置的右下角的坐标
                    erase_xmax = random.randint(xmin + erase_w, xmax - erase_w)
                    erase_ymax = random.randint(ymin + erase_h, ymax - erase_h)
                    # 计算左上角坐标
                    erase_xmin = erase_xmax - erase_w
                    erase_ymin = erase_ymax - erase_h
                    # 选取随机擦除模式，1: 填充为灰度图，2：做马赛克处理
                    mode_select = random.randint(1, 3)
                    if mode_select == 1:
                        erase_area = np.ones((erase_h, erase_w, c), dtype=np.uint8) * 128
                        # 将擦除区域填充到原图像对应的区域位置
                        image[erase_ymin:erase_ymax, erase_xmin:erase_xmax, :] = erase_area
                    else:
                        erase_area = image[erase_ymin:erase_ymax, erase_xmin:erase_xmax, :]
                        kernel_w = erase_w // 5 if erase_w // 5 % 2 else erase_w // 5 + 1
                        erase_area = cv2.blur(erase_area, (kernel_w, kernel_w))
                        image[erase_ymin:erase_ymax, erase_xmin:erase_xmax, :] = erase_area
        return image
    def image_enhance(self,image,boxes,min_w_h):
        #------------------色域变换---------------#
        #以50%的几率进行色域变换
        hsv_option = np.random.random() < 0.
        if hsv_option:
            #image = self.image_hsv_transform(image)
            image = self.__augment_hsv(image,hgain=0.015,sgain=0.4,vgain=0.7)
        #-----------------高斯模糊-----------------#
        # 以30%的几率进行高斯模糊处理
        blur = np.random.random() < 0.3
        if blur:
            image = self.__image_blur_option(image, boxes, min_w_h)
        #----------------灰度变换------------------#
        if not blur and not hsv_option:
            # 如果不进行色域变换/且不进行高斯模糊，则进行随机的灰度变换
            gray_option = np.random.random() < 0.1
            if gray_option:
                image = self.__image_gray_transform(image)
        #-----------------高斯噪声-----------------#
        gaussian_noise_option = np.random.random() < 0
        if gaussian_noise_option:
            image = self.__image_gaussian_noise_transform(image)
        #-----------------随机擦除处理----------------#
        erase_option = np.random.random() < 0
        if erase_option:
            image = self.__image_erase_transform(image, boxes)
        return image 