import os  
import cv2 
import copy 
import random 
import numpy as np 
from dataset.image_enhancement import ImageEnhancement
class DataList(object):
    def __init__(self,opt,datalist):
        self.opt = opt 
        self.datalist = datalist 
        self.input_w,self.input_h = opt.input_size 
        self.ImageEnhance = ImageEnhancement(opt)
        
    def __len__(self):
        '''训练数据数目'''
        return len(self.datalist)

    def __filter_unfit_box(self,b1,b2,min_size=8):
        '''
            b1:操作之后的box
            b2:原始的box
            作用:通过计算b1与b2的iou,过滤掉裁剪过大的box
        '''
        b1_x1,b1_y1,b1_x2,b1_y2 = b1[:,0],b1[:,1],b1[:,2],b1[:,3]
        b2_x1,b2_y1,b2_x2,b2_y2 = b2[:,0],b2[:,1],b2[:,2],b2[:,3]
        x1_insert = np.maximum(b1_x1,b2_x1)
        y1_insert = np.maximum(b1_y1,b2_y1)
        x2_insert = np.minimum(b1_x2,b2_x2)
        y2_insert = np.minimum(b1_y2,b2_y2)
        insert_area = (x2_insert-x1_insert) * (y2_insert-y1_insert)
        b1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        b2_area = (b2_x2-b2_x1)*(b2_y2-b2_y1)

        iou = insert_area /(b1_area+b2_area-insert_area+1e-16)
        select_index = iou>0.3
        labels = b1[select_index]
        labels = self.__filter_small_box(labels,filter_value=min_size)
        return labels 
        
    def __filter_small_box(self,labels,filter_value=8):
        '''
           过滤掉过于小的box
           标准 : 宽度或高度小于8
        '''
        filter_w_mask = (labels[:,2] - labels[:,0])>=filter_value
        labels = labels[filter_w_mask]
        filter_h_mask = (labels[:,3] - labels[:,1])>=filter_value
        labels = labels[filter_h_mask]
        return labels 
    
    def __random_cutting(self,img):
        '''对图像进行随机裁剪'''
        h,w,c = img.shape 
        # 获取w，h的随机裁剪值
        w_cut = int(w*random.randint(0,3)/10)
        h_cut = int(h*random.randint(0,3)/10)
        cut_img = img[h_cut:,w_cut:,:]
        return cut_img,[w_cut,h_cut] 
    
    def __img_padding(self,img):
        '''对图像进行padding操作得到符合网络输入大小的图像'''
        if np.random.random()<0.3 and self.opt.argument:
            # 随机裁剪
            cut_img,cut_value = self.__random_cutting(img)
        else:
            cut_img = img
            cut_value = [0,0] 
        # padding
        ch,cw,_ = cut_img.shape 
        i_rate = self.input_w*1.0/self.input_h
        n_rate = cw*1.0/ch
        p_w,p_h=0,0
        if n_rate<=i_rate:# 图像过窄,左右padding
            p_w = int(ch*i_rate-cw) 
        else:# 图像过宽，上下padding
            p_h = int(cw*1.0/i_rate - ch) 
        # 初始化一个新图像
        n_img = np.full((ch+p_h,cw+p_w,cut_img.shape[2]),114,dtype=np.uint8)
        n_img[p_h//2:(p_h//2+ch),p_w//2:(p_w//2+cw),:] = cut_img
        # resize 
        resize_scale = self.input_w*1.0/n_img.shape[1]
        n_img = cv2.resize(n_img,(self.input_w,self.input_h))
        
        return n_img,[p_w,p_h],cut_value,cut_img.shape[:2],resize_scale

    def __mendment_boxes(self,boxes,cut,cw,ch):
        '''对图像进行修正'''
        boxes[:,[0,2]] = boxes[:,[0,2]] - cut[0]
        boxes[:,[1,3]] = boxes[:,[1,3]] - cut[1]
        labels = copy.deepcopy(boxes)

        labels[:,[0,2]] = np.clip(labels[:,[0,2]],0,cw)      
        labels[:,[1,3]] = np.clip(labels[:,[1,3]],0,ch)     

        labels = self.__filter_unfit_box(labels,boxes,min_size=8)
        return labels 

    def __load_image(self,index):
        '''根据索引从数据列表中获取一张图像并解析其info'''
        try:
            data_info = self.datalist[index].strip().split()
        except:
            data_info = self.datalist[1].strip().split()
        # image_path 
        img_path = data_info[0]
        # 标注boxes
        box_info = np.array([np.array(list(map(int,box.split(',')))) for box in data_info[1:]])
        
        img = cv2.imread(img_path)
        assert img is not None,'image not found: {}'.format(img_path)
        h0,w0 = img.shape[:2]
        r = max(self.input_w/w0,self.input_h/h0)
        if r!=1:
            img= cv2.resize(img,(int(w0*r),int(h0*r)),
                        interpolation=cv2.INTER_AREA if r<1 and not self.opt.argument else cv2.INTER_LINEAR)
            #img= cv2.resize(img,(int(w0*r),int(h0*r)))
            try:
                box_info[:,:4] = box_info[:,:4] * r 
            except:
                print(img_path)
        # 对图像进行随机的镜像处理
        if random.random()<0.5 and self.opt.argument:
            img = cv2.flip(img,1)
            box_info[:,[0,2]] = img.shape[1] - box_info[:,[2,0]]
        
        
        return img,(h0,w0),img.shape[:2],box_info
    
    def __get_min_size(self,boxes):
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        min_size = min(min(w),min(h))
        return min_size
    
    def __load_one_image(self,index):
        '''获取单张图像即非Mosaic图像'''
        #加载图片并根据设定的输入大小与图片原大小的比例ratio进行resize(未做填充pad)
        img,(h0,w0),(h,w),boxes = self.__load_image(index)
        # cut + padding
        img,pad,cut,cut_shape,resize_scale = self.__img_padding(img)
        # box 修正
        c_w,c_h = cut_shape[1],cut_shape[0]
        labels = self.__mendment_boxes(boxes,cut,c_w,c_h)
        labels[:,[0,2]] = (labels[:,[0,2]]+pad[0]//2)*resize_scale
        labels[:,[1,3]] = (labels[:,[1,3]]+pad[1]//2)*resize_scale

        return img,labels 
    def __load_mosaic(self,index):
        '''获取mosaic图像'''
        # 从图像集合中再随机抽取3张图象作为组合
        indices = [index] + random.choices(
            range(0,len(self.datalist)),k=3
        )
        # 随机打乱
        random.shuffle(indices)
        # 随机获取mosaic的中心点
        xc = int(random.uniform(self.input_w//2,3*self.input_w//2))
        yc = int(random.uniform(self.input_h//2,3*self.input_h//2))
        # labels用于存放所有的box信息
        labels = [] 
        for idx,index in enumerate(indices):
            img,_,(h,w),boxes = self.__load_image(index)
          
            # 放置图像
            if idx==0:
                # 初始化大图
                img4 = np.full((self.input_h*2,self.input_w*2,img.shape[2]),114,dtype=np.uint8)
                # 设置大图上的位置(左上角) 
                # 计算左上角的贴图区域
                x1a,y1a,x2a,y2a = max(xc-w,0),max(yc - h, 0),xc,yc #xmin, ymin, xmax, ymax (large image)
                # 选取小图上的位置
                # 在小图上随机选取对应大小的区域
                c_w = x2a - x1a 
                c_h = y2a - y1a 
                # 随机计算裁切区域 的 xmin, ymin, xmax, ymax (small image)
                x1b = np.random.randint(w-c_w) if w>c_w else 0
                y1b = np.random.randint(h-c_h) if h>c_h else 0
                x2b = x1b + c_w 
                y2b = y1b + c_h 

            elif idx==1:# top right右上角
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.input_w * 2), yc
                # 选取小图上的位置
                # 在小图上随机选取对应大小的区域
                c_w = x2a - x1a 
                c_h = y2a - y1a 
                # 随机计算裁切区域 的 xmin, ymin, xmax, ymax (small image)
                x1b = np.random.randint(w-c_w) if w>c_w else 0
                y1b = np.random.randint(h-c_h) if h>c_h else 0
                x2b = x1b + c_w 
                y2b = y1b + c_h 
                
            elif idx == 2: # bottom left左下角
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.input_h * 2, yc + h)
                # 选取小图上的位置
                # 在小图上随机选取对应大小的区域
                c_w = x2a - x1a 
                c_h = y2a - y1a 
                # 随机计算裁切区域 的 xmin, ymin, xmax, ymax (small image)
                x1b = np.random.randint(w-c_w) if w>c_w else 0
                y1b = np.random.randint(h-c_h) if h>c_h else 0
                x2b = x1b + c_w 
                y2b = y1b + c_h 
            elif idx ==3:# 右下角
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.input_w * 2), min(self.input_h * 2, yc + h)
                # 选取小图上的位置
                # 在小图上随机选取对应大小的区域
                c_w = x2a - x1a 
                c_h = y2a - y1a 
                # 随机计算裁切区域 的 xmin, ymin, xmax, ymax (small image)
                x1b = np.random.randint(w-c_w) if w>c_w else 0
                y1b = np.random.randint(h-c_h) if h>c_h else 0
                x2b = x1b + c_w 
                y2b = y1b + c_h 
                
            # 将小图上截取的部分贴到大图上
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            # 计算小图到大图上时所产生的偏移，用来计算mosaic增强后的标签框的位置
            padw = x1a - x1b
            padh = y1a - y1b
            
            
            #重新调整标签的位置
            boxes[:,[0,2]] = boxes[:,[0,2]] + padw -1 
            boxes[:,[1,3]] =  boxes[:,[1,3]] + padh-1
            
            boxes_ = copy.deepcopy(boxes)
            boxes_[:,[0,2]] = np.clip(boxes_[:,[0,2]],x1a,x2a)
            boxes_[:,[1,3]] = np.clip(boxes_[:,[1,3]],y1a,y2a)
            
            fit_boxes = self.__filter_unfit_box(boxes_,boxes,min_size=8)
            labels.append(fit_boxes)
            
        labels = np.concatenate(labels,0)
        
        
        img4 = cv2.resize(img4,(self.input_w,self.input_h))
        labels[:,:4] = labels[:,:4] * 0.5
        labels = self.__filter_small_box(labels)
        return img4,labels 
    def __mixup(self,img1,labels1,img2,labels2):
        # mixup ratio alpha=beta=32.0
        r = np.random.beta(32.0,32.0)
        img = (img1*r + img2*(1-r)).astype(np.uint8)
        labels = np.concatenate([labels1,labels2],0)
        return img,labels
        
        
    def __box_test(self,img,box_info,test_name='test.jpg'):
        '''图像效果测试'''
        for box in box_info:
            xmin,ymin,xmax,ymax,label = box 
            cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,0),1)
        cv2.imwrite(test_name,img)
        
    def __getitem__(self,index):
        index = index%len(self.datalist)
        if index==0:
            random.shuffle(self.datalist)
        if self.opt.mosaic and np.random.random()<0.1:
            img,labels = self.__load_mosaic(index)
        else:
            img,labels = self.__load_one_image(index)
            if random.random()<0 and self.opt.argument:
                img2,labels2 = self.__load_one_image(np.random.randint(len(self.datalist)-1))
                img,labels = self.__mixup(img,labels,img2,labels2)
        
        # 以上得到了网络所需大小的图像，只是做了裁剪，padding，平移等操作，下面将进行其他方面的数据增强操作
        # 进行随机的放射变换
        # self.__box_test(img,labels)
        if random.random()<0.1 and self.opt.argument:
            img,labels = self.ImageEnhance.random_affine(img,labels)
        # 进行随机的数据增强处理
        n_l = len(labels)
        if n_l>0:
            min_size = self.__get_min_size(labels)
            img = self.ImageEnhance.image_enhance(img,labels,min_size)
            #self.__box_test(img,labels)
            img = np.array(img,dtype=np.float32)
            
            tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
            labels = np.array(labels,dtype=np.float32)
            boxes_copy = copy.deepcopy(labels)
            x1 = boxes_copy[:,0]
            y1 = boxes_copy[:,1]
            x2 = boxes_copy[:,2]
            y2 = boxes_copy[:,3]
            labels[:,0] = ((x1 + x2) / 2) / self.input_w
            labels[:,1] = ((y1 + y2) / 2) / self.input_h
            labels[:,2]=(x2-x1)/self.input_w
            labels[:,3]=(y2-y1)/self.input_h
            
            return tmp_inp,labels 
        else:
            img = np.array(img,dtype=np.float32)
            tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
            return tmp_inp,[] 
        
def yolo_dataset_collate(batch):
    images = [] 
    bboxes = [] 
    batch_idx = 0
    for img,box in batch:
        if box ==[]:
            continue
        b = np.ones((box.shape[0],1),dtype=np.float32) * batch_idx
        box = np.concatenate((b,box),axis=1)
        images.append(img)
        bboxes.append(box)
        batch_idx += 1

    images = np.array(images)
    bboxes = np.concatenate(bboxes,axis=0)
    return images,bboxes