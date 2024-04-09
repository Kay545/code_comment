#coding:utf-8
from PIL import Image
import cv2
import copy
import numpy as np
def pad2need_shape_rate(img,net_w,net_h,frame=False):
    #cv2.imwrite('aaaa.jpg',img)
    ih,iw,_ = img.shape
    i_rate = iw * 1.0 /ih
    net_rate = net_w * 1.0 /net_h
    # 初始化高度、宽度方向的填充值
    pad_h,pad_w = 0 , 0
    # 新的宽高值
    n_w , n_h = 0 , 0
    if i_rate>=net_rate:#图像过于宽
        #高度方向需要填充的总值
        pad_h=int(iw*1.0/net_rate-ih)
        #宽度方向不需要填充
        pad_w = 0
        n_w = iw
        n_h = int(iw*1.0/net_rate)+1
        #初始化一个空白图像并用原图像的均值进行填充
        n_img=np.ones((n_h,n_w,3),dtype=np.uint8)*114
        n_img[pad_h // 2:(pad_h // 2 + ih), :, :] = img
    elif i_rate<net_rate:#图像过于窄
        pad_w=int(ih*net_rate-iw)
        pad_h=0
        n_w=int(ih*net_rate+1)
        n_h=ih
        #初始化一个空白图像用原图像的均值进行填充
        n_img=np.ones((n_h,n_w,3),dtype=np.uint8)*114
        n_img[:, pad_w // 2:(pad_w // 2 + iw), :] = img
    pad = [pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2]
    n_img = cv2.resize(n_img, (net_w, net_h),cv2.INTER_LINEAR)
    w_scale = net_w * 1.0 / n_w
    h_scale = net_h * 1.0 / n_h
    net_scale = [w_scale, h_scale]
    return n_img,pad,net_scale
class TestDataset(object):
    def __init__(self,dataset,input_size):
        self.img_files = dataset
        self.img_w = input_size[0]
        self.img_h = input_size[1]
    def get_test_data(self,index):
        img_path = self.img_files[index]
        img_info = {}
        try:
            image = cv2.imread(img_path)
        except:
            print('No this image {}'.format(img_path))
            index +=1
        net_w , net_h = self.img_w,self.img_h

        image,pad,scale = pad2need_shape_rate(image,net_w,net_h)
        img_info['image_path']= img_path
        img_info['pad'] =pad
        img_info['scale']= scale
        return image,img_info
    def get_video_data(self,frame):
        frame_info ={}
        net_w, net_h = self.img_w, self.img_h
        frame, pad, scale = pad2need_shape_rate(frame, net_w, net_h,frame=True)
        frame_info['pad'] = pad
        frame_info['scale']= scale
        return frame,frame_info
    def __getitem__(self,index):
        img,img_info=self.get_test_data(index)
        #img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #cv2.imwrite('2.jpg',img)
        img = np.array(img,dtype=np.float32)
        img = np.transpose(img/255.0,(2,0,1))
        return img,img_info
    def __len__(self):
        return len(self.img_files)

def yolo_test_collate(batch):
    images = []
    infoes = []
    for img,info in batch:
        images.append(img)
        infoes.append(info)
    images = np.array(images)
    return images,infoes