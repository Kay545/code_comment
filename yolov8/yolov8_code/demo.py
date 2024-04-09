#coding:utf-8
import os 
import time 
import torch 
import cv2 
import colorsys 
import argparse
from torch.autograd import Variable
from nets.yolov8 import YOLOv8
import numpy as np 
from utils.decode import Decode
from utils.utils import *
class YOLO_Detect(object):
    def __init__(self,opt):
        self.opt = opt 
        self.input_w = opt.input_size[0]
        self.input_h = opt.input_size[1]
        self.__init_device()
        self.__init_classes()
        self.__creat_model()
        self.__init_colors()
        self.decode = Decode(opt,class_num=self.class_num)
    def __init_device(self):
        'GPU/CPU设备初始化'
        self.cuda = False 
        gpu_id = self.opt.gpu_id 
        if gpu_id !='' and torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            self.device = torch.device('cuda')
            self.cuda = True 
        else:
            self.device = torch.device('cpu')

    def  __creat_model(self):
        '''模型初始化'''
        #self.model = YOLOV5(class_num=self.class_num)
        self.model = YOLOv8(net_version=self.opt.net_version,class_num=self.class_num)
        if os.path.exists(self.opt.weights_file):
            self.__load_weights()
        self.model.eval()
        self.model.to(self.device)

    def __load_weights(self):
        '''加载模型参数'''
        print('Loading pertrained weights from {}'.format(self.opt.weights_file))
        model_dict = self.model.state_dict()
        weights_dict = torch.load(self.opt.weights_file,map_location='cpu')
        weights_dict = {k.replace('module.',''):v for k,v in weights_dict.items()}
        model_dict.update(weights_dict)
        self.model.load_state_dict(model_dict)
        print('Load Weights Finished!!')

    def __init_classes(self):
        '''获取数据集类别及数目'''
        class_list = open(self.opt.class_file,'r').readlines()
        self.class_names = [i.strip() for i in class_list if i!=""]
        self.class_num = len(self.class_names)

    def __init_colors(self):
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    def __image_padding(self,image):
        '''图像尺寸校正'''
        net_w,net_h = self.opt.input_size[0],self.opt.input_size[1]
        img_h,img_w,_ = image.shape
        img_rate = img_w*1.0/img_h 
        net_rate = net_w*1.0/net_h
        self.pad_w,self.pad_h = 0,0 
        if img_rate>=net_rate:# 图像过宽,高度填充
            self.pad_h = int(img_w/net_rate - img_h)
        else:#宽度填充
            self.pad_w = int(img_h*net_rate - img_w)
        # 新建一张空白图像
        new_img = np.full((int(img_h+self.pad_h),int(img_w+self.pad_w),3),114,dtype=np.uint8)
        new_h,new_w,_ = new_img.shape
        new_img[self.pad_h//2:self.pad_h//2+img_h,self.pad_w//2:self.pad_w//2+img_w,:] = image 
        # 将new_image resize到input_size
        new_img = cv2.resize(new_img,(net_w,net_h))
        self.scale = net_w *1.0/new_w
        return new_img
    
    def __image_process(self,image):
        ''' 图像预处理 '''
        #首先调整到网络所需大小
        self.image_h,self.image_w ,_ = image.shape
        img = self.__image_padding(image)
        
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.array(img/255.0,dtype=np.float32)
        image = np.transpose(image,(2,0,1))
        image = np.expand_dims(image,0)
        return img,image


    def detect(self,image):
        ori_img,img = self.__image_process(image)
        img = Variable(torch.from_numpy(img).type(torch.FloatTensor)).cuda()
        with torch.no_grad():
            outputs = self.model(img)
        preds = self.decode(outputs)[0]
        try:
            results = nms(preds,conf_thresh=self.opt.conf_thresh,nms_thresh=self.opt.nms_thresh)
        except:
            results = []
        if results != []:            
            
            for box in results:
                x1,y1,x2,y2,conf,label = box[0],box[1],box[2],box[3],box[4],box[5]
                name = self.class_names[int(label)]
                #print('{} {:.2f} {} {} {} {} '.format(label,conf,int(x1),int(y1),int(x2),int(y2)))
                cv2.rectangle(ori_img,(int(x1),int(y1)),(int(x2),int(y2)),self.colors[int(label)],2)
        return ori_img 


def set_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_version',type=str,default='yolov8s')
    parser.add_argument('--weights_file',type=str,default='./checkpoints/best.pth')
    parser.add_argument('--class_file',type=str,default='./data/person_classes.txt')
    parser.add_argument('--conf_thresh',type=float,default=0.3)
    parser.add_argument('--nms_thresh',type=float,default=0.45)
    parser.add_argument('--input_size',type=list,default=[960,544])
    parser.add_argument('--gpu_id',type=str,default='0')
    opt = parser.parse_args()
    return opt 

if __name__=='__main__':
    opt = set_option()
    os.makedirs('./outputs',exist_ok=True)
    mot_detect = YOLO_Detect(opt)
    test_images_dir = './test_imgs'
    img_list = os.listdir(test_images_dir)
    for item in img_list:
        image_path = os.path.join(test_images_dir,item)
        
        print(item)
        image = cv2.imread(image_path)
        s_time = time.time()
        image = mot_detect.detect(image)
        e_time = time.time()
        print(item,'-----------------{:.3f}'.format(e_time-s_time))
        save_path = os.path.join('./outputs',item)
        cv2.imwrite(save_path,image)


