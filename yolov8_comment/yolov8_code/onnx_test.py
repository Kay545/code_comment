import os 
import cv2 
import argparse
import torch 
import colorsys
import onnxruntime
import time 
import numpy as np            
from utils.utils import *

class Onnx_Test(object):
    def __init__(self,opt):
        self.opt = opt 
        
        # GPU/CPU
        self.__init_device()
        # model onnx 加载
        self.__load_onnx()
        # 字典加载
        self.__load_dict()
        #
        self.__get_cls_color()
        
    def __load_onnx(self):
        '''onnx session 初始化'''
        self.onnx_session = onnxruntime.InferenceSession(self.opt.onnx_file,
                                                         providers=['TensorrtExecutionProvider', 
                                                                    'CUDAExecutionProvider', 
                                                                    'CPUExecutionProvider'])
    
    def __init_device(self):
        '''GPU/CPU'''
        if self.opt.cuda:
            os.environ['CUDA_VISIBLE_DEVICES']='0'
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def __load_dict(self):
        '''获取数据集类别'''
        class_list = open(self.opt.class_file,'r').readlines()
        self.class_set = [] 
        for line in class_list:
            line = line.strip()
            self.class_set.append(line)
        self.class_num = len(self.class_set)
        
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
        new_img = np.ones((int(img_h+self.pad_h),int(img_w+self.pad_w),3),dtype=np.uint8)*128 
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
        image = self.__image_padding(image)
        cv2.imwrite('input.jpg',image)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.array(image,dtype=np.float32)
        image = np.transpose(image/255.0,(2,0,1))
        image = np.expand_dims(image,0)
        return image
    
    def __get_cls_color(self):
        hsv_tuples = [(x / len(self.class_set), 1., 1.)
                      for x in range(len(self.class_set))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        
    def __decode(self,outputs):
        fit_index = outputs[:,-1]>self.opt.conf_thresh
        pred = outputs[fit_index]
        result = nms(pred,self.opt.nms_thresh)
        return result 
    
    def detect(self,image):
        img = self.__image_process(image)
        s_t = time.time()
        outputs = self.onnx_session.run(['output'],{'input':img})
        e_t = time.time()
        print('Use time: {:.4f}s'.format(e_t-s_t))
        outputs = torch.from_numpy(outputs[0])
        outputs = outputs.cpu().numpy()
        results = self.__decode(outputs)

        if results != []:
            results[:,[0,2]] = results[:,[0,2]] /self.scale -self.pad_w//2
            results[:,[1,3]] = results[:,[1,3]] /self.scale - self.pad_h//2

            
            for box in results:
                x1,y1,x2,y2,conf,label = box[0],box[1],box[2],box[3],box[4],box[5]
                name = self.class_set[int(label)]
                #print('{} {:.2f} {} {} {} {} '.format(label,conf,int(x1),int(y1),int(x2),int(y2)))
                cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),self.colors[int(label)],2)
                #cv2.putText(image,name,(int(x1),int(y1)+30),cv2.FONT_HERSHEY_SIMPLEX,1,self.colors[int(label)],2)
            cv2.imwrite('onnx.jpg',image)
            
def set_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file',type=str,default='./yolov8.onnx')
    parser.add_argument('--class_file',type=str,default='./data/person_classes.txt')
    parser.add_argument('--conf_thresh',type=float,default=0.25)
    parser.add_argument('--nms_thresh',type=float,default=0.4)
    parser.add_argument('--input_size',type=list,default=[960,544])
    parser.add_argument('--cuda',type=bool,default=True)
    opt = parser.parse_args()
    return opt 

if __name__=='__main__':
    opt = set_option()
    test = Onnx_Test(opt)
    image_path = ''
    image = cv2.imread(image_path)
    test.detect(image)