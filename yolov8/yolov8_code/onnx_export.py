#coding:utf-8
import os
import torch
import cv2 
import argparse
from torch.autograd import Variable
from nets.yolov8 import YOLOv8
import numpy as np 

class Onnx_Export(object):
    def __init__(self,opt):
        self.opt = opt 
        
        self.input_w = opt.input_size[0]
        self.input_h = opt.input_size[1]
        
        # 设备初始化
        self.__init_device()
        # 字典初始化
        self.__load_dict()
        # 创建模型
        self.__creat_model()
    
    
    def __init_device(self):
        '''GPU/CPU设备初始化'''
        self.cuda = False 
        if torch.cuda.is_available():
            self.cuda = True 
            os.environ['CUDA_VISIBLE_DEVICES']=self.opt.gpu_id
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # self.device = torch.device('cpu')
    def __load_dict(self):
        '''获取数据集类别'''
        class_list = open(self.opt.class_file,'r').readlines()
        self.class_set = [] 
        for line in class_list:
            line = line.strip()
            self.class_set.append(line)
        self.class_num = len(self.class_set)

    def __creat_model(self):
        '''模型初始化'''
        self.model = YOLOv8(class_num=self.class_num,input_size=self.opt.input_size,onnx_export=True)
        if os.path.exists(self.opt.weights_file):
            self.__load_weights()
        self.model.eval()
        self.model.to(self.device)

    def __load_weights(self):
        '''加载模型参数'''
        print('Loading weights from :{}'.format(self.opt.weights_file))
        model_dict = self.model.state_dict()
        weights_dict = torch.load(self.opt.weights_file,map_location=self.device)
        weights_dict = {k:v for k,v in weights_dict.items()}
        model_dict.update(weights_dict)
        self.model.load_state_dict(model_dict)
        print("Load weights finished!!")
        
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
        image = self.__image_padding(image)
        cv2.imwrite('2.jpg',image)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.array(image/255.0,dtype=np.float32)
        image = np.transpose(image,(2,0,1))
        image = np.expand_dims(image,0)
        return image
    
    def __export_onnx(self,img):
        self.onnx_file = './yolov8.onnx'
        self.dynamic_axes = {
            'input':{0:'N'},
            'output':{0:'N'}
        }
        torch.onnx.export(
            self.model,
            img,
            self.onnx_file,
            #export_params=True,
            verbose=False,
            opset_version=12,
            #training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['input'],
            output_names = ['output'],
            dynamic_axes=self.dynamic_axes
        )
    
    def main(self,image):
        img = self.__image_process(image)
        img = Variable(torch.from_numpy(img).type(torch.FloatTensor)).to(self.device)
        img = torch.cat([img,img,img,img,img],dim=0)
        self.__export_onnx(img)
        
def set_option():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone',type=str,default='yolov5s')
    parser.add_argument('--weights_file',type=str,default='./checkpoints/best.pth')
    parser.add_argument('--class_file',type=str,default='./data/person_classes.txt')
    parser.add_argument('--input_size',type=list,default=[960,544])
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--gpu_id',type=str,default='0')
    opt = parser.parse_args()
    return opt     

if __name__=='__main__':
    opt = set_option()
    export = Onnx_Export(opt)
    img_path = '.jpg'
    image = cv2.imread(img_path)
    export.main(image)
    







