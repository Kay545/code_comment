import os 
import numpy as np 
import time 
import torch 
import warnings
import random 
import torch.nn as nn 
from tqdm import tqdm
from set_option import *
import torch.optim as optim
import torch.backends.cudnn as cudnn 
from torch.utils.data import DataLoader
from torch.autograd import Variable
from loss.loss import YOLOv8_Loss
from nets.yolov8 import YOLOv8
from dataset.dataloader_yolov8 import DataList,yolo_dataset_collate
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
class Trainer(object):
    def __init__(self,opt):
        self.opt = opt 
        self.__get_class_num()
        self.__init_device()
        self.__build_model()
        self.__init_loss()
        self.__init_optimizer()

    def __get_class_num(self):
        '''获取数据集类别及数目'''
        class_list = open(self.opt.class_file,'r').readlines()
        self.class_set = [i.strip() for i in class_list if i!=""]
        self.class_num = len(self.class_set)

    def __init_device(self):
        '''GPU/CPU设备初始化'''
        self.gpu_num = len(self.opt.gpu_id)
        self.cuda = True if self.gpu_num>0 else False 
        if self.cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def __load_weights(self):
        '''加载模型参数'''
        print('Loading pertrained weights from {}'.format(self.opt.weights_file))
        model_dict = self.model.state_dict()
        weights_dict = torch.load(self.opt.weights_file,map_location=self.device)
        weights_dict = {k.replace('module.',''):v for k,v in weights_dict.items()}
        model_dict.update(weights_dict)
        self.model.load_state_dict(model_dict)
        print('Load Weights Finished!!')

    def __build_model(self):
        '''创建模型'''

        self.model = YOLOv8(net_version=self.opt.net_version,class_num=self.class_num)
        if os.path.exists(self.opt.weights_file):
            self.__load_weights()
        self.model.train()
        if self.cuda:
            self.model = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True 
            self.model.to(self.device)
        else:
            self.model.to(self.device)
    def __init_loss(self):
        '''损失函数初始化'''

        
        self.criterion = YOLOv8_Loss(self.class_num,self.opt.input_size)
        
        if self.cuda:
            self.criterion.cuda()
    
    def __init_optimizer(self):
        '''设置优化方法'''
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.opt.lr,weight_decay=self.opt.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.step_size, gamma=self.opt.gamma)

    def __load_train_or_val_data(self,data_file):
        '''获取训练数据或验证数据的列表'''
        lines = open(data_file,'r').readlines()
        # random.shuffle(lines)
        # lines = lines[:10000]
        
        return lines 

    def __get_train_or_val_dataloader(self,datalist,input_size,batch_size):
        '''获取训练或验证数据的队列'''
        dataset = DataList(self.opt,datalist)
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,
                        pin_memory=True,drop_last=True,collate_fn=yolo_dataset_collate)
        return dataloader 

    def __get_lr(self):
        '''获取当前的学习率参数'''
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        return lr 

    def __train_one_epoch(self,train_dataloader,epoch,val_dataloader=None):
        '''训练一轮'''
        train_total_loss = 0.0
        dfl_total_loss = 0.0
        loc_total_loss = 0.0
        cls_total_loss = 0.0
        val_total_loss = 0.0

        print('Start Training ...')
        train_step_num = len(train_dataloader) 
        with tqdm(total=train_step_num,desc='Epoch {}/{}'.format(epoch,self.opt.max_epoch),postfix=dict,
                mininterval=0.3) as pbar:
            for batch_i,(images,targets) in enumerate(train_dataloader):
                start_time = time.time()
                if batch_i>=train_step_num:
                    break
                with torch.no_grad():
                    if self.cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor)).cuda()
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = Variable(torch.from_numpy(targets).type(torch.FloatTensor))
                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                loss,loss_set =self.criterion(outputs,targets)


                dfl_loss = loss_set[2]
                loc_loss = loss_set[0]
                cls_loss = loss_set[1]
                loss.backward()
                self.optimizer.step()
                train_total_loss += loss
                dfl_total_loss += dfl_loss 
                loc_total_loss += loc_loss
                cls_total_loss += cls_loss
                waste_time = time.time() - start_time
                pbar.set_postfix(**{'total': train_total_loss.item() / (batch_i + 1),
                                    'dfl': dfl_total_loss.item() / (batch_i + 1),
                                    'loc': loc_total_loss.item() / (batch_i + 1),
                                    'cls':cls_total_loss.item()/(batch_i + 1),
                                    'lr': self.__get_lr(),
                                    'step/s': waste_time})
                pbar.update(1)
        if epoch%self.opt.save_skip==0 or epoch == self.opt.max_epoch:
            torch.save(self.model.state_dict(),
                    './checkpoints/Epoch{}-loss-{:.4f}.pth'.format(epoch,
                        train_total_loss / train_step_num))    
        

    def train(self):
        train_data = self.__load_train_or_val_data(self.opt.train_file)
        train_dataloader = self.__get_train_or_val_dataloader(train_data,self.opt.input_size,self.opt.batch_size)
        # 整体训练
        for epoch in range(1,self.opt.max_epoch):
            # train_data = self.__load_train_or_val_data(self.opt.train_file)
            # train_dataloader = self.__get_train_or_val_dataloader(train_data,self.opt.input_size,self.opt.batch_size)
            self.__train_one_epoch(train_dataloader,epoch)
            self.scheduler.step()

if __name__=='__main__':
    opt = set_option()
    os.makedirs('./checkpoints',exist_ok=True)
    trainer = Trainer(opt)
    trainer.train()