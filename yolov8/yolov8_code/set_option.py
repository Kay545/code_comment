import argparse
#-----------------------------------#
#          训练参数设置
#-----------------------------------#
def set_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch',type=int,default=100)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--net_version',type=str,default='yolov8s')
    parser.add_argument("--class_file", type=str, default="data/person_classes.txt")
    parser.add_argument("--train_file", type=str, default="data/train.txt")
    parser.add_argument("--weights_file", type=str,default='checkpoints/best.pth')
    parser.add_argument("--input_size", type=list, default=[960,544])
    parser.add_argument("--gpu_id",type = list,default=[2])
    parser.add_argument("--save_skip", type=int, default=2)
    parser.add_argument("--mosaic", type=bool,default=True)
    parser.add_argument('--argument',type=bool,default=True)
    parser.add_argument('--step_size',type=int,default=10)
    parser.add_argument('--gamma',type=float,default=0.5)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    opt = parser.parse_args()
    return opt