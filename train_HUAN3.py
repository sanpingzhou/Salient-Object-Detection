import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import duts_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from huanet_model3 import HUANet
from torch.backends import cudnn

import torch.nn.functional as F

cudnn.benchmark = True

torch.manual_seed(2018)
#torch.cuda.set_device(2)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
ckpt_path = './ckpt'
exp_name = 'HUAN3'

args = {
    'iter_num': 40000,
    'train_batch_size': 40,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': ''
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(300),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(duts_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)

BCE = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = HUANet().cuda().train()
    net = nn.DataParallel(net)

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print 'training resumes from ' + args['snapshot']
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, pred_loss_record = AvgMeter(), AvgMeter()
        down1_loss_record, down2_loss_record, down3_loss_record = AvgMeter(), AvgMeter(), AvgMeter()
        up1_loss_record, up2_loss_record, up3_loss_record = AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()

            pred0_1, pred0_2, out1, out2, out3,\
            d11, d12, d13, d14, u11, u12, u13, u14,\
	    d21, d22, d23, d24, u21, u22, u23, u24,\
	    d31, d32, d33, d34, u31, u32, u33, u34 = net(inputs)

            loss_pred = BCE(pred0_1, labels) + BCE(pred0_2, labels)
            loss_down1 = BCE(d11, labels) + BCE(d12, labels) + BCE(d13, labels) + BCE(d14, labels)
            loss_up1 = BCE(u11, labels) + BCE(u12, labels) + BCE(u13, labels) + BCE(u14, labels) + BCE(out1, labels)
            loss_down2 = BCE(d21, labels) + BCE(d22, labels) + BCE(d23, labels) + BCE(d24, labels)
            loss_up2 = BCE(u21, labels) + BCE(u22, labels) + BCE(u23, labels) + BCE(u24, labels) + BCE(out2, labels)
            loss_down3 = BCE(d31, labels) + BCE(d32, labels) + BCE(d33, labels) + BCE(d34, labels)
            loss_up3 = BCE(u31, labels) + BCE(u32, labels) + BCE(u33, labels) + BCE(u34, labels) + BCE(out3, labels)
            
            loss = loss_pred + loss_down1 + loss_up1 + loss_down2 + loss_up2 +\
                   loss_down3 + loss_up3 
 
            loss.backward()
            optimizer.step()

            total_loss_record.update(loss.data[0], batch_size)

            pred_loss_record.update(loss_pred.data[0], batch_size)
            down1_loss_record.update(loss_down1.data[0], batch_size)
            up1_loss_record.update(loss_up1.data[0], batch_size)
            down2_loss_record.update(loss_down2.data[0], batch_size)
            up2_loss_record.update(loss_up2.data[0], batch_size)
            down3_loss_record.update(loss_down3.data[0], batch_size)
            up3_loss_record.update(loss_up3.data[0], batch_size)
           

            curr_iter += 1

            log = '[iter %d], [total:%.5f], [pred:%.5f], [d1:%.5f], [u1:%.5f], [d2:%.5f], [u2:%.5f], [d3:%.5f], [u3:%.5f], [lr:%.5f]'\
                   % (curr_iter, total_loss_record.avg, pred_loss_record.avg, down1_loss_record.avg, up1_loss_record.avg, down2_loss_record.avg, up2_loss_record.avg, \
                     down3_loss_record.avg, up3_loss_record.avg, optimizer.param_groups[1]['lr'])
            print log
            open(log_path, 'a').write(log + '\n')

            if curr_iter%5000 == 0:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
            if curr_iter == args['iter_num']:
                return



if __name__ == '__main__':
    main()
