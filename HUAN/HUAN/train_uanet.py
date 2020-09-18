import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import msra10k_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from uanet import UANet
from torch.backends import cudnn

import torch.nn.functional as F

cudnn.benchmark = True

torch.manual_seed(2018)
torch.cuda.set_device(2)

ckpt_path = './ckpt'
exp_name = 'UAN_uanet'

args = {
    'iter_num': 40000,
    'train_batch_size': 10,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': ''
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(80),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)

criterion_BCE = nn.BCEWithLogitsLoss().cuda()
criterion_MSE = nn.MSELoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = UANet(n_channels=3,n_classes=1).cuda().train()

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
        final_loss_record, total_loss_record = AvgMeter(), AvgMeter()
        down1_loss_record, down2_loss_record, down3_loss_record, down4_loss_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        up1_loss_record, up2_loss_record, up3_loss_record, up4_loss_record  = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

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
            outputs, up1, up2, up3, up4, down1, down2, down3, down4 = net(inputs)
            final_loss = criterion_BCE(F.upsample(outputs, size=labels.size()[2:], mode='bilinear'), labels)
           
            up1_loss = criterion_BCE(F.upsample(up1, size=labels.size()[2:], mode='bilinear'), labels)
            up2_loss = criterion_BCE(F.upsample(up2, size=labels.size()[2:], mode='bilinear'), labels)
            up3_loss = criterion_BCE(F.upsample(up3, size=labels.size()[2:], mode='bilinear'), labels)
            up4_loss = criterion_BCE(F.upsample(up4, size=labels.size()[2:], mode='bilinear'), labels)
           
            down1_loss = criterion_BCE(F.upsample(down1, size=labels.size()[2:], mode='bilinear'), labels)
            down2_loss = criterion_BCE(F.upsample(down2, size=labels.size()[2:], mode='bilinear'), labels)
            down3_loss = criterion_BCE(F.upsample(down3, size=labels.size()[2:], mode='bilinear'), labels)
            down4_loss = criterion_BCE(F.upsample(down4, size=labels.size()[2:], mode='bilinear'), labels)

            total_loss = final_loss + up1_loss + up2_loss + up3_loss + up4_loss  + down1_loss + down2_loss + down3_loss + down4_loss
                  
            total_loss.backward()
            optimizer.step()

            final_loss_record.update(final_loss.data[0], batch_size)
            total_loss_record.update(total_loss.data[0], batch_size)

            up1_loss_record.update(up1_loss.data[0], batch_size)
            up2_loss_record.update(up2_loss.data[0], batch_size)
            up3_loss_record.update(up3_loss.data[0], batch_size)
            up4_loss_record.update(up4_loss.data[0], batch_size)  
            down1_loss_record.update(down1_loss.data[0], batch_size)
            down2_loss_record.update(down2_loss.data[0], batch_size)
            down3_loss_record.update(down3_loss.data[0], batch_size)
            down4_loss_record.update(down4_loss.data[0], batch_size)

            curr_iter += 1

            log = '[iter %d], [final:%.5f], [total:%.5f], [up1:%.5f], [up2:%.5f], [up3:%.5f], [up4:%.5f],'\
                  '[down1:%.5f], [down2:%.5f], [down3:%.5f], [down4:%.5f], [lr %.13f]' % \
                  (curr_iter, final_loss_record.avg, total_loss_record.avg, up1_loss_record.avg, up2_loss_record.avg, up3_loss_record.avg, up4_loss_record.avg,\
                  down1_loss_record.avg, down2_loss_record.avg, down3_loss_record.avg, down4_loss_record.avg, optimizer.param_groups[1]['lr'])
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
