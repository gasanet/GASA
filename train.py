import os
import time
import torch
from torch.autograd import Variable
from util import AverageMeter, Log
from rankingloss import *
import numpy as np
import torch.nn.functional as F

def train(train_loader, args, model, criterion, D_criterion, center_loss, optimizer, optimizer1, epoch, num_epochs, mymodel, discriminator):
    since = time.time()

    running_loss0 = AverageMeter()
    running_loss1 = AverageMeter()
    running_loss2 = AverageMeter()
    running_loss3 = AverageMeter()
    running_loss4 = AverageMeter()
    running_loss5 = AverageMeter()
    running_loss6 = AverageMeter()
    running_loss = AverageMeter()
    running_myloss = AverageMeter()
    running_lossC = AverageMeter()

    log = Log()
    model.train()
    mymodel.train()
    discriminator.train()

    img_onehot = torch.zeros(1, 4)
    vid_onehot = torch.zeros(1, 4)
    aud_onehot = torch.zeros(1, 4)
    txt_onehot = torch.zeros(1, 4)
    img_onehot[0][0] = 1
    vid_onehot[0][1] = 1
    aud_onehot[0][2] = 1
    txt_onehot[0][3] = 1

    sumx = 0
    sumy = 0
    for (i, (input,input1,input2,input3, target))in enumerate(train_loader):
        input_var = Variable(input.cuda())
        input_var1 = Variable(input1.cuda())
        input_var2 = Variable(input2.cuda())
        input_var3 = Variable(input3.cuda())

        target_var = Variable(target.cuda())
        target_var1 = Variable(target.cuda())
        target_var2 = Variable(target.cuda())
        target_var3 = Variable(target.cuda())

        outputs= model(input_var, input_var1, input_var2)
        myloss, mytxt = mymodel.loss(input_var3, target_var3)

        size = int(outputs.size(0) / 3)
        img = outputs.narrow(0, 0, size)
        vid = outputs.narrow(0, size, size)
        aud = outputs.narrow(0, 2 * size, size)

        loss0 = criterion(img, target_var)
        loss1 = criterion(vid, target_var1)
        loss2 = criterion(aud, target_var2)
        loss4 = loss0 + loss1 + loss2 + myloss

        if (args.loss_choose == 'r'):
            loss6, _ = ranking_loss(targets, outputs, margin=1, margin2=0.5, squared=False)
            loss6 = loss6 * 0.1
        else:
            loss6 = 0.0

        loss = loss4 + loss6

        mysize1,mysize2 = img.size()
        real_label = torch.ones(mysize1)

        lossC = D_criterion(torch.sum(discriminator(img) * img_onehot.repeat(mysize1, 1).cuda(), dim=1),real_label.cuda()) \
                + D_criterion(torch.sum(discriminator(vid) * vid_onehot.repeat(mysize1, 1).cuda(), dim=1), real_label.cuda()) \
                + D_criterion(torch.sum(discriminator(aud) * aud_onehot.repeat(mysize1, 1).cuda(), dim=1), real_label.cuda()) \
                + D_criterion(torch.sum(discriminator(mytxt) * txt_onehot.repeat(mysize1, 1).cuda(), dim=1), real_label.cuda())
        g_loss = loss-lossC
        d_loss = -(loss-lossC)

        batchsize = input_var.size(0)
        running_loss0.update(loss0.item(), batchsize)
        running_loss1.update(loss1.item(), batchsize)
        running_loss2.update(loss2.item(), batchsize)
        running_loss4.update(loss4.item(), batchsize)
        if (args.loss_choose == 'r'):
            running_loss6.update(loss6.item(), batchsize)
        running_loss.update(loss.item(), batchsize)
        running_myloss.update(myloss.item(), batchsize)
        running_lossC.update(lossC.item(), batchsize)

        optimizer.zero_grad()
        g_loss.backward(retain_graph=True)
        optimizer.step()
        optimizer1.zero_grad()
        d_loss.backward()
        optimizer1.step()

        sumx += mymodel.loss_n_acc(input_var3, target_var3)[1]
        sumy += input_var3.size()[0]
        mytext_acc = sumx / sumy

        if (i % args.print_freq == 0):
            print('-' * 20)
            print('Epoch [{0}/{1}][{2}/{3}]'.format(epoch, num_epochs, i, len(train_loader)))
            print('Image Loss: {loss.avg:.5f}'.format(loss=running_loss0))
            print('Video Loss: {loss.avg:.5f}'.format(loss=running_loss1))
            print('Audio Loss: {loss.avg:.5f}'.format(loss=running_loss2))
            print('AllMedia Loss: {loss.avg:.5f}'.format(loss=running_loss4))
            print('lstm+selfattention Loss: {loss.avg:.5f}'.format(loss=running_myloss))
            print('Discriminator Loss: {loss.avg:.5f}'.format(loss=running_lossC))
            if (args.loss_choose == 'r'):
                print('Ranking Loss: {loss.avg:.5f}'.format(loss=running_loss6))
            print('All Loss: {loss.avg:.5f}'.format(loss=running_loss))
            print("Text train Acc:", mytext_acc)

            log.save_train_info(epoch, i, len(train_loader), running_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


