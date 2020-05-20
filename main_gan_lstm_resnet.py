import torch
from torch import nn
import torch.nn.functional as F
import argparse
import torch
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from train import *
from centerloss import CenterLoss
from validate import *
from model_lstm_selfattention import LSTMClassifier
from model_Resnet import resnet50, Discriminator
from dataset_lstm import CubDataset, CubDataset1, CubTextDataset,CubDataset2

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=1, type=int, help='GPU nums to use')
    parser.add_argument('--lr', default=0.01, type=float, help='learnging rate')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--model_path', default='./textmodel/', type=str, help='path to model')
    parser.add_argument('--snapshot', default='./model_image/model_image0.849.pkl', type=str, required=False, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--snapshot1', default='./pretrained/epoch_680_0.99825_0.399.pkl', type=str, required=False,
                        metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--loss_choose', default='c', type=str, required=False,
                        help='choose loss(c:centerloss, r:rankingloss)')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')
    parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency')
    parser.add_argument('--eval_epoch', default=1, type=int, help='every eval_epoch we will evaluate')
    parser.add_argument('--eval_epoch_thershold', default=2, type=int, help='eval_epoch_thershold')
    args = parser.parse_args()
    return args


def print_args(args):
    print ("==========================================")
    print ("==========       CONFIG      =============")
    print ("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print ("\n")


def main():
    args = arg_parse()
    # print_args(args)
    vector = torch.rand([50000, 100])
    model = LSTMClassifier(emb_vectors=vector).cuda()
    model1 = resnet50(num_classes=200).cuda()
    discriminator = Discriminator().cuda()
    cudnn.benchmark = True
    if True:
        print("==> loading checkpoint '{}'".format(args.snapshot))
        checkpoint = torch.load(args.snapshot)
        model_dict1 = model1.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict1}
        model_dict1.update(restore_param)
        model1.load_state_dict(model_dict1)
        print("==> loaded checkpoint '{}'".format(args.snapshot))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshot))

    criterion = nn.CrossEntropyLoss()
    D_criterion=torch.nn.BCELoss()
    center_loss = CenterLoss(200, 200, True)
    param1 = list(model1.parameters())+ list(center_loss.parameters())
    param2= list(model.parameters())
    params=[
        {"params":param1,"lr":0.002},
        {"params": param2, "lr": 1}
    ]
    params1 = list(discriminator.parameters())

    opt = torch.optim.Adadelta(params, rho=0.9, eps=1e-6)
    opt1 = torch.optim.Adadelta(params1, lr=1, rho=0.9, eps=1e-6)

    train_list = './list/four4_audio.txt'
    data_set = get_train_set('./dataset/', train_list, args)

    data_set0 = get_test_set('./dataset/', 'list/image/test.txt', args)
    data_loader0 = DataLoader(dataset=data_set0, num_workers=4, batch_size=4, shuffle=False)
    data_set1 = CubTextDataset('dataset', 'list/text/test.txt', 'test')
    data_loader1 = DataLoader(dataset=data_set1, num_workers=4, batch_size=4, shuffle=False)
    data_set2 = get_test_set1('./dataset/', 'list/video/test_cut.txt', args)
    data_loader2 = DataLoader(dataset=data_set2, num_workers=4, batch_size=4, shuffle=False)
    data_set3 = get_test_set('./dataset/', 'list/audio/test.txt', args)
    data_loader3 = DataLoader(dataset=data_set3, num_workers=4, batch_size=4, shuffle=False)
    savepath = './gan_noise_att/'

    for epoch in range(800):
        sum = 0
        labelsum = 0
        data_loader = DataLoader(dataset=data_set, num_workers=4, batch_size=4, shuffle=True)
        train(data_loader, args, model1, criterion, D_criterion, center_loss, opt, opt1, epoch, args.epochs, model, discriminator)
        print('-' * 20)
        print("Video Acc:")
        video_acc = validate1(data_loader2, model1, args, False)
        print("Image Acc:")
        image_acc = validate(data_loader0, model1, args, False)
        print("Audio Acc:")
        audio_acc = validate(data_loader3, model1, args, False)
        model.eval()
        for a, b in data_loader1:
            testa = Variable(a.cuda())
            testb = Variable(b.cuda())
            sum += model.loss_n_acc(testa, testb)[1]
            labelsum += testb.size()[0]
            testacc=sum / labelsum
        print('lstm+selfattetnion test', epoch, testacc)
        save_model_path = savepath + 'lstmselfattention'+'epoch_' + str(epoch) + '_' + str(image_acc) +'_' + str(testacc) + '.pkl'
        save_model_path1 = savepath + 'resnet50'+'epoch_' + str(epoch) + '_' + str(image_acc) +'_' + str(testacc) + '.pkl'
        torch.save(model.state_dict(), save_model_path)
        torch.save(model1.state_dict(), save_model_path1)

def get_train_set(data_dir, train_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = 448
    scale_size = 512
    train_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_set = CubDataset(data_dir, train_list, train_data_transform)
    return train_set


def get_test_set(data_dir, test_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = 448
    scale_size = 512
    test_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    test_set = CubDataset1(data_dir, test_list, test_data_transform)
    return test_set

def get_test_set1(data_dir, test_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = 448
    scale_size = 512
    test_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    test_set = CubDataset2(data_dir, test_list, test_data_transform)
    return test_set

def get_text_set(data_dir, test_list, args, split):
    data_set = CubTextDataset(data_dir, test_list, split)
    return data_set


if __name__ == "__main__":
    main()
