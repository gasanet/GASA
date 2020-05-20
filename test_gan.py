import argparse
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_lstm import CubDataset, CubDataset1, CubTextDataset
from model_Resnet import resnet50
from model_lstm_selfattention import LSTMClassifier
from retrieval import *
from torch.autograd import Variable
import pickle

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('--gpu', default=2, type=int, help='GPU nums to use')
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--data_path', default='./dataset/', type=str, required=False, help='path to dataset')
    parser.add_argument('--snapshot', default='./model_all/lstmselfattentionepoch_30_0.8584498532711894_0.40375.pkl', type=str, required=False,
                        help='path to latest checkpoint')
    parser.add_argument('--snapshot1', default='./model_all/resnet50epoch_30_0.8584498532711894_0.40375.pkl', type=str, required=False,
                        help='path to latest checkpoint')
    parser.add_argument('--feature', default='./feature_gan', type=str, required=False, help='path to feature')
    parser.add_argument('--crop_size', default=448, type=int, help='crop size')
    parser.add_argument('--scale_size', default=512, type=int, help='the size of the rescale image')

    args = parser.parse_args()
    return args


def print_args(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")


def main():
    args = arg_parse()
    print_args(args)
    print("==> Creating dataloader...")
    data_dir = args.data_path
    test_list1 = './list/image/test.txt'
    test_loader1 = get_test_set(data_dir, test_list1, args)
    test_list2 = './list/video/test.txt'
    test_loader2 = get_test_set(data_dir, test_list2, args)
    test_list3 = './list/audio/test.txt'
    test_loader3 = get_test_set(data_dir, test_list3, args)
    test_list4 = './list/text/test.txt'
    test_loader4 = get_text_set(data_dir, test_list4, args, 'test')

    out_feature_dir1 = os.path.join(args.feature, 'image')
    out_feature_dir2 = os.path.join(args.feature, 'video')
    out_feature_dir3 = os.path.join(args.feature, 'audio')
    out_feature_dir4 = os.path.join(args.feature, 'text')

    mkdir(out_feature_dir1)
    mkdir(out_feature_dir2)
    mkdir(out_feature_dir3)
    mkdir(out_feature_dir4)

    print("==> Loading the modelwork ...")
    model = resnet50(num_classes=200)
    vector = torch.rand([50000, 100])
    model1= LSTMClassifier(emb_vectors=vector)
    model = model.cuda()
    model1.cuda()

    if True:
        print("==> loading checkpoint '{}'".format(args.snapshot))
        checkpoint = torch.load(args.snapshot)
        model_dict = model.state_dict()
        restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(restore_param)
        model.load_state_dict(model_dict)
        print("==> loaded checkpoint '{}'".format(args.snapshot))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshot))

    if True:
        print("==> loading checkpoint '{}'".format(args.snapshot1))
        checkpoint1 = torch.load(args.snapshot1)
        model_dict1 = model1.state_dict()
        restore_param1 = {k: v for k, v in checkpoint1.items() if k in model_dict1}
        model_dict1.update(restore_param1)
        model1.load_state_dict(model_dict1)
        print("==> loaded checkpoint '{}'".format(args.snapshot1))
    else:
        print("==> no checkpoint found at '{}'".format(args.snapshot1))
    model.eval()
    model1.eval()

    print("Video Features ...")
    vid = extra(model, test_loader2, out_feature_dir2, args, flag='v')
    print("Text Features ...")
    txt = extra(model1, test_loader4, out_feature_dir4, args, flag='t')
    print("Image Features ...")
    img = extra(model, test_loader1, out_feature_dir1, args, flag='i')
    print("Audio Features ...")
    aud = extra(model, test_loader3, out_feature_dir3, args, flag='a')

    compute_mAP(img, vid, aud, txt)


def mkdir(out_feature_dir):
    if not os.path.exists(out_feature_dir):
        os.makedirs(out_feature_dir)

def extra(model, test_loader, out_feature_dir, args, flag):
    size = args.batch_size
    out_sum = {}
    id_num = {}
    with open('nums' + '.pkl', 'rb') as f:
        dict_a = pickle.load(f)
    with open('label' + '.pkl', 'rb') as f:
        dict_label = pickle.load(f)
    for i in dict_a.keys():
        b = np.zeros(200)
        out_sum[i] = b
        id_num[i] = 0
    if flag != 'v':
        f = np.zeros((len(test_loader)*size, 200))
        num = 0
        for i, (input, target, _) in enumerate(test_loader):
            target = target.cuda()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input).cuda()
            if (flag == 't'):
                output = model.predict(input_var)
            else:
                output = model.forward_share(input_var)
            output = F.softmax(output, dim=1).detach().cpu().numpy()
            num += output.shape[0]
            if (i == len(test_loader) - 1):
                f[i * size:num, :] = output
            else:
                f[i * size:(i + 1) * size, :] = output
    else:
        f = np.zeros((5290, 200))
        num = 0
        for i, (input, target, name) in enumerate(test_loader):
            v_id = name[0].split('/')[-1].split(' ')[0].split('.')[0][:-6]
            if v_id in dict_a:
                id_num[v_id]+=1
            else:
                print("不存在此id:",v_id)
            with torch.no_grad():
                input_var = Variable(input).cuda()
                target_var = Variable(target).cuda()
            output = model.forward_share(input_var)[0].detach().cpu().numpy()
            out_sum[v_id] += output
        for i in dict_a.keys():
            out_sum[i] /= id_num[i]
        count=0
        for i in dict_label.keys():
            output = torch.tensor([out_sum[i]])
            output = F.softmax(output, dim=1).detach().numpy()
            num += output.shape[0]
            if (count == 0):
                f[count * size:num, :] = output
            else:
                f[count * size:(count + 1) * size, :] = output
            count += 1
    np.savetxt(out_feature_dir + '/features_t.txt', f)
    return out_feature_dir + '/features_t.txt'


def get_test_set(data_dir, test_list, args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    test_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    test_set = CubDataset1(data_dir, test_list, test_data_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return test_loader


def get_text_set(data_dir, test_list, args, split):
    data_set = CubTextDataset(data_dir, test_list, split)
    data_loader = DataLoader(dataset=data_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)
    return data_loader


if __name__ == "__main__":
    main()
