import sys
import torch
from torch.autograd import Variable
import numpy as np
import pickle

def validate(loader, model, args, flag):
    model.eval()
    total_output = []
    total_label = []
    start_model = True
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()
        target = target.cuda()
        if(flag):
            output = model.forward_txt(input_var)
        else:
            output = model.forward_share(input_var)
        if start_model:
            total_output = output.data.float()
            total_label = target.data.float()
            start_model = False
        else:
            total_output = torch.cat((total_output, output.data.float()), 0)
            total_label = torch.cat((total_label, target.data.float()), 0)

    _, predict = torch.max(total_output, 1)
    acc = torch.sum(torch.squeeze(predict.float() == total_label)).item() / float(total_label.size()[0])
    print('Prec@1:' + str(acc))
    return acc

def validate1(loader, model, args, flag):
    model.eval()
    out_sum={}
    with open('nums' + '.pkl', 'rb') as f:
        id_num = pickle.load(f)
    with open('label' + '.pkl', 'rb') as f:
        id_label = pickle.load(f)
    for i in id_num.keys():
        b = np.zeros(200)
        out_sum[i] = b
    vount=0
    for i, (input,target,name) in enumerate(loader):
        vount+=1
        with torch.no_grad():
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()
            target = target.cuda().cpu().detach().numpy()
        output = model.forward_share(input_var).cpu().detach().numpy()
        if vount<=18588:
            v_id = name[0].split('/')[-1].split(' ')[0].split('.')[0][:-6]
            v_id1 = name[1].split('/')[-1].split(' ')[0].split('.')[0][:-6]
            v_id2 = name[2].split('/')[-1].split(' ')[0].split('.')[0][:-6]
            v_id3 = name[3].split('/')[-1].split(' ')[0].split('.')[0][:-6]
            out_sum[v_id] += output[0]
            out_sum[v_id1] += output[1]
            out_sum[v_id2] += output[2]
            out_sum[v_id3] += output[3]
        else:
            v_id = name[0].split('/')[-1].split(' ')[0].split('.')[0][:-6]
            v_id1 = name[1].split('/')[-1].split(' ')[0].split('.')[0][:-6]
            v_id2 = name[2].split('/')[-1].split(' ')[0].split('.')[0][:-6]

            out_sum[v_id] += output[0]
            out_sum[v_id1] += output[1]
            out_sum[v_id2] += output[2]

    count_T=0
    for i in id_label.keys():
        out_sum[i]/=id_num[i]
        if np.argmax(out_sum[i])==id_label[i]:
            count_T+=1
    print(count_T/len(id_label))