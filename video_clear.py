from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch
from model_image import resnet50
import pickle
import os
"""此处将train和text两个text文件中的所有视频数据按照ID分类存放"""
# name=[]
# with open('./list/video/test.txt','r') as f:
#     for line in f:
#         a=line.split()[0]
#         b=a.split('/')[3].split('.')[0]
#         c=b[:-6]
#         name.append(c)

# with open('./list/video/test.txt','r') as f:
#  for line in f:
#      a1 = line.split()[0]
#      b1 = a1.split('/')[3].split('.')[0]
#      c1 = b1[:-6]
#      if c1 in name:
#          path='./list/video/test/'+c1+'.txt'
#          if os.path.exists(path):
#              f1 = open(path,'a')
#              f1.write(line)
#          else:
#              f1 = open(path, 'a')
#              f1.write(line)

"""模型初始化和加载预训练模型"""
model = resnet50(num_classes=200)
model = model.cuda()
checkpoint = torch.load('./model_image/model_image0.849.pkl')
model_dict = model.state_dict()
restore_param = {k: v for k, v in checkpoint.items() if k in model_dict}
model_dict.update(restore_param)
model.load_state_dict(model_dict)
print("==> loaded checkpoint '{}'".format('./model_image/model_image0.849.pkl'))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_data_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                normalize,
            ])
with open('id_num_train' + '.pkl', 'rb') as f:
    id_num_train = pickle.load(f)
with open('id_label_train' + '.pkl', 'rb') as f1:
    id_label_train = pickle.load(f1)
for i in id_num_train.keys():
    imgs = []
    features=[]
    path='./list/video/train/'+i+'.txt'
    with open(path) as f:
        for line in f:
            input = Image.open('./dataset/' + line.split()[0]).convert('RGB')
            input=train_data_transform(input)
            imgs.append(input.unsqueeze(0))
    for mn in range(len(imgs)):
        input=imgs[mn].cuda()
        feature=model(input)
        features.append(feature)

    distance=0
    distances=[]
    finalldis=[]
    for m in range(len(imgs)):
        sum=0
        for n in range(len(imgs)):
            distance=torch.sqrt(torch.sum((features[m].data.cpu()-features[n].data.cpu())**2))
            sum+=distance
        distances.append(sum)

    # print('-'*20)
    # for ik in range(len(imgs)):
    #     print(distances[ik])
    # print('-'*20)

    min=10000
    x=0
    for io in range(len(imgs)):
        #print(distances[i])
        if distances[io]<min:
            min=distances[io]
            x=io

    # print('-'*20)
    # print(x)
    # print('-'*20)

    # center=features[-1].data.cpu()
    # for i in range(len(imgs)-1):
    #      center=center+features[i].data.cpu()
    # center=center/len(imgs)
    # print(center)


    for h in range(len(imgs)):
        #hj=torch.sqrt(torch.sum((center,features[h])**2))
        # hj=torch.sqrt(torch.sum((center.cpu()-features[h].data.cpu())**2))
        hj = torch.sqrt(torch.sum((features[x].data.cpu() - features[h].data.cpu()) ** 2))
        finalldis.append(hj)
    # print(finalldis)
    sum_final=0
    sig=[]
    for y in range(len(imgs)):
        sum_final=sum_final+finalldis[y]
    avg_dis=sum_final/len(imgs)
    # print(avg_dis)
    for k in range(len(imgs)):
        if finalldis[k]>avg_dis*1.05:
            sig.append(k)
    count=0
    with open(path) as f:
        for line in f:
            path_name='./list/video/train_cut/'+i+'.txt'
            if count not in sig:
                if os.path.exists(path_name):
                    f1 = open(path_name,'a')
                    f1.write(line)
                else:
                    f1 = open(path_name, 'w')
                    f1.write(line)
                count += 1
            else:
                count+=1


