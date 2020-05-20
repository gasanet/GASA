import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import random
#四种媒体数据共同加载
class CubDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None):
        super(CubDataset, self).__init__()
        self.input_transform = input_transform

        self.vocabulary = list(" abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        self.max_length = 448

        image_list = []
        video_list = []
        audio_list = []
        text_list = []

        label_list = []

        with open(list_path, 'r') as f:
            for line in f.readlines():
                imagename, videoname, audioname, textname, class_label = line.split()
                image_list.append(imagename)
                video_list.append(videoname)
                audio_list.append(audioname)
                for line in open(os.path.join(image_dir, textname), encoding="utf-8"):
                    line = line.lower()
                    textname = line.split("\n")[0]
                text_list.append(textname)
                label_list.append(int(class_label))

        self.image_filenames = [os.path.join(image_dir, x) for x in image_list]
        self.video_filenames = [os.path.join(image_dir, x) for x in video_list]
        self.audio_filenames = [os.path.join(image_dir, x) for x in audio_list]
        self.text_filenames = text_list
        #  self.name_lists=name_list
        self.label_list = label_list

    def __getitem__(self, index):
        data = []
        input_image = Image.open(self.image_filenames[index]).convert('RGB')
        input_video = Image.open(self.video_filenames[index]).convert('RGB')
        input_audio = Image.open(self.audio_filenames[index]).convert('RGB')
        if self.input_transform:
            input_image = self.input_transform(input_image)
            input_video = self.input_transform(input_video)
            input_audio = self.input_transform(input_audio)
        num = random.randrange(0, 10, 2)  # 0, 2, 4, 8
        data += [0] * num
        data += [self.vocabulary.index(i) + 1 for i in list(raw_text) if i in self.vocabulary]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif len(data) < self.max_length:
            data += [0] * (self.max_length - len(data))
        input_text = np.array(data, dtype=np.int64)
        class_label = self.label_list[index]
        return input_image,input_video,input_audio, input_text, class_label

    def __len__(self):
        return len(self.image_filenames)
#加载音频和图片数据
class CubDataset1(data.Dataset):
    def __init__(self, image_dir, list_path,input_transform=None):
        super(CubDataset1, self).__init__()
        self.input_transform = input_transform

        name_list = []
        label_list = []

        with open(list_path, 'r') as f:
            for line in f.readlines():
                imagename, class_label = line.split()
                name_list.append(imagename)
                label_list.append(int(class_label))

        self.image_filenames = [os.path.join(image_dir, x) for x in name_list]
        self.label_list = label_list

    def __getitem__(self, index):
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        class_label = self.label_list[index]
        return input, class_label

    def __len__(self):
        return len(self.image_filenames)
#加载文本数据
class CubTextDataset(data.Dataset):
    def __init__(self, image_dir, list_path, split):
        super(CubTextDataset, self).__init__()
        self.split = split
        self.vocabulary = list(" abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
        self.max_length = 448

        texts, labels = [], []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                if self.split== 'train':
                    path = line.split()[3]
                elif self.split== 'test':
                    path = line.split()[0]
                label = int(line.split()[-1])
                for line in open(os.path.join(image_dir, path), encoding="utf-8"):
                    line = line.lower()
                    text = line.split("\n")[0]
                texts.append(text)
                labels.append(label)

        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = []
        if (self.split == 'train'):
            num = random.randrange(0, 10, 2)  # 0, 2, 4, 8
            data += [0] * num
            data += [self.vocabulary.index(i) + 1 for i in list(raw_text) if i in self.vocabulary]
        else:
            data = [self.vocabulary.index(i) + 1 for i in list(raw_text) if i in self.vocabulary]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif len(data) < self.max_length:
            data += [0] * (self.max_length - len(data))
        input = np.array(data, dtype=np.int64)
        class_label = self.labels[index]
        return input, class_label

    def __len__(self):
        return len(self.labels)
#加载视频数据
class CubDataset2(data.Dataset):
    def __init__(self, image_dir, list_path,input_transform=None):
        super(CubDataset2, self).__init__()
        self.input_transform = input_transform
        name_list = []
        label_list = []
        with open(list_path, 'r') as f:
            for line in f.readlines():
                imagename, class_label = line.split()
                name_list.append(imagename)
                label_list.append(int(class_label))
        self.image_filenames = [os.path.join(image_dir, x) for x in name_list]
        self.label_list = label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        class_label = self.label_list[index]
        return input, class_label,imagename

    def __len__(self):
        return len(self.image_filenames)