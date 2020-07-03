import os
from PIL import Image, ImageOps
import pickle
import numpy as np
import torchvision.transforms as transforms
from scripts.predict import load_model
import torch
from torch.utils.data import DataLoader

class LogoData():

    def __init__(self, transform=None, train=True):

        self.transform = transform
        self.train = train
        self.path = './targetlist_updated_clean'
        self.data, self.label = self.train_loader()  ## get all data and labels
        self.num_classes = len(set(self.label))
        self.data_group = self.label_group()

    def train_loader(self):  ## load all the data
        print("begin loading dataset")
        label = []
        data = []

        for brand in os.listdir(self.path):
            if brand.startswith('.'): # skip hidden files
                continue
            for file in os.listdir(os.path.join(self.path, brand)):
                if file.endswith('.png') and not file.startswith('loginpage') and not file.startswith('homepage'):  ## protected logos
                    img = Image.open(os.path.join(self.path, brand, file)).convert('RGB')
                    img = ImageOps.expand(img, ((max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
                                                (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2),
                                          fill=(255, 255, 255))
                    img = img.resize((128, 128))
                    data.append(np.array(img))
                    label.append(brand)

        with open('./data/targetlist_labeldict.pkl', 'rb') as handle:
            label_dict = pickle.load(handle)
        label = [label_dict[x] for x in label]
        print("finish loading dataset")
        return np.array(data), np.array(label)

    def label_group(self):  ## grouping images from the same label together
        label_unique = set(self.label)
        data_group = {}
        for i in label_unique:
            data_group[str(i)] = self.data[np.where(self.label == i)[0].tolist(), :, :]
        return data_group

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if self.transform:
            image = self.transform(Image.fromarray(self.data[index]))

        return image, self.label[index]


if __name__ == '__main__':

    '''define dataloader'''
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std),
     ])

    test_data = LogoData(transform=transform_test, train=False)
    testloader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=8)

    '''initialize model'''
    classes = 180
    modelpath = './model/rgb_ar.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(classes, modelpath)
    model.to(device)
    model.eval()

    '''get all prediction'''
    pred_prob = torch.tensor([], device='cuda:0')
    targets = torch.tensor([], device='cuda:0')
    pred_feat = torch.tensor([], device='cuda:0')

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.float)
            pred_feat = torch.cat((pred_feat, model.features(inputs)), 0)
            pred_prob = torch.cat((pred_prob, model(inputs)), 0)
            targets = torch.cat((targets, labels), 0)

    _, pred_cls = torch.max(pred_prob, 1)

    print(pred_feat.shape)
    print(pred_prob.shape)
    print(pred_cls.shape)
