from DeepLabmodel import DeepLab
import torch
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset,DataLoader
#from torch.utils.tensorboard import SummaryWriter
from dl_utils import BinaryDiceLoss,openDataset,loss_joint,DiceLoss
from Attention DeepLab import BNDDeepLab

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(train_img, train_label, opt_name, train_epoch, model_path, last_model):
    #writer = SummaryWriter()
    dataset = openDataset(train_img, train_label, 'softmax')
    train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=0)
    model = BNDDeepLab(in_ch=12, num_classes=2,backbone="resnet34", downsample_factor=16)
    if opt_name == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0=2,  
        T_mult=2, 
        eta_min=1e-6  
    )
    #model.to(device)
    diceloss = DiceLoss()
    CE = torch.nn.BCEWithLogitsLoss()
    BCE = torch.nn.BCEWithLogitsLoss()
    Binary_criterion = loss_joint(diceloss, BCE, [0.5, 0.5])
    losslist = []
    epoch = []
    min_loss = 1000
    all_train = 0
    for i in range(train_epoch):
        loss_number_sum = 0
        model.train()
        for a, train_data in enumerate(train_loader):
            data, label = train_data
            #data = data.to(device)
            #label = label.to(device)
            opt.zero_grad()
            pred = model(data)
            loss = diceloss(pred, label)
            loss.requires_grad_(True)
            loss.backward()
            opt.step()
            loss_number = loss.item()
            if loss_number < min_loss:
                min_loss = loss_number
                print("save model")
                torch.save(model, model_path)
            print('finish')
            #writer.add_scalar('Loss/train', loss_number, all_train)
            all_train = all_train + 1
            loss_number_sum = loss_number_sum + loss_number
        scheduler.step()
        #writer.add_scalar('loss/epoch', loss_number_sum/(a+1), i)
        losslist.append(loss_number_sum/(a+1))
        epoch.append(i)
        print('finish' + str(i) + 'epoch')
    #torch.save(model.state_dict(), last_model)
    torch.save(model, last_model)  
    plt.plot(epoch, losslist)
    plt.show()

if __name__ == '__main__':
    star = time.time()
    best_model_path = r''
    last_model = r''
    train_img = r''
    train_label = r''
    train(train_img, train_label, 'adwas',1, best_model_path, last_model)
