from DeepLabmodel import DeepLab
import torch
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset,DataLoader
#from torch.utils.tensorboard import SummaryWriter
from dl_utils import BinaryDiceLoss,openDataset,loss_joint,DiceLoss
from GNDeepLabmodel import GNDeepLab
from BDILDeepLabmodel import BNDDeepLab
import segmentation_models_pytorch as smp

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(train_img, train_label, opt_name, train_epoch, model_path, last_model):
    # 加载数据
    #writer = SummaryWriter()
    dataset = openDataset(train_img, train_label, 'softmax')
    train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=0)
    # 定义网络
    model = BNDDeepLab(in_ch=3, num_classes=5,backbone="resnet34", downsample_factor=16)
    # 选择优化器
    if opt_name == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    # 余弦退火优化
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0=2,  # 学习率第一次回到初始值的epoch位置
        T_mult=2,  # 控制学习率的变化速度,T_mult=2意思是周期翻倍，第一个周期是1，则第二个周期是2，第三个周期是4,第四个周期是8，16，32，64
        eta_min=1e-6  # 最低学习率,如果T_mult>1,则学习率在T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,.....,(1+T_mult+T_mult**2+...+T_0**i)*T0
    )
    #model.to(device)
    # 只用diceloss训练会很暴躁可能不收敛，采用diceloss和ce相结合
    diceloss = DiceLoss()
    CE = torch.nn.BCEWithLogitsLoss()
    #BCE = torch.nn.BCEWithLogitsLoss()
    #Binary_criterion = loss_joint(diceloss, BCE, [0.5, 0.5])
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
            # 在用交叉熵损失函数时可以不用对label进行one hot转换，直接用三维【batch，heigh，witch】输入即可
            loss = diceloss(pred, label)
            loss.requires_grad_(True)
            loss.backward()
            opt.step()
            loss_number = loss.item()
            # 保存最优模型
            if loss_number < min_loss:
                min_loss = loss_number
                print("save model")
                torch.save(model.state_dict(), model_path)
            print('finish')
            #writer.add_scalar('Loss/train', loss_number, all_train)
            all_train = all_train + 1
            loss_number_sum = loss_number_sum + loss_number
        scheduler.step()
        #writer.add_scalar('loss/epoch', loss_number_sum/(a+1), i)
        losslist.append(loss_number_sum/(a+1))
        epoch.append(i)
        print('finish' + str(i) + 'epoch')
    torch.save(model.state_dict(), last_model)
    plt.plot(epoch, losslist)
    plt.show()

if __name__ == '__main__':
    star = time.time()
    # 最优模型保存路径
    best_model_path = r'C:\Users\cxd\Desktop\model\best_model'
    # 最后一个模型路径
    last_model = r'C:\Users\cxd\Desktop\model\last_model'
    train_img = r'C:\Users\cxd\Desktop\训练'
    train_label = r'C:\Users\cxd\Desktop\标签'
    train(train_img, train_label, 'adwas',126, best_model_path, last_model)