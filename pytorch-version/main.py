from img_for_regression import ImgForRegression
from module import BiONet
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomRotation, RandomResizedCrop, ToPILImage, RandomHorizontalFlip, RandomVerticalFlip
import sys
from torch.nn import BCELoss
from torch.optim import Adam
from torch import no_grad, device
import torch
from matplotlib import pyplot as plt
from csv import writer
if __name__ == "__main__":
    #  常量
    module_save_dict = ""
    optim_save_dict = ""
    train_set_dict = ""
    test_set_dict = ""
    test_example_result_dict = ""
    result_csv_path = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gray2b_threadhold = 127
    gray2b_table=[0 if i <= gray2b_threadhold else 1 for i in range(256)]
    #  训练和预测常量
    learn_rate = 0.01
    decay_rate = 0.00003
    batch_size_train = 2
    batch_size_test = 1
    epoch = 300
    pre_transform_init_train = Compose([])
    pre_transform_load_train = Compose([
        RandomRotation([-15, +15]),
        #  random shifting好像pytorch没有
        RandomResizedCrop((512, 512)),
        RandomHorizontalFlip(),
        RandomVerticalFlip()
    ])
    pre_transform_init_test = Compose([])
    pre_transform_load_test = Compose([])
    
    #  训练用对象
    net = BiONet(iterations=3, integrate=True)
    net.to(device)
    train_set = ImgForRegression(train_set_dict, pre_transform_init_train, pre_transform_load_train)
    test_set = ImgForRegression(test_set_dict, pre_transform_init_test, pre_transform_load_test)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)
    loss = BCELoss()
    optim = Adam(params=net.parameters(), lr=learn_rate, weight_decay=decay_rate)
    
    with open(result_csv_path, "w") as csv_file:
        csv_writer = writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "test_loss"])
        #  训练开始
        for epo in range(epoch):
            csv_row = []
            csv_row.append(epo)
            print("Epoch {0} Train".format(epo))
            all_loss = 0.0
            for train_batch, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                #训练部分
                all_loss = 0.0
                net.train()
                optim.zero_grad()
                output = net(x)
                l = loss(output, y)
                all_loss = all_loss+float(l)
                l.backward()
                optim.step()
            csv_row.append(all_loss)
            print("Loss {0}".format(all_loss))
            print("Epoch {0} Test".format(epo))
            all_loss = 0.0
            for test_batch, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                #  测试部分
                net.eval()
                with no_grad():
                    output = net(x)
                    l = loss(output, y)
                    all_loss = all_loss + float(l)
                    if test_batch == 0:
                        print("Saving Example Result")
                        x_img = ToPILImage()(x[0])
                        y_img = ToPILImage()(y[0])
                        output_img = (ToPILImage()(output[0])).point(gray2b_table, "1")
                        x_img.save("{0}\\{1}_input.png".format(test_example_result_dict, epo))
                        y_img.save("{0}\\{1}_truth.png".format(test_example_result_dict, epo))
                        output_img.save("{0}\\{1}_output.png".format(test_example_result_dict, epo))
            csv_row.append(all_loss)
            csv_writer.writerow(csv_row)
            print("Loss {0}".format(all_loss))
            print("Saving Module State")
            torch.save(net.state_dict(), "{0}\\{1}_module.dat".format(module_save_dict, epo))
            print("Saving Optim State")
            torch.save(optim.state_dict(), "{0}\\{1}_optim.dat".format(optim_save_dict, epo))