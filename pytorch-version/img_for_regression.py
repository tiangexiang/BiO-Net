from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize,ToTensor
from os.path import join, isfile
from os import listdir
from torchvision.transforms import Compose
from torchvision import transforms
from PIL import Image
from torch import cat

#  本实验中不考虑输入输出中的错误情况，不进行稳定性测试
#  本类假定在folder_path下存在x和y两个文件夹，每个x文件夹数据和y文件夹数据在文件名上一一对应
#  本类认为x文件夹中的数据最终会回归成y文件夹的数据


class ImgForRegression(Dataset):
    
    #  特别注意，此处的transform不需要指定ToTensor！！！
    def __init__(self, folder_path: str, pre_transform_init: transforms = Compose([]), pre_transform_load: transforms = Compose([])):
        super(ImgForRegression, self).__init__()
        #  定义加载预处理和使用预处理
        self.pre_transform_init = pre_transform_init
        self.pre_transform_load = pre_transform_load
        #  x和y的位置
        x_folder = join(folder_path, "x")
        y_folder = join(folder_path, "y")
        #  完整文件路径，理论上下边两个列表应该是完全一样的
        x_file_name_list = listdir(x_folder)
        y_file_name_list = listdir(y_folder)
        #  获取所有的文件，以(x, y)的列表表示
        self.all_image_file = []
        for file_name in x_file_name_list:
            assert file_name in y_file_name_list
            x_file_path = join(x_folder, file_name)
            y_file_path = join(y_folder, file_name)
            x_image = Image.open(x_file_path).convert("RGB")
            y_image = Image.open(y_file_path).convert("L")
            x_tensor = ToTensor()(x_image)
            y_tensor = ToTensor()(y_image)
            all_pic = self.pre_transform_init(cat([x_tensor, y_tensor], dim=0))
            x_transed_tensor, y_transed_tensor = all_pic[0:3], all_pic[3:4]  # 这里还有问题，因为x和y的变换当且仅当变换是形状和位置变换必须是一样的，对于随机处理的变换目前还没有办法
            self.all_image_file.append((x_transed_tensor, y_transed_tensor))
            
    def __len__(self):
        return len(self.all_image_file)
    
    def __getitem__(self, index: int):
        all_pic = self.pre_transform_load(cat([self.all_image_file[index][0], self.all_image_file[index][1]], dim=0))
        x_transed_tensor, y_transed_tensor = all_pic[0:3], all_pic[3:4]
        return x_transed_tensor, y_transed_tensor



