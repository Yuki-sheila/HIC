import numpy as np
from torch.utils.data import Dataset, DataLoader
from DataLoad import dataset_process as dsp


class MyDataset(Dataset):
    def __init__(self, data, gt_onehot, gt, class_num, is_train=True):
        self.data = data
        self.height, self.width, self.bands = data.shape
        self.class_num = class_num
        self.gt_onehot = gt_onehot.reshape((self.height, self.width, self.class_num))
        self.gt = gt
        self.patch_size = 19
        if is_train:
            self.stride = 6
        else:
            self.stride = 18
        self.point_list = self.generate_patch_point_list(pacth_size=self.patch_size,
                                                         stride=self.stride)

    def __getitem__(self, idx):
        i, j = self.point_list[idx]
        data = self.data[i:i+self.patch_size, j:j+self.patch_size, :]
        gt_onehot = self.gt_onehot[i:i+self.patch_size, j:j+self.patch_size, :]
        gt_onehot_flatten = gt_onehot.reshape(self.patch_size ** 2, self.class_num)
        gt = self.gt[i:i+self.patch_size, j:j+self.patch_size]
        gt_flatten = gt.reshape(self.patch_size ** 2)
        return data, gt_onehot_flatten, gt_flatten

    def __len__(self):
        return len(self.point_list)

    def generate_patch_point_list(self, pacth_size, stride):
        point_list = []
        for i in range(0, self.height - pacth_size + 1, stride):
            for j in range(0, self.width - pacth_size + 1, stride):
                point = (i, j)
                point_list.append(point)
        # print(point_list)
        return point_list



if __name__ == "__main__":
    # 读取数据
    data_name = 'IP'
    data, labels = dsp.readData(data_name)
    data, pca = dsp.apply_PCA(data, num_components=32)
    # 数据可视化
    # dataset_visual(data_name, data, labels, save_img=True)
    # draw_gt(labels, data_name, save_img=True)
    height, width, bands = data.shape
    class_num = np.max(labels)
    print("数据集：", data_name)
    print("data：", data.shape)
    print("标签：", labels.shape)
    labels_flatten = labels.flatten()
    print("标签展平：", labels_flatten.shape)

    train_index, test_index = dsp.divide_data(labels_flatten, class_num=16, train_num=100)

    print("训练数据索引：", train_index.shape)
    train_label_flatten, test_label_flatten = dsp.idx2label(labels_flatten, train_index, test_index)
    print("训练数据标签展平：", train_label_flatten.shape)

    train_gt = np.reshape(train_label_flatten, (height, width))
    test_gt = np.reshape(test_label_flatten, (height, width))
    print("train_gt:", train_gt.shape)
    # 划分数据集后可视化
    # data_info(train_gt, test_gt)
    # draw_gt(train_gt, 'train', save_img=True)
    # draw_gt(test_gt, 'test', save_img=True)

    # 标签one-hot encode
    # (21025, 16)
    train_gt_onehot = dsp.label2one_hot(train_gt, class_num)
    test_gt_onehot = dsp.label2one_hot(test_gt, class_num)
    print("train_gt_onehot", train_gt_onehot.shape)


    train = MyDataset(data, train_gt_onehot, train_gt, class_num, is_train=True)
    test = MyDataset(data, test_gt_onehot, train_gt, class_num, is_train=False)

    data_loader_train = DataLoader(dataset=train, batch_size=128, shuffle=True)
    data_loader_test = DataLoader(dataset=test, batch_size=64)
    for i_batch, (batch_data, batch_label, batch_gt) in enumerate(data_loader_train):
        print(i_batch)
        print(batch_data.shape)
        print(batch_label.shape)
        print(batch_gt.shape)

