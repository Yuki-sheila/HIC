import numpy as np
import os
import scipy.io as sio
import spectral as spy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

# data(145, 145, 200)
# labels(145, 145)
def readData(name):
    #默认name == 'IP'
    raw_data = sio.loadmat(r"D:\PycharmProjects\HIC\FDGC\Data\Indian_pines_corrected.mat")
    data = raw_data["indian_pines_corrected"].astype(np.float32)
    labels = sio.loadmat(r"D:\PycharmProjects\HIC\FDGC\Data\Indian_pines_gt.mat")
    labels = labels["indian_pines_gt"].astype(np.int64)
    data = data_normal(data)
    return data, labels


def data_normal(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def divide_data(labels_flatten, class_num, train_num):
    train_index = []
    test_index = []

    sample_num = train_num
    for i in range(class_num):
        idx = np.where(labels_flatten == i + 1)[0]
        #print(idx)
        #idx = np.where(labels_flatten == i + 1)[0]
        count = len(idx)
        # print("Class ", i + 1, ":", count)
        np.random.shuffle(idx)
        sample_num = 15 if sample_num > count else train_num
        # 取出每个类别选择出的训练集
        train_index.append(idx[: sample_num])
        test_index.append(idx[sample_num :])

    train_index = np.concatenate(train_index, axis=0)
    test_index = np.concatenate(test_index, axis=0)
    return train_index, test_index


def idx2label(label_flatten, train_index, test_index):
    train_label_flatten = np.zeros(label_flatten.shape)
    for i in range(len(train_index)):
        train_label_flatten[train_index[i]] = label_flatten[train_index[i]]

    test_label_flatten = np.zeros(label_flatten.shape)
    for i in range(len(test_index)):
        test_label_flatten[test_index[i]] = label_flatten[test_index[i]]
    return train_label_flatten, test_label_flatten


def data_info(train_label=None, test_label=None, start=1):
    # 类别数
    class_num = np.max(train_label.astype('int32'))
    if train_label is not None and test_label is not None:
        total_train_pixel = 0
        total_test_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        test_mat_num = Counter(test_label.flatten())
        for i in range(start, class_num+1):
            print("class", i, "\t", train_mat_num[i],"\t", test_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_test_pixel += test_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_test_pixel)
    
    elif train_label is not None:
        total_pixel = 0
        # 计数,返回的是字典
        data_mat_num = Counter(train_label.flatten())
        for i in range(start, class_num+1):
            print("class", i, "\t", data_mat_num[i])
            total_pixel += data_mat_num[i]
        print("total:   ", total_pixel)

    else:
        raise ValueError("labels are None")


def dataset_visual(data_name, data, labels, save_img=False):
    img_data = spy.imshow(data, bands=(30,20,10), classes=labels)
    img_data.set_display_mode('data')
    img_data.class_alpha = 0.5

    img_classes = spy.imshow(data, bands=(30,20,10), classes=labels)
    img_classes.set_display_mode('classes')
    img_classes.class_alpha = 0.5

    img_overlay = spy.imshow(data, bands=(30,20,10), classes=labels)
    img_overlay.set_display_mode('overlay')
    img_overlay.class_alpha = 0.5
    #plt.pause(20)

    if save_img:
        save_path = r'D:\PycharmProjects\HIC\FDGC\DataSetVisual'
        spy.save_rgb(os.path.join(save_path, str(data_name)+"_rgb.png"), data, bands=[30, 20, 10])
        spy.save_rgb(os.path.join(save_path, str(data_name)+"_gt.png"), labels, colors=spy.spy_colors)


def draw_gt(label, data_name, scale: float = 4.0, dpi: int = 400, save_img=False):
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        save_path = r'D:\PycharmProjects\HIC\FDGC\DataSetVisual'
        foo_fig.savefig(os.path.join(save_path, str(data_name) + '_gt_scale.png'), format='png', transparent=True, dpi=dpi, pad_inches=0)


def label2one_hot(labels, class_num, weight=0.01):
    # 标签平滑
    height, width = labels.shape
    one_hot_label = []
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_num, dtype=np.float32)
            class_id = int(labels[i, j])
            if class_id != 0:
                temp[class_id - 1] = 1
                temp = (1 - weight) * temp + weight * (1 / class_num)
            one_hot_label.append(temp)
    one_hot_label = np.reshape(one_hot_label, (height * width, class_num))
    # (21025, 16)
    return one_hot_label


# PCA降维
def apply_PCA(data, num_components=100):
    pca = PCA(n_components=num_components, whiten=True)
    # (145x145, 200)
    new_data = np.reshape(data, (-1, data.shape[2]))
    # (145x145, n)
    new_data = pca.fit_transform(new_data)
    # (145, 145, n)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca


if __name__ == "__main__":
    # 读取数据
    data_name = 'IP'
    data, labels = readData(data_name)
    data, pca = apply_PCA(data, num_components=32)
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

    train_index, test_index = divide_data(labels_flatten, class_num=16, train_num=100)

    print("训练数据索引：", train_index.shape)
    train_label_flatten, test_label_flatten = idx2label(labels_flatten, train_index, test_index)
    print("训练数据标签展平：", train_label_flatten.shape)

    train_gt = np.reshape(train_label_flatten, (height, width))
    test_gt = np.reshape(test_label_flatten, (height, width))
    # 划分数据集后可视化
    data_info(train_gt, test_gt)
    # draw_gt(train_gt, 'train', save_img=True)
    # draw_gt(test_gt, 'test', save_img=True)

    # 标签one-hot encode
    # (21025, 16)
    train_gt_onehot_fla = label2one_hot(train_gt, class_num)
    test_gt_onehot_fla = label2one_hot(test_gt, class_num)
    print("train_gt_onehot_fla", train_gt_onehot_fla.shape)

    train_gt_onehot = train_gt_onehot_fla.reshape((height, width, class_num))
    print("train_gt_onehot", train_gt_onehot.shape)





