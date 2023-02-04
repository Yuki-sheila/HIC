import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import torch
import time
import yaml
from torch import nn
from torch.autograd import Variable
from Model import utils, FDGC
from DataLoad import dataset_process as dsp
from DataLoad import dataload as dl
from torch.utils.data import DataLoader


# 参数设置
parser = argparse.ArgumentParser(description='FDGC')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--path-config', type=str, default='D:\PycharmProjects\HIC\FDGC\Config\config.yaml')
parser.add_argument('-pc', '--print-config', action='store_true', default=False)
parser.add_argument('-pdi','--print-data-info', action='store_true', default=False)
parser.add_argument('-sr','--show-results', action='store_true', default=False)
parser.add_argument('--save-results', action='store_true', default=True)
args = parser.parse_args()  # running in command line


# 加载配置文件
config = yaml.load(open(args.path_config, "r"), Loader=yaml.FullLoader)
dataset_name = config["data_input"]["dataset_name"]
patch_size = config["data_input"]["patch_size"]
train_num = config["data_split"]["train_num"]
max_epoch = config["network_config"]["max_epoch"]
learning_rate = config["network_config"]["learning_rate"]
lr_decay = config["network_config"]["lr_decay"]
lb_smooth = config["network_config"]["lb_smooth"]
path_weight = config["result_output"]["path_weight"]
path_result = config["result_output"]["path_result"]

if args.print_config:
    print(config)


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


# 划分数据集
train_index, test_index = dsp.divide_data(labels_flatten, class_num, train_num)
print("训练数据索引：", train_index.shape)
train_label_flatten, test_label_flatten = dsp.idx2label(labels_flatten, train_index, test_index)
print("训练数据标签展平：", train_label_flatten.shape)


# 划分数据集后可视化
train_gt = np.reshape(train_label_flatten, (height, width))
test_gt = np.reshape(test_label_flatten, (height, width))
if args.print_data_info:
    dsp.data_info(train_gt, test_gt)


# 标签转one-hot（二维的，展开的）
train_gt_onehot = dsp.label2one_hot(train_gt, class_num)
test_gt_onehot = dsp.label2one_hot(test_gt, class_num)
print("train_gt_onehot：", train_gt_onehot.shape)


# 切patch，装batch
train = dl.MyDataset(data, train_gt_onehot, train_gt, class_num, is_train=True)
test = dl.MyDataset(data, test_gt_onehot, test_gt, class_num, is_train=False)
data_loader_train = DataLoader(dataset=train, batch_size=128, shuffle=True)
data_loader_test = DataLoader(dataset=test, batch_size=64)


# Model
net = FDGC.FDGC(height=patch_size, width=patch_size, channel=bands, class_num=class_num).to(args.device)


# train
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
# criterion = nn.CrossEntropyLoss().to(args.device)
best_OA = 0
net.train()
time0 = time.time()
for epoch in range(max_epoch):
    print("\n==============第 {} 轮训练==============".format(epoch + 1))
    for batch_id, (data_input, target, gt) in enumerate(data_loader_train):
        data_input, target = Variable(data_input).to(args.device), Variable(target).to(args.device)
        gt = Variable(gt).to(args.device)

        #学习率衰减
        if epoch == 50 or epoch == 83:
        #if epoch % 10 == 0:
            for p in optimizer.param_groups:  # 更新每个group里的参数lr
                p['lr'] *= lr_decay

        # 清空梯度
        optimizer.zero_grad()
        output = net(data_input)

        # 计算loss
        loss = utils.compute_loss(output, target)
        # loss = criterion()
        loss.backward(retain_graph=False)

        # 更新权重
        optimizer.step()
        print('[{}/{} ({:.0%})]\ttrain loss: {:.4f}'.format((batch_id + 1) * len(data_input),
                            len(data_loader_train.dataset), (batch_id + 1) / len(data_loader_train), loss))
    # 验证
    with torch.no_grad():
        net.eval()
        for (test_input, test_target, test_gt) in data_loader_test:
            test_input = test_input.to(args.device)
            test_target = test_target.to(args.device)
            test_gt = test_gt.to(args.device)
            output = net(test_input)
            testloss = utils.compute_loss(output, test_target)
            testOA = utils.evaluate_performance(output, test_target, test_gt)

        if testOA > best_OA:
            best_OA = testOA
            torch.save(net.state_dict(), path_weight + r"model.pt")
            print('save Model...')

    torch.cuda.empty_cache()
    print("val loss：{:.4f}\tval OA：{:.4%}".format(testloss, testOA))

time1 = time.time()


# test
print("\n\n==============starting testing===============\n")
torch.cuda.empty_cache()
net_output = torch.zeros((1))
net_gt = torch.zeros((1))
time2 = time.time()
with torch.no_grad():
    net.load_state_dict(torch.load(path_weight + r"model.pt"))
    net.eval()
    for (test_input, test_target, test_gt) in data_loader_test:
        test_input = test_input.to(args.device)
        test_target = test_target.to(args.device)
        test_gt = test_gt.to(args.device)
        output = net(test_input)
        testloss = utils.compute_loss(output, test_target)
        testOA = utils.evaluate_performance(output, test_target, test_gt)
        net_output = output
        net_gt = test_gt
    print("test loss={:.4f}\t test OA={:.4%}".format(testloss, testOA))
time3 = time.time()
torch.cuda.empty_cache()
del net


training_time = time1 - time0
testing_time = time3 - time2


# classification report
gt_flatten = torch.flatten(net_gt, start_dim=0, end_dim=1).cpu().numpy().astype('int64')
test_label_mask = (gt_flatten != 0)
net_output = torch.flatten(net_output, start_dim=0, end_dim=1)
predict = torch.argmax(net_output, dim=1).cpu().numpy()

classification = classification_report(gt_flatten[test_label_mask],
                                       predict[test_label_mask]+1, digits=4)

kappa = cohen_kappa_score(gt_flatten[test_label_mask], predict[test_label_mask]+1)

if args.show_results:
    print(classification, kappa)


# 保存结果
if args.save_results:
    print("save Results")
    run_date = time.strftime('%Y%m%d-%H%M-', time.localtime(time.time()))
    f = open(path_result + run_date +dataset_name + '.txt', 'a+')
    str_results = '\n ======================' \
                + '\nrun data = ' + run_date \
                + "\nlearning rate = " + str(learning_rate) \
                + "\nepochs = " + str(max_epoch) \
                + "\ntrain num = " + str(train_num) \
                + '\ntrain time = ' + str(training_time) \
                + '\ntest time = ' + str(testing_time) \
                + '\n' + classification \
                + "kappa = " + str(kappa) \
                + '\n'
    f.write(str_results)
    f.close()