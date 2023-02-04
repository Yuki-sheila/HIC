import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_loss(network_output: torch.Tensor, target: torch.Tensor):
    # 交叉熵损失
    # 输出(b, 361, 16)
    network_output = torch.flatten(network_output, start_dim=0, end_dim=1)
    target = torch.flatten(target, start_dim=0, end_dim=1)
    real_labels = target
    loss = -torch.mul(real_labels, torch.log(network_output))#点乘
    pool_cross_entropy = torch.sum(loss)
    # F.binary_cross_entropy_with_logits()
    return pool_cross_entropy


def evaluate_performance(network_output, target, gt):
    network_output = torch.flatten(network_output, start_dim=0, end_dim=1)
    target = torch.flatten(target, start_dim=0, end_dim=1)
    gt = torch.flatten(gt, start_dim=0, end_dim=1)
    available_label_idx = (gt != 0).float()        # 有效标签的坐标,用于排除背景
    available_label_count = available_label_idx.sum()          # 有效标签的个数
    size = network_output.shape
    zeros = torch.zeros([size[0]]).to(device)
    correct_prediction = torch.where(torch.argmax(network_output, dim=1) == torch.argmax(target, dim=1),
                                     available_label_idx, zeros).sum()
    OA = correct_prediction.cpu() / available_label_count
    return OA

