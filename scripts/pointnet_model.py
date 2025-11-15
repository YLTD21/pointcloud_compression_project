# scripts/pointnet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet2Seg(nn.Module):
    """修复的PointNet++语义分割模型"""

    def __init__(self, num_classes=3):
        super(PointNet2Seg, self).__init__()

        # 第一层：输入只有坐标信息 (3通道)
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                          in_channel=0, mlp=[64, 64, 128])

        # 第二层：输入有坐标+128维特征 (3+128=131通道)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128, mlp=[128, 128, 256])

        # 第三层：全局特征
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256, mlp=[256, 512, 1024], group_all=True)

        # 特征传播层
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])  # 256 + 1024
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])  # 128 + 256
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128])  # 0 + 128

        # 分割头
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        # xyz: [B, N, 3]
        B, N, _ = xyz.shape

        # 编码路径
        l0_xyz = xyz
        l0_points = None  # 第一层没有额外的特征

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B, 512, 3], [B, 128, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 128, 3], [B, 256, 128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 1, 3], [B, 1024, 1]

        # 解码路径
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B, 256, 128]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B, 128, 512]
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)  # [B, 128, N]

        # 分类头
        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(x)
        x = self.conv2(x)  # [B, num_classes, N]

        return x


class PointNetSetAbstraction(nn.Module):
    """集合抽象层"""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        # 第一层卷积的输入通道：坐标(3) + 输入特征(in_channel)
        first_conv_in_channel = 3 + in_channel

        last_channel = first_conv_in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: 输入点云 [B, N, 3]
            points: 输入特征 [B, C, N] 或 None
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        # new_points: [B, 3+C, nsample, npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # [B, C, npoint]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """特征传播层"""

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: 前一层的点 [B, N, 3]
            xyz2: 插值的点 [B, S, 3]
            points1: 前一层的特征 [B, C, N]
            points2: 插值的特征 [B, D, S]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # 全局特征传播
            interpolated_points = points2.repeat(1, 1, N)
        else:
            # 三线性插值
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            # 索引特征点
            interpolated_points = torch.sum(
                index_points(points2.transpose(1, 2), idx) * weight.view(B, N, 3, 1),
                dim=2
            )
            interpolated_points = interpolated_points.transpose(1, 2)  # [B, D, N]

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    """采样和分组"""
    B, N, C = xyz.shape

    # 最远点采样
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]

    # 球查询
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)  # 相对坐标

    if points is not None:
        # points: [B, C, N] -> [B, N, C]
        points_t = points.transpose(1, 2)
        grouped_points = index_points(points_t, idx)  # [B, npoint, nsample, C]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+C]
    else:
        new_points = grouped_xyz_norm

    new_points = new_points.permute(0, 3, 2, 1)  # [B, 3+C, nsample, npoint]
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """全局采样和分组"""
    B, N, C = xyz.shape
    device = xyz.device

    new_xyz = torch.zeros(B, 1, C).to(device)

    if points is not None:
        # points: [B, C, N] -> [B, N, C]
        points_t = points.transpose(1, 2)
        grouped_points = points_t.view(B, 1, N, -1)
        grouped_xyz = xyz.view(B, 1, N, C)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # [B, 1, N, 3+C]
    else:
        new_points = xyz.view(B, 1, N, C)

    new_points = new_points.permute(0, 3, 2, 1)  # [B, 3+C, N, 1]
    return new_xyz, new_points


def farthest_point_sample(xyz, npoint):
    """最远点采样"""
    device = xyz.device
    B, N, C = xyz.shape

    if npoint is None:
        return torch.arange(N, device=device).expand(B, N)

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """球查询"""
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]

    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, M, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def square_distance(src, dst):
    """计算平方距离"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """索引点"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def get_model(num_classes=3):
    """获取模型实例"""
    return PointNet2Seg(num_classes=num_classes)