# scripts/train_semantic_segmentation.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from pathlib import Path
import sys
from tqdm import tqdm
import datetime

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pointnet2_complete import get_complete_model
from pointnet_dataloader import get_data_loaders


class SemanticSegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 使用CrossEntropyLoss，忽略背景类(0)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []

        # 训练统计
        self.start_time = None
        self.epoch_times = []

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        # 创建进度条
        pbar = tqdm(total=len(self.train_loader),
                    desc=f'Epoch {epoch + 1}/{total_epochs}',
                    ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, (points, labels) in enumerate(self.train_loader):
            points = points.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(points)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * points.size(0)
            total_samples += points.size(0)

            # 更新进度条
            current_loss = running_loss / total_samples
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            pbar.update(1)

        pbar.close()
        epoch_loss = running_loss / total_samples
        return epoch_loss

    def validate(self, epoch, total_epochs):
        self.model.eval()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 验证进度条
        pbar = tqdm(total=len(self.val_loader),
                    desc=f'Validation {epoch + 1}/{total_epochs}',
                    ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        with torch.no_grad():
            for points, labels in self.val_loader:
                points = points.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(points)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * points.size(0)

                # 计算准确率
                pred = outputs.argmax(dim=1)
                mask = labels > 0  # 只考虑车辆(1)和行人(2)

                if mask.sum() > 0:
                    correct = (pred[mask] == labels[mask]).sum().item()
                    total_correct += correct
                    total_samples += mask.sum().item()

                pbar.update(1)

        pbar.close()

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = total_correct / total_samples if total_samples > 0 else 0

        return val_loss, val_acc

    def print_training_info(self):
        """打印训练信息"""
        print("\n" + "=" * 80)
        print("🏋️‍♂️ 点云语义分割训练开始")
        print("=" * 80)
        print(f"📊 设备: {self.device}")
        print(f"📈 训练样本: {len(self.train_loader.dataset):,}")
        print(f"📉 验证样本: {len(self.val_loader.dataset):,}")
        print(f"🔢 Batch size: {self.train_loader.batch_size}")
        print(f"📍 每样本点数: {self.train_loader.dataset.num_points}")
        print(f"🔄 总Epoch数: {self.epochs}")

        # 模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 模型参数: {total_params:,}")

        # 测试维度
        test_points, test_labels = next(iter(self.train_loader))
        test_points = test_points.to(self.device)
        test_outputs = self.model(test_points)
        print(f"📐 维度检查 - 输入: {test_points.shape}, 输出: {test_outputs.shape}, 标签: {test_labels.shape}")
        print("=" * 80 + "\n")

    def print_epoch_summary(self, epoch, train_loss, val_loss, val_acc, epoch_time, total_epochs):
        """打印epoch总结"""
        # 计算剩余时间
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = total_epochs - epoch - 1
        remaining_time = avg_epoch_time * remaining_epochs
        remaining_str = str(datetime.timedelta(seconds=int(remaining_time)))

        # 计算总训练时间
        total_time = time.time() - self.start_time
        total_str = str(datetime.timedelta(seconds=int(total_time)))

        print(f"\n📊 Epoch {epoch + 1}/{total_epochs} 总结:")
        print(f"   ⏱️  本轮时间: {epoch_time:.1f}s")
        print(f"   ⏳ 总训练时间: {total_str}")
        print(f"   🎯 剩余时间: {remaining_str}")
        print(f"   📉 训练损失: {train_loss:.4f}")
        print(f"   📊 验证损失: {val_loss:.4f}")
        print(f"   🎯 验证准确率: {val_acc:.4f}")
        print(f"   📈 最佳准确率: {self.best_val_acc:.4f}")
        print(f"   🔧 学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

    def train(self, epochs=50, save_dir='checkpoints'):
        self.epochs = epochs
        self.start_time = time.time()
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # 打印训练信息
        self.print_training_info()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # 训练
            train_loss = self.train_epoch(epoch, epochs)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, val_acc = self.validate(epoch, epochs)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])

            # 学习率调度
            self.scheduler.step()

            # 计算epoch时间
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)

            # 打印epoch总结
            self.print_epoch_summary(epoch, train_loss, val_loss, val_acc, epoch_time, epochs)

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_loss': train_loss
                }, save_dir / 'best_model.pth')
                print(f"   💾 保存最佳模型! 准确率: {val_acc:.4f}")

            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies
                }, checkpoint_path)
                print(f"   💾 保存检查点: {checkpoint_path}")

            print("-" * 60)

        # 训练完成
        self.print_final_summary()

    def print_final_summary(self):
        """打印最终总结"""
        total_time = time.time() - self.start_time
        total_str = str(datetime.timedelta(seconds=int(total_time)))

        print("\n" + "=" * 80)
        print("🎉 训练完成!")
        print("=" * 80)
        print(f"⏱️  总训练时间: {total_str}")
        print(f"📈 最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"📉 最佳验证损失: {self.best_val_loss:.4f}")
        print(f"🔄 总Epoch数: {self.epochs}")
        print(f"📊 最终训练损失: {self.train_losses[-1]:.4f}")
        print(f"📊 最终验证损失: {self.val_losses[-1]:.4f}")
        print("=" * 80)

        # 保存训练历史
        self.save_training_history()

    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss
        }

        history_path = Path('checkpoints/training_history.npy')
        np.save(history_path, history)
        print(f"📁 训练历史已保存: {history_path}")


def plot_training_curves(train_losses, val_losses, val_accuracies, learning_rates):
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 损失曲线
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('训练和验证损失')

        # 准确率曲线
        ax2.plot(epochs, val_accuracies, 'g-', label='验证准确率', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('验证准确率')
        ax2.set_ylim(0, 1)

        # 学习率曲线
        ax3.plot(epochs, learning_rates, 'purple', label='学习率', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title('学习率变化')
        ax3.set_yscale('log')

        # 损失和准确率对比
        ax4.plot(val_losses, val_accuracies, 'o-', color='orange', linewidth=2)
        ax4.set_xlabel('验证损失')
        ax4.set_ylabel('验证准确率')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('损失 vs 准确率')

        plt.tight_layout()
        plt.savefig('training_curves_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 训练曲线图已保存: training_curves_detailed.png")

    except ImportError:
        print("⚠️  Matplotlib未安装，跳过绘图")


def main():
    # 参数设置
    batch_size = 4
    num_points = 2048
    epochs = 50

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size, num_points=num_points
    )

    # 使用完整的PointNet++模型
    model = get_complete_model(num_classes=3)

    # 训练器
    trainer = SemanticSegmentationTrainer(model, train_loader, val_loader, device)

    # 开始训练
    train_losses, val_losses, val_accuracies = trainer.train(epochs=epochs)

    # 绘制详细的训练曲线
    plot_training_curves(train_losses, val_losses, val_accuracies, trainer.learning_rates)


if __name__ == "__main__":
    main()