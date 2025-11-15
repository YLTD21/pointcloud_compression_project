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
try:
    from gpu_optimized_dataloader import get_gpu_optimized_loaders
    GPU_OPTIMIZED_AVAILABLE = True
except ImportError:
    from pointnet_dataloader import get_data_loaders
    GPU_OPTIMIZED_AVAILABLE = False
    print("⚠️  GPU优化数据加载器不可用，使用普通版本")

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
        self.batch_times = []
        self.data_loading_times = []

    def get_detailed_gpu_info(self):
        """获取详细的GPU信息"""
        if not torch.cuda.is_available():
            return "CPU模式"

        gpu_info = {}
        try:
            # 内存信息
            gpu_info['allocated_gb'] = torch.cuda.memory_allocated() / 1024 ** 3
            gpu_info['reserved_gb'] = torch.cuda.memory_reserved() / 1024 ** 3
            gpu_info['max_allocated_gb'] = torch.cuda.max_memory_allocated() / 1024 ** 3

            # 利用率和温度（需要nvidia-smi）
            import subprocess
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                util, temp = result.stdout.strip().split(', ')
                gpu_info['utilization'] = f"{util}%"
                gpu_info['temperature'] = f"{temp}°C"
            else:
                gpu_info['utilization'] = "N/A"
                gpu_info['temperature'] = "N/A"

        except Exception as e:
            gpu_info['error'] = str(e)

        return gpu_info
    # 在 train_semantic_segmentation.py 中修改以下部分

    # 在 train_semantic_segmentation.py 中修改以下部分

    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        # 重置GPU内存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024 ** 3

        # 性能计时
        epoch_start_time = time.time()
        data_loading_time = 0
        computation_time = 0

        pbar = tqdm(total=len(self.train_loader),
                    desc=f'Epoch {epoch + 1}/{total_epochs}',
                    ncols=120,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        # 混合精度设置
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler('cuda')
            autocast_device = 'cuda'
        else:
            scaler = None
            autocast_device = 'cpu'

        for batch_idx, (points, labels) in enumerate(self.train_loader):
            batch_data_loading_time = time.time()
            data_loading_time += batch_data_loading_time - epoch_start_time

            # 数据传输
            points = points.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # 前向传播
            batch_computation_start = time.time()

            if scaler is not None:
                with torch.amp.autocast(autocast_device):
                    outputs = self.model(points)
                    loss = self.criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(points)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            computation_time += time.time() - batch_computation_start

            running_loss += loss.item() * points.size(0)
            total_samples += points.size(0)

            # 更新进度条
            current_loss = running_loss / total_samples

            # 获取详细的GPU信息
            gpu_info = self.get_detailed_gpu_info()
            memory_allocated = gpu_info.get('allocated_gb', 0)

            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'gpu_mem': f'{memory_allocated:.1f}GB',
                'data_time': f'{data_loading_time / (batch_idx + 1):.3f}s',
                'comp_time': f'{computation_time / (batch_idx + 1):.3f}s'
            })
            pbar.update(1)

            epoch_start_time = time.time()  # 重置计时

        pbar.close()

        # 打印详细的性能统计
        epoch_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        self.batch_times.append(epoch_time)

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"💾 GPU内存: 初始 {initial_memory:.1f}GB, 峰值 {peak_memory:.1f}GB")
            print(f"⏱️  数据加载: {data_loading_time:.2f}s, 计算: {computation_time:.2f}s")

        epoch_loss = running_loss / total_samples
        return epoch_loss

    def validate(self, epoch, total_epochs):
        self.model.eval()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_points = 0

        # 使用更流畅的进度条
        from alive_progress import alive_bar

        print(f"🔍 验证 Epoch {epoch + 1}/{total_epochs}")

        with torch.no_grad():
            # 使用 alive_bar 替代 tqdm
            with alive_bar(len(self.val_loader),
                           title='验证进度',
                           bar='smooth',
                           spinner='dots',
                           length=50,
                           stats=False,  # 关闭统计信息减少开销
                           monitor=False,  # 关闭监控减少开销
                           elapsed=False,  # 关闭耗时显示
                           receipt=False) as bar:  # 关闭收据显示

                for batch_idx, (points, labels) in enumerate(self.val_loader):
                    # 异步数据传输
                    points = points.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # 使用混合精度加速推理
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            outputs = self.model(points)
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs = self.model(points)
                        loss = self.criterion(outputs, labels)

                    running_loss += loss.item() * points.size(0)

                    # 计算准确率 - 使用异步操作
                    pred = outputs.argmax(dim=1)
                    mask = labels > 0  # 只考虑车辆(1)和行人(2)

                    if mask.sum() > 0:
                        correct = (pred[mask] == labels[mask]).sum().item()
                        total_correct += correct
                        total_samples += mask.sum().item()

                    total_points += points.size(0)

                    # 实时更新进度条信息
                    current_loss = running_loss / total_points
                    current_acc = total_correct / total_samples if total_samples > 0 else 0

                    # 每10个batch或最后一个batch更新一次显示，减少开销
                    if batch_idx % 10 == 0 or batch_idx == len(self.val_loader) - 1:
                        bar.text = f'损失: {current_loss:.4f} | 准确率: {current_acc:.4f}'

                    bar()  # 更新进度条

        val_loss = running_loss / len(self.val_loader.dataset)
        val_acc = total_correct / total_samples if total_samples > 0 else 0

        print(f"✅ 验证完成: 损失={val_loss:.4f}, 准确率={val_acc:.4f}")
        return val_loss, val_acc

    def print_training_info(self):
        """打印训练信息"""
        print("\n" + "=" * 80)
        print("🏋️‍♂️ 点云语义分割训练开始 - 优化版本")
        print("=" * 80)
        print(f"📊 设备: {self.device}")
        print(f"📈 训练样本: {len(self.train_loader.dataset):,}")
        print(f"📉 验证样本: {len(self.val_loader.dataset):,}")
        print(f"🔢 Batch size: {self.train_loader.batch_size}")
        print(f"📍 每样本点数: {self.train_loader.dataset.num_points}")
        print(f"🔄 总Epoch数: {self.epochs}")

        # 数据加载器信息
        dataset_info = self.train_loader.dataset.get_dataset_info()
        print(f"💾 数据预加载: {dataset_info['preloaded']}")
        print(f"📦 缓存样本: {dataset_info['cached_samples']}")

        # 模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 模型参数: {total_params:,}")

        # GPU信息
        if torch.cuda.is_available():
            gpu_info = self.get_detailed_gpu_info()
            print(f"🎮 GPU内存: {gpu_info.get('allocated_gb', 0):.1f}GB / {gpu_info.get('reserved_gb', 0):.1f}GB")
            print(f"🔥 GPU利用率: {gpu_info.get('utilization', 'N/A')}")

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
    # 大幅增加batch_size来充分利用GPU！
    batch_size = 64  # 从4增加到64，甚至128
    num_points = 4096
    epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 使用设备: {device}")

    # 选择数据加载器
    if GPU_OPTIMIZED_AVAILABLE and torch.cuda.is_available():
        print("🚀 使用GPU优化数据加载器")
        train_loader, val_loader, test_loader = get_gpu_optimized_loaders(
            batch_size=batch_size,
            num_points=num_points
        )
    else:
        print("⚡ 使用普通数据加载器")
        train_loader, val_loader, test_loader = get_data_loaders(
            batch_size=batch_size,
            num_points=num_points,
            preload_to_ram=False  # 如果内存不足
        )

    # 测试数据加载性能
    test_data_loading_performance(train_loader)

    # 使用完整的PointNet++模型
    model = get_complete_model(num_classes=3)

    # 训练器
    trainer = SemanticSegmentationTrainer(model, train_loader, val_loader, device)

    # 开始训练
    trainer.train(epochs=epochs)

    # 绘制训练曲线
    plot_training_curves(trainer.train_losses, trainer.val_losses,
                         trainer.val_accuracies, trainer.learning_rates)


def test_data_loading_performance(train_loader):
    """测试数据加载性能"""
    print("\n🧪 测试数据加载性能...")
    import time

    # 预热
    for i, batch in enumerate(train_loader):
        if i == 2:
            break

    # 正式测试
    start_time = time.time()
    batch_count = 0

    for i, (points, labels) in enumerate(train_loader):
        batch_count += 1
        if i == 10:  # 测试10个batch
            break

    total_time = time.time() - start_time
    avg_batch_time = total_time / batch_count

    print(f"📊 数据加载性能:")
    print(f"   - Batch大小: {train_loader.batch_size}")
    print(f"   - 平均每个batch: {avg_batch_time:.4f}s")
    print(f"   - 预计训练速度: {train_loader.batch_size / avg_batch_time:.1f} 样本/秒")
    print(f"   - GPU利用率预测: {'高' if avg_batch_time < 0.1 else '中' if avg_batch_time < 0.5 else '低'}")
if __name__ == "__main__":
    main()