import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ========================================================
# 1. 导入你组员写好的模型接口
# ========================================================
from mlp import build_mlp
from cnn import build_cnn_32x16  # 如果想测试纯手写CNN，可以换用这个

def train_one_round(model, train_loader, criterion, optimizer, device):
    """用当前的数据集（含伪标签滚动后）训练模型一个 Epoch"""
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, device):
    """在验证集上评估模型，返回准确率 (Accuracy)"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

def run_self_training(x_label, y_label, x_unlabel, x_val, y_val, threshold=0.90, max_rounds=5, model_type="mlp"):
    """
    核心半监督自训练控制中心（C负责的业务逻辑）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 开始执行半监督自训练 | 模型: {model_type.upper()} | 过滤阈值: {threshold} | 最大迭代: {max_rounds}")
    
    # 克隆一份原始数据，防止多轮实验之间的数据污染
    current_x_label = x_label.clone()
    current_y_label = y_label.clone()
    current_x_unlabel = x_unlabel.clone()
    
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)
    
    # 记录每一轮的验证指标，用来完成你的分工表格
    round_records = []

    for round_idx in range(max_rounds):
        print(f"\n--- 🔄 迭代进度 Round {round_idx + 1} / {max_rounds} ---")
        print(f"当前有标签样本规模: {current_x_label.size(0)} 个 | 剩余无标签样本: {current_x_unlabel.size(0)} 个")
        
        # 1. 每次迭代用当前积累的所有标签数据构建 Loader
        train_dataset = TensorDataset(current_x_label, current_y_label)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # 2. 调用组员 B 开发的精细模型
        if model_type == "mlp":
            model = build_mlp(input_dim=512, num_classes=10, hidden_dims=(256, 128, 64), dropout=0.3).to(device)
        elif model_type == "cnn":
            model = build_cnn_32x16(input_dim=512, num_classes=10).to(device)
            
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # 3. 训练当前轮次的模型 (对只有100+数据，20~30个Epoch即可收敛)
        for epoch in range(25):
            loss = train_one_round(model, train_loader, criterion, optimizer, device)
            
        # 4. 在测试/验证集上评估效果
        val_acc = evaluate_model(model, val_loader, device)
        print(f"✨ Round {round_idx + 1} 验证集准确率 (Accuracy): {val_acc:.4f}")
        
        # 最后一轮直接退出，不需要再生成伪标签
        if round_idx == max_rounds - 1:
            round_records.append({'round': round_idx+1, 'added': 0, 'acc': val_acc})
            break
            
        # 5. 核心：给无标签数据打上伪标签 (Pseudo-labeling)
        model.eval()
        current_x_unlabel = current_x_unlabel.to(device)
        with torch.no_grad():
            outputs = model(current_x_unlabel)
            probabilities = torch.softmax(outputs, dim=1)
            max_probs, pseudo_labels = torch.max(probabilities, dim=1)
            
        # 6. 过滤机制：筛选高置信度预测
        mask = max_probs >= threshold
        selected_x = current_x_unlabel[mask].cpu()
        selected_y = pseudo_labels[mask].cpu()
        
        print(f"👉 本轮捞出了 {selected_x.size(0)} 个高置信度无标签样本注入训练集！")
        round_records.append({'round': round_idx+1, 'added': selected_x.size(0), 'acc': val_acc})
        
        if selected_x.size(0) == 0:
            print("没有新样本能跨过置信度门槛，提前结束雪球滚动。")
            break
            
        # 7. 滚雪球：将合格的伪标签样本拼入下一轮的有标签训练集
        current_x_label = torch.cat([current_x_label, selected_x], dim=0)
        current_y_label = torch.cat([current_y_label, selected_y], dim=0)
        
        # 将被选中的样本从无标签池中剔除，避免重复预测
        current_x_unlabel = current_x_unlabel[~mask].cpu()
        
    return round_records

if __name__ == "__main__":
    # ========================================================
    # 模拟对接 A 的数据接口。当 A 把数据读取写好后，
    # 只需要把下面 3 行替换成 A 的数据加载函数即可：
    # x_train_pure, y_train_pure = load_labelled_data()
    # ========================================================
    print("正在准备底层特征数据...")
    x_train_pure = torch.randn(100, 512)
    y_train_pure = torch.randint(0, 10, (100,))
    x_unlabelled_pool = torch.randn(10000, 512)
    x_val_set = torch.randn(200, 512)
    y_val_set = torch.randint(0, 10, (200,))
    
    # ----------------------------------------------------
    # 🧪 实验一：测试不同阈值对半监督的影响 (完成分工表 1)
    # ----------------------------------------------------
    print("\n=== ⚙️ 正在执行实验一：单次伪标签下的阈值搜索 ===")
    thresholds_to_test = [0.80, 0.85, 0.90, 0.95]
    for t in thresholds_to_test:
        # 只迭代 2 轮（即只加一次伪标签看效果）
        run_self_training(x_train_pure, y_train_pure, x_unlabelled_pool, x_val_set, y_val_set, 
                          threshold=t, max_rounds=2, model_type="mlp")
        
    # ----------------------------------------------------
    # 🧪 实验二：选最优阈值进行多轮滚动自训练 (完成分工表 2)
    # ----------------------------------------------------
    print("\n=== ⚙️ 正在执行实验二：最佳阈值下的多轮滚动迭代 ===")
    # 假设实验一中 0.90 效果最好，我们用它滚动迭代 5 轮
    run_self_training(x_train_pure, y_train_pure, x_unlabelled_pool, x_val_set, y_val_set, 
                      threshold=0.90, max_rounds=5, model_type="mlp")