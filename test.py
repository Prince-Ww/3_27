import torch
print(torch.__version__); 
print(torch.tensor([1,2,3])); 
print(torch.tensor([1,2,3]).device)


# # 1. 地图构造
# def create_custom_map():
#     ...

# # 2. 环境构造
# def make_env(...):
#     ...

# # 3. epsilon-greedy
# def epsilon_greedy_action(...):
#     ...

# # 4. 训练
# def train_q_learning(...):
#     ...

# # 5. 测试
# def evaluate_policy(...):
#     ...

# # 6. 提取最终策略
# def extract_greedy_policy(...):
#     ...

# # 7. 打印策略
# def print_policy(...):
#     ...

# # 8. 画图
# def plot_results(...):
#     ...

# # 9. 主函数：分别跑 8x8 和 12x12
# def main():
#     # 跑 8x8
#     # 训练、测试、画图、输出策略
    
#     # 跑 12x12
#     # 训练、测试、画图、输出策略
    
#     # 对比结果
#     ...

# if __name__ == "__main__":
#     main()