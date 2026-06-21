import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import argparse
import os
import seaborn as sns

# ================= 设置画图风格 =================
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
    sns.set_palette("tab10")
except ImportError:
    sns.set_style("whitegrid")
    sns.set_palette("tab10")

plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})


def load_env_core(path):
    """从 .pkl 文件中加载‘冻结’的环境对象"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")
    with open(path, 'rb') as f:
        env_core = pickle.load(f)
    return env_core


def plot_uav_trajectory(ax, env, title_str="UAV Trajectory"):
    """
    绘制轨迹的函数 (已完美兼容单智能体 env.uav 和 多智能体 env.UAVs)
    """
    # ---------------- 1. 绘制地面用户 (Blue Dots) ----------------
    users_x = [u.position[0] for u in env.Users]
    users_y = [u.position[1] for u in env.Users]
    ax.scatter(users_x, users_y, c='#1f77b4', marker='o', s=15, label='Users', zorder=2, alpha=0.7)

    # ---------------- 2. 绘制禁飞区 NFZ (Red Circle) ----------------
    nfz_circle = patches.Circle((env.nfz_center[0], env.nfz_center[1]), 
                                env.nfz_radius,
                                edgecolor='#d62728', 
                                facecolor='#d62728', 
                                alpha=0.25, 
                                label='NFZ', 
                                zorder=1)
    ax.add_patch(nfz_circle)

    # ---------------- 3. 提取无人机列表 (核心兼容逻辑) ----------------
    uav_list = []
    if hasattr(env, 'UAVs'):
        uav_list = env.UAVs         # 匹配未来的多智能体环境
    elif hasattr(env, 'uav'):
        uav_list = [env.uav]        # 匹配当前的单智能体环境，转为列表统一处理
    
    if not uav_list:
        print("⚠️ 警告: 环境快照中未找到任何无人机数据 (uav 或 UAVs)。")
        return

    # ---------------- 4. 循环绘制无人机轨迹 ----------------
    # 预设一个颜色列表，供多无人机使用
    colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, uav_obj in enumerate(uav_list):
        if len(uav_obj.trajectory) > 0:
            traj = np.array(uav_obj.trajectory)
            
            # 如果只有一个无人机，就叫 UAV，否则叫 UAV 1, UAV 2...
            label_name = 'UAV' if len(uav_list) == 1 else f'UAV {i+1}'
            # 从颜色列表中循环取色
            line_color = colors[i % len(colors)]
            
            # 绘制飞行路线
            ax.plot(traj[:, 0], traj[:, 1], c=line_color, linestyle='-', linewidth=2, label=label_name, zorder=3)
            
            # 标出起点和终点 (为了防止图例重复，只在画第一架无人机时添加 Start/End 图例)
            start_label = 'Start' if i == 0 else ""
            end_label = 'End' if i == 0 else ""
            
            ax.scatter(traj[0, 0], traj[0, 1], c='#ff7f0e', marker='*', s=180, zorder=4, edgecolor='black', label=start_label)
            ax.scatter(traj[-1, 0], traj[-1, 1], c='#9467bd', marker='X', s=120, zorder=4, edgecolor='black', label=end_label)

    # ---------------- 5. 设置画布属性 ----------------
    ax.set_xlim(env.min_x, env.max_x)
    ax.set_ylim(env.min_y, env.max_y)
    ax.set_xlabel('X-coordinate (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y-coordinate (m)', fontsize=13, fontweight='bold')
    ax.set_title(title_str, fontsize=15, fontweight='bold')
    
    # 优化图例：去除重复项
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # 将图例放在外面，避免遮挡轨迹 (可以根据需要把 bbox_to_anchor 删掉让它自动找位置)
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)
    
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_aspect('equal', adjustable='box')


def main():
    parser = argparse.ArgumentParser(description="PPO UAV Trajectory Plotter")
    
    parser.add_argument("--path", type=str, 
                        default="./results/PPO/best_env_XXXX.pkl", # 填入你实际跑出来的pkl名字
                        help="Path to the env .pkl file")
    
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the output figure. Default: same as path but .png")
    
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = args.path.replace(".pkl", ".png")
    args.save_path = os.path.abspath(args.save_path)

    try:
        print(f"正在加载环境快照: {args.path}...")
        env_core = load_env_core(args.path)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        filename = os.path.basename(args.path)
        title_date = filename.split('_')[-1].split('.')[0] 
        title = f"UAV Mission Trajectory - {title_date.capitalize()}"
        
        # 调用画图函数
        plot_uav_trajectory(ax, env_core, title_str=title)
        
        plt.tight_layout()
        plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
        print(f"Saved absolute path: {args.save_path}")
        print(f"✅ 轨迹图已成功保存至: {args.save_path}")

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    main()
