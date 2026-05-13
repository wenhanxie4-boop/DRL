import numpy as np
import os


def generate_users(num_users=50, length=500, width=500,
                   filename=r"D:\workspace\DRL_code_pytorch\DRL-code-pytorch-main\ENV\users_50_v2.txt"):
    """
    生成 50 个用户的坐标和初始数据量，并确保不在禁飞区内，保存到指定 Windows 目录下
    """
    # 自动检查并创建目标文件夹（如果 D:\workspace\MARL_Satellite 不存在，会自动创建）
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 定义禁飞区 (与环境中的设置保持一致)
    nfz_center = np.array([length / 2, width / 2])
    nfz_radius = 50.0

    with open(filename, 'w') as f:
        count = 0
        while count < num_users:
            x = np.random.uniform(0, length)
            y = np.random.uniform(0, width)

            # 排查：计算当前生成坐标到禁飞区中心的距离
            dist_to_nfz = np.linalg.norm(np.array([x, y]) - nfz_center)

            # 如果落在禁飞区内，跳过本次循环，重新生成
            if dist_to_nfz <= nfz_radius:
                continue

            # 初始数据量 (单位: bits，代表 3.75KB 到 6.25KB)
            data = np.random.uniform(10000, 50000)

            f.write(f"{x:.2f} {y:.2f} {data:.2f}\n")
            count += 1

    print(f"已成功生成包含 {num_users} 个用户的新文件！")
    print(f"文件保存位置：{filename}")
    print(f"安全检查：所有用户均已避开中心 {nfz_center}，半径 {nfz_radius}m 的禁飞区。")


if __name__ == "__main__":
    generate_users()