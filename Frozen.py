import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(BASE_DIR, "run")


# =========================================================
# 1. 环境构建
# =========================================================
def build_frozenlake_env(map_name=None, custom_map=None, render_mode=None):
    
    if map_name is not None and custom_map is not None:
        raise ValueError("map_name 和 custom_map 不能同时传入，请二选一。")


#test  
#test2
    if custom_map is not None:
        env = gym.make(
            "FrozenLake-v1",
            desc=custom_map,
            is_slippery=True,
            render_mode=render_mode,
            max_episode_steps=1000
        )
    elif map_name is not None:
        env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=True,
            render_mode=render_mode,
            max_episode_steps=1000
        )
    else:
        raise ValueError("必须传入 map_name 或 custom_map 其中之一。")

    return env


def get_custom_map_12x12():
    """
    返回一个 12x12 自定义地图
    S: 起点
    F: 安全冰面
    H: 洞
    G: 终点
    """
    custom_map = [
        "SFFFFFFFFFFF",
        "FFFHFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFHFFFFFFF",
        "FFFFFFFFHFFF",
        "FFFFFFFFFFFF",
        "FHHFFFFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFHFFFFFF",
        "FFFFFFFFFFFF",
        "FFFFFFFHFFFF",
        "FFFFFFFFFFFG"
    ]
    return custom_map


# =========================================================
# 2. epsilon-greedy 动作选择
# =========================================================
def epsilon_greedy_action(Q, state, epsilon, n_actions):
    """
    epsilon-greedy 动作选择
    """
    if np.random.rand() < epsilon:
        action = np.random.randint(n_actions)
    else:
        action = np.argmax(Q[state])

    return action


# =========================================================
# 3. 一些工具函数
# =========================================================
def moving_average(data, window=100):
    """
    简单滑动平均，便于画曲线
    """
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def extract_greedy_policy(Q, map_desc):
    """
    从 Q 表中提取贪婪策略
    对于 F/S 状态，显示箭头
    对于 H/G，保留原地图字符
    """
    action_symbols = {
        0: "←",   # 左
        1: "↓",   # 下
        2: "→",   # 右
        3: "↑"    # 上
    }

    n_rows = len(map_desc)
    n_cols = len(map_desc[0])

    policy_grid = []
    for r in range(n_rows):
        row_policy = []
        for c in range(n_cols):
            s = r * n_cols + c
            cell = map_desc[r][c]

            if cell == "H":
                row_policy.append("H")
            elif cell == "G":
                row_policy.append("G")
            elif cell == "S":
                # 起点也给箭头，便于看最终策略
                best_action = np.argmax(Q[s])
                row_policy.append(action_symbols[best_action])
            else:  # F
                best_action = np.argmax(Q[s])
                row_policy.append(action_symbols[best_action])

        policy_grid.append(row_policy)

    return policy_grid


def print_policy(policy_grid, title="Greedy Policy"):
    """
    打印最终贪婪策略
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for row in policy_grid:
        print(" ".join(row))
    print("=" * 60 + "\n")


def get_map_desc_from_env(env, custom_map=None, map_name=None):
    if custom_map is not None:
        return custom_map

    elif map_name == "4x4":
        # 官方 4x4 标准地图
        return [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG",
        ]

    elif map_name == "8x8":
        # 官方 8x8 标准地图
        return [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]

    else:
        raise ValueError("目前该函数仅支持 map_name='4x4'、'8x8' 或 custom_map。")


# =========================================================
# 4. Q-learning 训练
# =========================================================
def train_q_learning(
    env,
    episodes,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.02,
    epsilon_decay=0.999,
    max_steps_per_episode=300,
):
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Q表初始化
    Q = np.zeros((n_states, n_actions))

    rewards_per_episode = []
    success_per_episode = []
    steps_per_episode = []
    epsilon_history = []

    for episode in range(episodes):
        state, info = env.reset()
        done = False

        total_reward = 0
        step_count = 0
        success = 0

        for step in range(max_steps_per_episode):
            step_count += 1

            # epsilon-greedy 选动作
            action = epsilon_greedy_action(Q, state, epsilon, n_actions)

            # 与环境交互
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 奖励重塑
            if done and reward == 0:   #掉坑里了
                shaped_reward = -10

            elif done and reward == 1:  #到终点了
                shaped_reward = 10
            else:
                shaped_reward = -0.01

            # Q-learning 核心更新公式
            best_next_q = np.max(Q[next_state])
            td_target = shaped_reward if done else shaped_reward + gamma * best_next_q
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            total_reward += shaped_reward
            state = next_state

            if done:
                if reward > 0:   # 注意这里用原始 reward 判断是否成功
                    success = 1
                break

        # 一回合结束，记录指标
        rewards_per_episode.append(total_reward)
        success_per_episode.append(success)
        steps_per_episode.append(step_count)

        # 探索率衰减
        if episode % 10 == 0 :
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            epsilon_history.append(epsilon)

        # 打印进度（每隔一定回合）
        if (episode + 1) % max(1, episodes // 10) == 0:
            recent_reward = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
            recent_success = np.mean(success_per_episode[-100:]) if len(success_per_episode) >= 100 else np.mean(success_per_episode)
            recent_steps = np.mean(steps_per_episode[-100:]) if len(steps_per_episode) >= 100 else np.mean(steps_per_episode)

            print(
                f"[Episode {episode + 1:>6}/{episodes}] "
                f"epsilon={epsilon:.4f} | "
                f"avg_reward(最近100)={recent_reward:.4f} | "
                f"success_rate(最近100)={recent_success:.4f} | "
                f"avg_steps(最近100)={recent_steps:.2f}"
            )

    return Q, rewards_per_episode, success_per_episode, steps_per_episode, epsilon_history


# =========================================================
# 5. 评估最终贪婪策略
# =========================================================
def evaluate_greedy_policy(env, Q, eval_episodes=1000, max_steps_per_episode=300):
    """
    用纯贪婪策略（不探索）评估训练结果
    """
    rewards = []
    successes = []
    steps_list = []

    for _ in range(eval_episodes):
        state, info = env.reset()
        total_reward = 0
        success = 0
        step_count = 0

        for step in range(max_steps_per_episode):
            step_count += 1

            action = np.argmax(Q[state])  # 纯贪婪
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

            if done:
                if reward > 0:
                    success = 1
                break

        rewards.append(total_reward)
        successes.append(success)
        steps_list.append(step_count)

    result = {
        "avg_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "avg_steps": float(np.mean(steps_list))
    }
    return result


# =========================================================
# 6. 可视化
# =========================================================
def plot_training_curves(
    rewards_8, success_8, steps_8,
    rewards_12, success_12, steps_12,
    ma_window=200,
    save_dir=RUN_DIR,
    show_plot=False
):
    """
    绘制并保存训练曲线

    参数:
        rewards_8, success_8, steps_8: 8x8环境的训练记录
        rewards_12, success_12, steps_12: 12x12环境的训练记录
        ma_window: 滑动平均窗口大小
        save_dir: 图片保存目录
        show_plot: 是否显示图片窗口
    """
    # 如果目录不存在，就创建
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 1. 回报曲线
    # =========================
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(rewards_8, ma_window), label="8x8 reward (moving avg)")
    plt.plot(moving_average(rewards_12, ma_window), label="12x12 reward (moving avg)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    reward_path = os.path.join(save_dir, "training_reward_curve.png")
    plt.savefig(reward_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # =========================
    # 2. 成功率曲线
    # =========================
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(success_8, ma_window), label="8x8 success rate (moving avg)")
    plt.plot(moving_average(success_12, ma_window), label="12x12 success rate (moving avg)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Training Success Rate Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    success_path = os.path.join(save_dir, "training_success_curve.png")
    plt.savefig(success_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    # =========================
    # 3. 平均步数曲线
    # =========================
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(steps_8, ma_window), label="8x8 steps (moving avg)")
    plt.plot(moving_average(steps_12, ma_window), label="12x12 steps (moving avg)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Training Steps Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    steps_path = os.path.join(save_dir, "training_steps_curve.png")
    plt.savefig(steps_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print("图片已保存到：")
    print(reward_path)
    print(success_path)
    print(steps_path)


def save_experiment_outputs(name, result, save_dir=RUN_DIR):
    os.makedirs(save_dir, exist_ok=True)
    file_prefix = name.lower().replace(" ", "_").replace("/", "_")

    q_path = os.path.join(save_dir, f"{file_prefix}_q_table.npy")
    history_path = os.path.join(save_dir, f"{file_prefix}_training_history.csv")
    policy_path = os.path.join(save_dir, f"{file_prefix}_policy.txt")
    summary_path = os.path.join(save_dir, f"{file_prefix}_summary.txt")

    np.save(q_path, result["Q"])

    with open(history_path, "w", encoding="utf-8") as f:
        f.write("episode,shaped_reward,success,steps\n")
        for episode, (reward, success, steps) in enumerate(
            zip(result["rewards"], result["success"], result["steps"]),
            start=1
        ):
            f.write(f"{episode},{reward:.8f},{int(success)},{int(steps)}\n")

    with open(policy_path, "w", encoding="utf-8") as f:
        for row in result["policy"]:
            f.write(" ".join(row) + "\n")

    eval_result = result["eval_result"]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"name: {name}\n")
        f.write(f"avg_reward: {eval_result['avg_reward']:.6f}\n")
        f.write(f"success_rate: {eval_result['success_rate']:.6f}\n")
        f.write(f"avg_steps: {eval_result['avg_steps']:.6f}\n")
        f.write(f"q_table: {q_path}\n")
        f.write(f"history: {history_path}\n")
        f.write(f"policy: {policy_path}\n")

    print(f"{name} results saved to: {save_dir}")
    return {
        "q_table": q_path,
        "history": history_path,
        "policy": policy_path,
        "summary": summary_path,
    }


def export_greedy_policy_gif(
    name,
    Q,
    output_path,
    map_name=None,
    custom_map=None,
    max_steps_per_episode=300,
    fps=3
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    env_gif = build_frozenlake_env(
        map_name=map_name,
        custom_map=custom_map,
        render_mode="rgb_array"
    )

    frames = []
    state, info = env_gif.reset()
    frames.append(env_gif.render())

    total_reward = 0
    success = 0
    steps = 0

    for step in range(max_steps_per_episode):
        action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, info = env_gif.step(action)

        frames.append(env_gif.render())
        total_reward += reward
        steps = step + 1
        state = next_state

        if terminated or truncated:
            success = int(reward > 0)
            break

    env_gif.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    image = ax.imshow(frames[0])
    ax.set_title(
        f"{name} final test | reward={total_reward:.0f}, "
        f"success={success}, steps={steps}"
    )

    def update(frame_index):
        image.set_data(frames[frame_index])
        return [image]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=True
    )
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    print(f"{name} GIF saved to: {output_path}")
    return {
        "path": output_path,
        "reward": total_reward,
        "success": success,
        "steps": steps,
    }


# =========================================================
# 7. 单个环境实验封装
# =========================================================
def run_one_experiment(
    name,
    env,
    map_desc,
    episodes,
    alpha,
    gamma,
    epsilon,
    epsilon_min,
    epsilon_decay,
    max_steps_per_episode,
    eval_episodes=1000
):
    """
    跑一个环境（8x8 或 12x12）的完整实验
    """
    print(f"\n{'#' * 80}")
    print(f"开始训练环境: {name}")
    print(f"{'#' * 80}")

    Q, rewards, success, steps, epsilon_history = train_q_learning(
        env=env,
        episodes=episodes,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        max_steps_per_episode=max_steps_per_episode,
    )

    # 评估最终贪婪策略
    eval_result = evaluate_greedy_policy(
        env=env,
        Q=Q,
        eval_episodes=eval_episodes,
        max_steps_per_episode=max_steps_per_episode
    )

    # 输出策略
    greedy_policy = extract_greedy_policy(Q, map_desc)
    print_policy(greedy_policy, title=f"{name} 最终贪婪策略")

    print(f"{name} 最终评估结果：")
    print(f"平均回报 avg_reward    = {eval_result['avg_reward']:.4f}")
    print(f"成功率   success_rate  = {eval_result['success_rate']:.4f}")
    print(f"平均步数 avg_steps     = {eval_result['avg_steps']:.2f}")

    return {
        "Q": Q,
        "rewards": rewards,
        "success": success,
        "steps": steps,
        "epsilon_history": epsilon_history,
        "eval_result": eval_result,
        "policy": greedy_policy,
    }








def play_one_episode_with_human_render(
    Q,
    map_name=None,
    custom_map=None,
    max_steps_per_episode=300,
    sleep_time=0.4
):
    """
    用训练好的 Q 表，在 human 模式下播放 1 个 episode

    参数:
        Q: 训练好的 Q 表
        map_name: 官方地图名，例如 "8x8"
        custom_map: 自定义地图（如果不是官方地图就传这个）
        max_steps_per_episode: 单回合最大步数
        sleep_time: 每一步之间暂停时间，单位秒，方便看动画
    """
    # 用 human 模式新建一个环境，专门用于演示
    env_demo = build_frozenlake_env(
        map_name=map_name,
        custom_map=custom_map,
        render_mode="human"
    )

    state, info = env_demo.reset()

    print("\n开始播放 1 个贪婪策略 episode ...")
    print(f"初始状态: {state}")

    total_reward = 0

    for step in range(max_steps_per_episode):
        # 纯贪婪动作：选择当前 Q 值最大的动作
        # 为避免并列最大值时永远偏向第一个动作，这里随机打破平局
        q_values = Q[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        action = np.random.choice(best_actions)

        next_state, reward, terminated, truncated, info = env_demo.step(action)
        done = terminated or truncated

        total_reward += reward

        print(
            f"step={step+1:03d}, "
            f"state={state:02d}, "
            f"action={action}, "
            f"next_state={next_state:02d}, "
            f"reward={reward}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )

        state = next_state

        # 放慢一点，方便看窗口动画
        time.sleep(sleep_time)

        if done:
            if reward > 0:
                print(f"\n本回合成功到达终点！总奖励 = {total_reward}")
            else:
                print(f"\n本回合结束，但没有到达终点。总奖励 = {total_reward}")
            break
    else:
        print(f"\n达到最大步数 {max_steps_per_episode}，本回合结束。总奖励 = {total_reward}")

    # 为了防止窗口一闪而过，结束后再停 2 秒
    time.sleep(2)
    env_demo.close()


# =========================================================
# 8. 主函数
# =========================================================
def main():
    # -----------------------------
    # 参数设置（按实验文档建议范围）
    # -----------------------------
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.999

    # 文档建议：
    # 8x8 至少 5000 回合
    # 12x12 至少 10000 回合
    episodes_8x8 = 50000
    episodes_12x12 = 50000

    max_steps_8x8 = 1000
    max_steps_12x12 = 1000
    eval_episodes = 1000
    gif_max_steps = 300

    os.makedirs(RUN_DIR, exist_ok=True)
    print(f"All results will be saved to: {RUN_DIR}")

    # -----------------------------
    # 构建 8x8 环境
    # -----------------------------
    env_8 = build_frozenlake_env(map_name="8x8", render_mode=None)
    map_desc_8 = get_map_desc_from_env(env_8, map_name="8x8")

    # -----------------------------
    # 构建 12x12 自定义环境
    # -----------------------------
    custom_map_12 = get_custom_map_12x12()
    env_12 = build_frozenlake_env(custom_map=custom_map_12, render_mode=None)
    map_desc_12 = custom_map_12

    # -----------------------------
    # 分别训练
    # -----------------------------
    result_8 = run_one_experiment(
        name="FrozenLake 8x8",
        env=env_8,
        map_desc=map_desc_8,
        episodes=episodes_8x8,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        max_steps_per_episode=max_steps_8x8,
        eval_episodes=eval_episodes
    )

    result_12 = run_one_experiment(
        name="FrozenLake 12x12",
        env=env_12,
        map_desc=map_desc_12,
        episodes=episodes_12x12,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        max_steps_per_episode=max_steps_12x12,
        eval_episodes=eval_episodes
    )

    # -----------------------------
    # 对比打印
    # -----------------------------
    print("\n" + "=" * 80)
    print("最终对比结果")
    print("=" * 80)
    print("8x8  : ",
          f"avg_reward={result_8['eval_result']['avg_reward']:.4f}, ",
          f"success_rate={result_8['eval_result']['success_rate']:.4f}, ",
          f"avg_steps={result_8['eval_result']['avg_steps']:.2f}")
    print("12x12: ",
          f"avg_reward={result_12['eval_result']['avg_reward']:.4f}, ",
          f"success_rate={result_12['eval_result']['success_rate']:.4f}, ",
          f"avg_steps={result_12['eval_result']['avg_steps']:.2f}")
    print("=" * 80)

    save_experiment_outputs("FrozenLake 8x8", result_8, save_dir=RUN_DIR)
    save_experiment_outputs("FrozenLake 12x12", result_12, save_dir=RUN_DIR)

    # -----------------------------
    # 画图
    # -----------------------------
    plot_training_curves(
        rewards_8=result_8["rewards"],
        success_8=result_8["success"],
        steps_8=result_8["steps"],
        rewards_12=result_12["rewards"],
        success_12=result_12["success"],
        steps_12=result_12["steps"],
        ma_window=200,
        save_dir=RUN_DIR,
        show_plot=False
    )

    export_greedy_policy_gif(
        name="FrozenLake 8x8",
        Q=result_8["Q"],
        output_path=os.path.join(RUN_DIR, "final_test_8x8.gif"),
        map_name="8x8",
        custom_map=None,
        max_steps_per_episode=gif_max_steps
    )

    export_greedy_policy_gif(
        name="FrozenLake 12x12",
        Q=result_12["Q"],
        output_path=os.path.join(RUN_DIR, "final_test_12x12.gif"),
        map_name=None,
        custom_map=custom_map_12,
        max_steps_per_episode=gif_max_steps
    )

    # 关闭环境
    env_8.close()
    env_12.close()


if __name__ == "__main__":
    main()
