import numpy as np

# 根据序列 chain 计算从某个索引（起始状态）开始到序列最后（终止状态）得到的回报
def compute_return(start_index, chain, gamma):
    G = 0
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G

# 定义状态转移概率矩阵 P
P = [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
     [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
     [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
     [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
P = np.array(P)

# 定义奖励函数
rewards = [-1, -2, -2, 10, 1, 0]

# 定义折扣因子
gamma = 0.5

# 状态序列: s1 -> s2 -> s3 -> s6
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index, chain, gamma)
print(G)

def compute(P, rewards, gamma, states_num):
    # 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数
    rewards = np.array(rewards).reshape((-1, 1))  # 将 rewards 写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value

V = compute(P, rewards, gamma, 6)
print(V)


# 状态集合
S = ["s1", "s2", "s3", "s4", "s5"]  

# 动作集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  

# 状态转移函数
P = {"s1-保持s1-s1": 1.0, "s1-前往s2-s2": 1.0, "s2-前往s1-s1": 1.0, "s2-前往s3-s3": 1.0,
     "s3-前往s4-s4": 1.0, "s3-前往s5-s5": 1.0, "s4-前往s5-s5": 1.0, "s4-概率前往-s2": 0.2,
     "s4-概率前往-s3": 0.4, "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {"s1-保持s1": -1, "s1-前往s2": 0, "s2-前往s1": -1, "s2-前往s3": -2,
     "s3-前往s4": -2, "s3-前往s5": 0, "s4-前往s5": 10, "s4-概率前往": 1,
}
# 折扣因子
gamma = 0.5  
MDP = (S, A, P, R, gamma)

# 策略 1,随机策略
Pi_1 = {"s1-保持s1": 0.5, "s1-前往s2": 0.5, "s2-前往s1": 0.5, "s2-前往s3": 0.5,
        "s3-前往s4": 0.5, "s3-前往s5": 0.5, "s4-前往s5": 0.5, "s4-概率前往": 0.5,
}

# 策略2
Pi_2 = {"s1-保持s1": 0.6, "s1-前往s2": 0.4, "s2-前往s1": 0.3, "s2-前往s3": 0.7,
        "s3-前往s4": 0.5, "s3-前往s5": 0.5, "s4-前往s5": 0.1, "s4-概率前往": 0.9,
}

# 把输入的两个字符串通过 - 连接,便于使用 P、R 变量
def join(str1, str2):
    return str1 + '-' + str2

gamma = 0.5

# 转化后的MRP的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
print(V)

def sample(MDP, Pi, timestep_max, number):
    # 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        
        # 随机选择一个除s5以外的状态s作为起点
        s = S[np.random.randint(4)]  
        
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            
            # 把（s,a,r,s_next）元组放入序列中
            episode.append((s, a, r, s_next))  
            
            # s_next变成当前状态,开始接下来的循环
            s = s_next  
        episodes.append(episode)
    return episodes


# 采样5次,每个序列最长不超过20步
episodes = sample(MDP, Pi_1, 20, 5)
print(episodes[0])
print(episodes[1])
print(episodes[4])

## 输出
# [('s1', '前往s2', 0, 's2'), ('s2', '前往s3', -2, 's3'), ('s3', '前往s5', 0, 's5')]
# [('s4', '概率前往', 1, 's4'), ('s4', '前往s5', 10, 's5')]
# [('s2', '前往s3', -2, 's3'), ('s3', '前往s4', -2, 's4'), ('s4', '前往s5', 10, 's5')]

def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        
        # 一个序列从后往前计算
        for i in range(len(episode) - 1, -1, -1):  
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]


timestep_max = 20

# 使用蒙特卡洛方法计算MDP的状态价值，采样 1000 次,可以自行修改
episodes = sample(MDP, Pi_1, timestep_max, 1000)
gamma = 0.5
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
MC(episodes, V, N, gamma)
print(V)

## 输出
# {'s1': -1.228923788722258, 's2': -1.6955696284402704, 's3': 0.4823809701532294, 's4': 5.967514743019431, 's5': 0}

def occupancy(episodes, s, a, timestep_max, gamma):
    # 计算状态动作对（s,a）出现的频率,以此来估算策略的占用度量
    rho = 0
    total_times = np.zeros(timestep_max)  # 记录每个时间步t各被经历过几次
    occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestep_max)):
        if total_times[i]:
            rho += gamma**i * occur_times[i] / total_times[i]
    return (1 - gamma) * rho


gamma = 0.5
timestep_max = 1000

episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
print(rho_1, rho_2)

## 输出
# 0.112567796310472 0.23199480615618912