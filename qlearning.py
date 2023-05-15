import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class EpsilonGreedyPolicy:
    def __init__(self, initial_epsilon, min_epsilon, decay_rate):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.ACTION = [[0,1,3], [0,1,3], [2,3], [3,4], [3,4], [1,2]]
    
    def choose_action(self, qTable, current_state):
        if random.random() < self.epsilon: # exploration
            if current_state < 2:
                randNum = random.randint(0, 2)
            else:
                randNum = random.randint(0, 1)
            choose_action = self.ACTION[current_state][randNum]
            return choose_action
        else: # exploitation
            max_index = self.ACTION[current_state][0]
            temp = 0
            max_Qvalue = qTable[current_state][self.ACTION[current_state][0]]

            if current_state in [0, 1]:
                for i in range(1, 3):
                    temp = self.ACTION[current_state][i]
                    if qTable[current_state][temp] > max_Qvalue:
                        max_Qvalue = qTable[current_state][temp]
                        max_index = temp
            else:
                for i in range(1, 2):
                    temp = self.ACTION[current_state][i]
                    if qTable[current_state][temp] > max_Qvalue:
                        max_Qvalue = qTable[current_state][temp]
                        max_index = temp
            return max_index

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay_rate, self.min_epsilon) #exponential
        episode_epsilons.append(self.epsilon)

NUM_TERMINAL = 3
NUM_STATES = 9 # 6 + terminal state 3개
NUM_ACTION = 5
ACTION = [[0, 1, 3], [0, 1, 3], [2, 3], [3, 4], [3, 4], [1, 2]]
episode_epsilons = []
avg_maxQvalue = []
print_maxQvalue = [[], [], [], [], [], []]

# 가우시안 분포로 reward 생성하는 함수
def generateGaussianReward(mean, stddev):
    # 랜덤 엔진 초기화
    generator = np.random.default_rng()
    distribution = generator.normal(mean, stddev)
    # 가우시안 분포를 따르는 랜덤 값을 생성하고 반환
    reward = distribution
    return reward

# qtable 초기화
qTable = np.zeros((NUM_STATES, NUM_ACTION))

print("Initial Q-table")
for i in range(NUM_STATES - NUM_TERMINAL):
    print(f"State {i+1}: {qTable[i]}")

# 학습 파라미터 설정
alpha = 0.01 # 학습률
gamma = 0.95 # 할인인자
numEpisodes = 10000 # 에피소드 횟수 

kInitialEpsilon = 0.8 # 초기 입실론 값
kMinEpsilon = 0.05 # 입실론 값의 최소값
kDecayRate = 0.9995 # 입실론 값의 감소 비율(= 기존 앱실론 값의 99.9%로 감소)

policy = EpsilonGreedyPolicy(kInitialEpsilon, kMinEpsilon, kDecayRate) # 입실론 그리디 정책 객체 생성

print(f"\nalpha : {alpha}")
print(f"gamma : {gamma}")
print(f"num of episode : {numEpisodes}")
print(f"initial epsilon : {kInitialEpsilon}")
print(f"min epsilon : {kMinEpsilon}")
print(f"decay rate : {kDecayRate}\n")


for i in range(numEpisodes):

    # 난수 설정
    random_number = random.randint(10000, 99999)
    random.seed(random_number)

    # 초기 상태 선택
    currentState = random.randint(0, NUM_STATES - NUM_TERMINAL - 1)
    
    # print("first state : ", currentState + 1)
    #print("epsilon : ", policy.epsilon)

    # 에피소드 실행 반복문. 터미널 state 만나면 종료
    while currentState < (NUM_STATES - NUM_TERMINAL):
        
        # 행동 선택 by 입실론 그리디
        chosen_action = policy.choose_action(qTable, currentState)
        
        #print("chosen action : ", chosen_action + 1)

        # 보상과 다음 상태 결정
        reward = 0.0
        nextState = 0

        if currentState == 0:  # MDP에서는 S1
            if chosen_action == 0:  # a1
                randValue = random.random()
                if randValue < 0.7:
                    reward = generateGaussianReward(-40.0, 1.0)
                    nextState = 1  # s2
                else:
                    reward = generateGaussianReward(-20.0, 0.5)
                    nextState = 2  # s3
            elif chosen_action == 1:  # a2
                reward = generateGaussianReward(20.0, 1.0)
                nextState = 2  # s3
            elif chosen_action == 3:  # a4
                randValue = random.random()
                if randValue < 0.8:
                    reward = generateGaussianReward(-30.0, 8.0)
                    nextState = 1  # s2
                else:
                    reward = generateGaussianReward(-150.0, 3.0)
                    nextState = 8  # t3
        elif currentState == 1:  # s2
            if chosen_action == 0:  # a1
                randValue = random.random()
                if randValue < 0.8:
                    reward = generateGaussianReward(20.0, 1.0)
                    nextState = 6  # t1
                else:
                    reward = generateGaussianReward(-80.0, 2.0)
                    nextState = 7  # t2
            elif chosen_action == 1:  # a2
                reward = generateGaussianReward(-90.0, 1.0)
                nextState = 7  # t2
            elif chosen_action == 3:  # a4
                reward = generateGaussianReward(30.0, 3.0)
                nextState = 6  # t1
        elif currentState == 2:  # s3
            if chosen_action == 2:  # a3
                reward = generateGaussianReward(20.0, 2.0)
                nextState = 6  # t1
            elif chosen_action == 3:  # a4
                reward = generateGaussianReward(-80.0, 0.5)
                nextState = 0  # s1
        elif currentState == 3:  # s4
            if chosen_action == 3:  # a4
                randValue = random.random()
                if randValue < 0.9:
                    reward = generateGaussianReward(0.0, 5.0)
                    nextState = 4  # s5
                else:
                    reward = generateGaussianReward(-100.0, 1.0)
                    nextState = 5  # s6
            elif chosen_action == 4:  # a5
                randValue = random.random()
                if randValue < 0.6:
                    reward = generateGaussianReward(-10.0, 4.0)
                    nextState = 4  # s5
                elif 0.6 <= randValue < 0.9:
                    reward = generateGaussianReward(-50.0, 1.0)
                    nextState = 3  # s4
                else:
                    reward = generateGaussianReward(-80.0, 3.0)
                    nextState = 7  # t2
        elif currentState == 4: # s5
            if chosen_action == 3: # a4
                randValue = random.random()
                if randValue < 0.7:
                    reward = generateGaussianReward(-70.0, 2.0)
                    nextState = 7 # t2
                else:
                    reward = generateGaussianReward(25.0, 2.0)
                    nextState = 6 # t1
            elif chosen_action == 4: # a5
                randValue = random.random()
                if randValue < 0.8:
                    reward = generateGaussianReward(-10.0, 0.8)
                    nextState = 2 # s3
                else:
                    reward = generateGaussianReward(-40.0, 0.5)
                    nextState = 4 # s5
        elif currentState == 5: # s6
            if chosen_action == 1: # a2
                randValue = random.random()
                if randValue < 0.75:
                    reward = generateGaussianReward(-10.0, 3.0)
                    nextState = 4 # s5
                else:
                    reward = generateGaussianReward(0.0, 1.0)
                    nextState = 2 # s3
            elif chosen_action == 2: # a3
                reward = generateGaussianReward(-20.0, 4.0)
                nextState = 1 # s2
        #print("immediate reward : ", reward)

        #q-value 함수 업데이트
        maxNextQValue = 0.0
        if 0 <= nextState <= 5:
            if 2 <= nextState <= 5:
                temp_j = 2
            elif 0 <= nextState <= 1:
                temp_j = 3

            maxNextQValue = qTable[nextState][ACTION[nextState][0]]
            for j in range(1, temp_j):
                if qTable[nextState][ACTION[nextState][j]] > maxNextQValue:
                    maxNextQValue = qTable[nextState][ACTION[nextState][j]]
        qTable[currentState][chosen_action] = qTable[currentState][chosen_action] + alpha * (reward + gamma * maxNextQValue - qTable[currentState][chosen_action])
        
        #print("Q-value updated : ", qTable[currentState][chosen_action])

        #다음 상태로 이동
        currentState = nextState
        
        #print("next state : ", currentState+1)
    
    # 입실론 값 갱신
    policy.update_epsilon() 

    # for i in range(NUM_STATES - NUM_TERMINAL):
    #     print("State ", i + 1, ": ", end="")
    #     for j in range(NUM_ACTION):
    #         print(qTable[i][j], " | ", end="")
    #     print()
    # print("\n")
    
    # maxQ(S,a) 값 갱신
    maxQvalue = 0.0
    avg = 0.0
    for k in range(0, NUM_STATES-NUM_TERMINAL):
        if 2 <= k <= 5:
            temp_l = 2
        elif 0 <= k <= 1:
            temp_l = 3
        maxQvalue = qTable[k][ACTION[k][0]]
        for l in range(1, temp_l):
            if qTable[k][ACTION[k][l]] > maxQvalue:
                maxQvalue = qTable[k][ACTION[k][l]]
        avg = avg + maxQvalue
        print_maxQvalue[k].append(maxQvalue)
    avg = avg / (NUM_STATES-NUM_TERMINAL)
    avg_maxQvalue.append(avg)

#학습 결과 출력
print("Final Q-table")
for i in range(NUM_STATES - NUM_TERMINAL):
    print("State ", i + 1, ": ", end="")
    for j in range(NUM_ACTION):
        print(qTable[i][j], " | ", end="")
    print()
print("final epsilon : ", policy.epsilon)

#epsilon plot
# plt.plot(list(range(numEpisodes)), episode_epsilons)
# plt.xlabel("#Episode")
# plt.ylabel("Epsilon")
# plt.title("Epsilon decay over episodes")
# plt.show()

# #avg plot
# plt.plot(list(range(numEpisodes)), avg_maxQvalue)
# plt.xlabel("#Episode")
# plt.ylabel("Average of maxQ(S,a)")
# plt.title("Average of maxQ(S,a) growth over episodes")
# plt.show()

# plot 한 번에 하기
# Figure 객체 생성과 서브플롯 생성
fig, ax1 = plt.subplots()

# 첫 번째 서브플롯에 그래프 그리기
color = 'tab:red'
ax1.set_xlabel("#Episode")
ax1.set_ylabel("Epsilon", color=color)
ax1.plot(list(range(numEpisodes)), episode_epsilons, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 두 번째 서브플롯 생성
ax2 = ax1.twinx()

# 두 번째 서브플롯에 그래프 그리기
# 각 state에 대해 그래프 출력
color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue[0], color=color, label="Q(S1,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue[0][-1], " Q(S1,a)", ha='left', va='center', color='#555555')

color = '#dddddd'
ax2.plot(list(range(numEpisodes)), print_maxQvalue[1], color=color, label="Q(S2,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue[1][-1], " Q(S2,a)", ha='left', va='center', color='#555555')

color = '#cdcdcd'
ax2.plot(list(range(numEpisodes)), print_maxQvalue[2], color=color, label="Q(S3,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue[2][-1], " Q(S3,a)", ha='left', va='center', color='#555555')

color = '#cccccc'
ax2.plot(list(range(numEpisodes)), print_maxQvalue[3], color=color, label="Q(S4,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue[3][-1], " Q(S4,a)", ha='left', va='center', color='#555555')

color = '#bcbcbc'
ax2.plot(list(range(numEpisodes)), print_maxQvalue[4], color=color, label="Q(S5,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue[4][-1], " Q(S5,a)", ha='left', va='center', color='#555555')

color = '#aaaaaa'
ax2.plot(list(range(numEpisodes)), print_maxQvalue[5], color=color, label="Q(S6,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue[5][-1], " Q(S6,a)", ha='left', va='center', color='#555555')

color = 'tab:blue'
ax2.set_ylabel("avg of maxQvalue", color=color)
ax2.plot(list(range(numEpisodes)), avg_maxQvalue, color=color, label="Average of maxQ(S,a)")
ax2.tick_params(axis='y', labelcolor='black')

# 그래프 출력
fig.tight_layout()
plt.show()
	
