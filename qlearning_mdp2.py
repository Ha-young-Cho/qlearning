import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class EpsilonGreedyPolicy:
    def __init__(self, initial_epsilon, min_epsilon, decay_rate):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.ACTION = [[2,4,5], [0,4,5], [0,2,3], [1,3], [0,2,4], [1,3,5], [0,3], [0,4], [1,3,5], [1,2,4], [2,3,5], [2,5]]
    
    def choose_action(self, qTable, current_state):
        if random.random() < self.epsilon: # exploration
            if current_state in [0,1,2,4,5,8,9,10]:
                randNum = random.randint(0, 2)
            else:
                randNum = random.randint(0, 1)
            choose_action = self.ACTION[current_state][randNum]
            return choose_action
        else: # exploitation
            max_index = self.ACTION[current_state][0]
            temp = 0
            max_Qvalue = qTable[current_state][self.ACTION[current_state][0]]

            if current_state in [0,1,2,4,5,8,9,10]:
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

NUM_TERMINAL = 3
NUM_STATES = 15 # 12 + terminal state 3개
NUM_ACTION = 6
ACTION = [[2,4,5], [0,4,5], [0,2,3], [1,3], [0,2,4], [1,3,5], [0,3], [0,4], [1,3,5], [1,2,4], [2,3,5], [2,5]]
episode_epsilons1 = []
episode_epsilons2 = []
avg_maxQvalue1 = []
avg_maxQvalue2 = []
print_maxQvalue1 = [[], [], [], [], [], [], [], [], [], [], [], []]
print_maxQvalue2 = [[], [], [], [], [], [], [], [], [], [], [], []]

# 가우시안 분포로 reward 생성하는 함수
def generateGaussianReward(mean, stddev):
    # 랜덤 엔진 초기화
    generator = np.random.default_rng()
    distribution = generator.normal(mean, stddev)
    # 가우시안 분포를 따르는 랜덤 값을 생성하고 반환
    reward = distribution
    return reward

# qtable 초기화
qTable1 = np.full((NUM_STATES, NUM_ACTION), -50.0)
qTable2 = np.full((NUM_STATES, NUM_ACTION), -50.0)

print("Initial Q-table")
for i in range(NUM_STATES - NUM_TERMINAL):
    print(f"State {i+1}: {qTable1[i]}")

# 학습 파라미터 설정
alpha = 0.005 # 학습률
gamma = 0.99 # 할인인자
numEpisodes = 10000 # 에피소드 횟수 

kInitialEpsilon = 0.9 # 초기 입실론 값
kMinEpsilon = 0.05 # 입실론 값의 최소값
kDecayRate = 0.9995 # 입실론 값의 감소 비율(= 기존 앱실론 값의 99.95%로 감소)

policy1 = EpsilonGreedyPolicy(kInitialEpsilon, kMinEpsilon, kDecayRate) # 입실론 그리디 정책 객체 생성
policy2 = EpsilonGreedyPolicy(1,1,1) # 입실론 그리디 정책 객체 생성

print(f"\nalpha : {alpha}")
print(f"gamma : {gamma}")
print(f"num of episode : {numEpisodes}")
print(f"initial epsilon : {kInitialEpsilon}")
print(f"min epsilon : {kMinEpsilon}")
print(f"decay rate : {kDecayRate}\n")

def qlearning(qTable, avg_maxQvalue, policy, episode_epsilons, print_maxQvalue):
    for i in range(numEpisodes):

        # 난수 설정
        random_number = random.randint(10000, 99999)
        random.seed(random_number)

        # 초기 상태 선택
        currentState = random.randint(0, NUM_STATES - NUM_TERMINAL - 1)

        rand_index = policy.choose_action(qTable, currentState)
        
        # print("first state : ", currentState + 1)
        #print("epsilon : ", policy.epsilon)

        # 에피소드 실행 반복문. 터미널 state 만나면 종료
        while currentState < (NUM_STATES - NUM_TERMINAL):
            
            # 행동 선택 by 입실론 그리디
            chosen_action = policy.choose_action(qTable, currentState)
            if policy.epsilon == 1:
                chosen_action = rand_index
            
            #print("chosen action : ", chosen_action + 1)

            # 보상과 다음 상태 결정
            reward = 0.0
            nextState = 0

            if currentState == 0:  # MDP에서는 S1
                if chosen_action == 2:  # a3
                    randValue = random.random()
                    if randValue < 0.35:
                        reward = generateGaussianReward(-20.0, 1.0)
                        nextState = 5  # s6
                    elif 0.35 <= randValue < 0.85:
                        reward = generateGaussianReward(-35.0, 0.5)
                        nextState = 2  # s3
                    else:
                        reward = generateGaussianReward(-140.0, 2.0)
                        nextState = 14  # t3
                elif chosen_action == 4:  # a5
                    randValue = random.random()
                    if randValue < 0.6:
                        reward = generateGaussianReward(-32.0, 1.0)
                        nextState = 3  # s4
                    elif 0.6 <= randValue < 0.9:
                        reward = generateGaussianReward(-18.0, 0.5)
                        nextState = 6  # s7
                    else:
                        reward = generateGaussianReward(-10.0, 0.1)
                        nextState = 9  # s10
                elif chosen_action == 5:  # a6
                    randValue = random.random()
                    if randValue < 0.8:
                        reward = generateGaussianReward(-5.0, 0.5)
                        nextState = 11  # s12
                    else:
                        reward = generateGaussianReward(-68.0, 1.0)
                        nextState = 13  # t2

            elif currentState == 1:  # s2
                if chosen_action == 0:  # a1
                    randValue = random.random()
                    if randValue < 0.7:
                        reward = generateGaussianReward(-32.0, 1.0)
                        nextState = 4  # s5
                    else:
                        reward = generateGaussianReward(-45.0, 1.0)
                        nextState = 3  # s4
                elif chosen_action == 4:  # a5
                    randValue = random.random()
                    if randValue < 0.4:
                        reward = generateGaussianReward(-25.0, 1.0)
                        nextState = 6  # s7
                    elif 0.4 <= randValue < 0.8:
                        reward = generateGaussianReward(-22.0, 1.0)
                        nextState = 7  # s8
                    else:
                        reward = generateGaussianReward(-17.0, 1.0)
                        nextState = 8  # s9
                elif chosen_action == 5:  # a6
                    randValue = random.random()
                    if randValue < 0.7:
                        reward = generateGaussianReward(-13.0, 0.5)
                        nextState = 9  # s10
                    else:
                        reward = generateGaussianReward(-10.0, 0.5)
                        nextState = 10  # s11

            elif currentState == 2:  # s3
                if chosen_action == 0:  # a1
                    reward = generateGaussianReward(-35.0, 2.0)
                    nextState = 5 # s6
                elif chosen_action == 2:  # a3
                    randValue = random.random()
                    if randValue < 0.8:
                        reward = generateGaussianReward(-25.0, 1.0)
                        nextState = 7  # s8
                    else:
                        reward = generateGaussianReward(-100.0, 1.0)
                        nextState = 0  # s1
                elif chosen_action == 3:  # a4
                    randValue = random.random()
                    if randValue < 0.8:
                        reward = generateGaussianReward(-45.0, 2.0)
                        nextState = 4  # s5
                    else:
                        reward = generateGaussianReward(-85.0, 3.0)
                        nextState = 1  # s2

            elif currentState == 3:  # s4
                if chosen_action == 1:  # a2
                    randValue = random.random()
                    if randValue < 0.75:
                        reward = generateGaussianReward(-18.0, 1.0)
                        nextState = 10  # s11
                    else:
                        reward = generateGaussianReward(-150.0, 1.0)
                        nextState = 14  # t3
                elif chosen_action == 3:  # a4
                    randValue = random.random()
                    if randValue < 0.6:
                        reward = generateGaussianReward(-55.0, 1.0)
                        nextState = 3  # s4
                    else:
                        reward = generateGaussianReward(-50.0, 0.5)
                        nextState = 4  # s5

            elif currentState == 4: # s5
                if chosen_action == 0:  # a1
                    reward = generateGaussianReward(-35.0, 1.0)
                    nextState = 7 # s8
                elif chosen_action == 2:  # a3
                    randValue = random.random()
                    if randValue < 0.8:
                        reward = generateGaussianReward(-48.0, 1.0)
                        nextState = 6  # s7
                    else:
                        reward = generateGaussianReward(-95.0, 2.0)
                        nextState = 1  # s2
                elif chosen_action == 4:  # a5
                    randValue = random.random()
                    if randValue < 0.5:
                        reward = generateGaussianReward(-22.0, 1.0)
                        nextState = 9  # s10
                    elif 0.5 <= randValue < 0.8:
                        reward = generateGaussianReward(-18.0, 1.0)
                        nextState = 10  # s11
                    else:
                        reward = generateGaussianReward(-16.0, 0.1)
                        nextState = 11  # s12

            elif currentState == 5: # s6
                if chosen_action == 1:  # a2
                    reward = generateGaussianReward(-52.0, 0.1)
                    nextState = 6 # s7
                elif chosen_action == 3:  # a4
                    randValue = random.random()
                    if randValue < 0.7:
                        reward = generateGaussianReward(-50.0, 2.0)
                        nextState = 7  # s8
                    else:
                        reward = generateGaussianReward(-33.0, 1.0)
                        nextState = 9  # s10
                elif chosen_action == 5:  # a6
                    randValue = random.random()
                    if randValue < 0.4:
                        reward = generateGaussianReward(-22.0, 1.0)
                        nextState = 10  # s11
                    elif 0.4 <= randValue < 0.8:
                        reward = generateGaussianReward(-20.0, 0.5)
                        nextState = 11  # s12
                    else:
                        reward = generateGaussianReward(50.0, 1.0)
                        nextState = 12  # t1
            
            elif currentState == 6: # s7
                if chosen_action == 0:  # a1
                    reward = generateGaussianReward(-47.0, 0.1)
                    nextState = 8 # s9
                elif chosen_action == 3:  # a4
                    randValue = random.random()
                    if randValue < 0.6:
                        reward = generateGaussianReward(-55.0, 1.0)
                        nextState = 7  # s8
                    else:
                        reward = generateGaussianReward(-40.0, 2.0)
                        nextState = 9  # s10

            elif currentState == 7: # s8
                if chosen_action == 0:  # a1
                    randValue = random.random()
                    if randValue < 0.7:
                        reward = generateGaussianReward(-53.0, 1.0)
                        nextState = 7  # s8
                    else:
                        reward = generateGaussianReward(-58.0, 0.5)
                        nextState = 8  # s9
                elif chosen_action == 4:  # a5
                    reward = generateGaussianReward(-37.0, 2.0)
                    nextState = 10 # s11
            
            elif currentState == 8: # s9
                if chosen_action == 1:  # a2
                    randValue = random.random()
                    if randValue < 0.6:
                        reward = generateGaussianReward(-49.0, 1.0)
                        nextState = 10  # s11
                    elif 0.6 <= randValue < 0.9:
                        reward = generateGaussianReward(-43.0, 1.0)
                        nextState = 11  # s12
                    else:
                        reward = generateGaussianReward(-90.0, 3.0)
                        nextState = 2  #s3
                elif chosen_action == 3:  # a4
                    reward = generateGaussianReward(-53.0, 1.0)
                    nextState = 9 # s10
                elif chosen_action == 5:  # a6
                    reward = generateGaussianReward(50.0, 1.0)
                    nextState = 12 #t1

            elif currentState == 9: # s10
                if chosen_action == 1:  # a2
                    reward = generateGaussianReward(-53.0, 0.5)
                    nextState = 10 # s11
                elif chosen_action == 2:  # a3
                    randValue = random.random()
                    if randValue < 0.7:
                        reward = generateGaussianReward(-48.0, 1.0)
                        nextState = 11  # s12
                    else:
                        reward = generateGaussianReward(-58.0, 1.0)
                        nextState = 13  #t2
                elif chosen_action == 4:  # a5
                    randValue = random.random()
                    if randValue < 0.9:
                        reward = generateGaussianReward(30.0, 1.0)
                        nextState = 12  #t1
                    else:
                        reward = generateGaussianReward(-70.0, 0.5)
                        nextState = 13  #t2

            elif currentState == 10: # s11
                if chosen_action == 2:  # a3
                    randValue = random.random()
                    if randValue < 0.45:
                        reward = generateGaussianReward(25.0, 2.0)
                        nextState = 12  #t1
                    elif 0.45 <= randValue < 0.85:
                        reward = generateGaussianReward(-60.0, 3.0)
                        nextState = 13  #t2
                    else:
                        reward = generateGaussianReward(-140.0, 2.0)
                        nextState = 14  #t3
                elif chosen_action == 3:  # a4
                    reward = generateGaussianReward(-55.0, 0.5)
                    nextState = 11  #s12
                elif chosen_action == 5:  # a6
                    reward = generateGaussianReward(45.0, 0.1)
                    nextState = 12  #t1

            elif currentState == 11: # s12
                if chosen_action == 2:  # a3
                    randValue = random.random()
                    if randValue < 0.8:
                        reward = generateGaussianReward(25.0, 2.0)
                        nextState = 12  #t1
                    else:
                        reward = generateGaussianReward(-60.0, 1.0)
                        nextState = 13  #t2
                elif chosen_action == 5:  # a6
                    reward = generateGaussianReward(45.0, 0.1)
                    nextState = 12 #t1

            #print("immediate reward : ", reward)

            #q-value 함수 업데이트
            maxNextQValue = 0.0
            if 0 <= nextState <= 11:
                if nextState in [0,1,2,4,5,8,9,10]:
                    temp_j = 3
                else:
                    temp_j = 2

                if policy.epsilon == 1:
                    rand_index = policy.choose_action(qTable, nextState)
                    maxNextQValue = qTable[nextState][rand_index]
                else:
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
        episode_epsilons.append(policy.epsilon)

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
            if k in [0,1,2,4,5,8,9,10]:
                temp_l = 3
            else:
                temp_l = 2
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


qlearning(qTable1, avg_maxQvalue1, policy1, episode_epsilons1, print_maxQvalue1);
qlearning(qTable2, avg_maxQvalue2, policy2, episode_epsilons2, print_maxQvalue2);

# Figure 객체 생성과 서브플롯 생성
fig, ax1 = plt.subplots()

# 첫 번째 서브플롯에 그래프 그리기
color = 'tab:red'
ax1.set_xlabel("#Episode")
ax1.set_ylabel("Epsilon", color=color)
ax1.plot(list(range(numEpisodes)), episode_epsilons1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 두 번째 서브플롯 생성
ax2 = ax1.twinx()

# 두 번째 서브플롯에 그래프 그리기
# 각 state에 대해 그래프 출력
color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[0], color=color, label="Q(S1,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[0][-1], "S1", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[1], color=color, label="Q(S2,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[1][-1], "S2", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[2], color=color, label="Q(S3,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[2][-1], "S3", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[3], color=color, label="Q(S4,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[3][-1], "S4", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[4], color=color, label="Q(S5,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[4][-1], "S5", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[5], color=color, label="Q(S6,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[5][-1], "S6", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[6], color=color, label="Q(S7,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[6][-1], "S7", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[7], color=color, label="Q(S8,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[7][-1], "S8", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[8], color=color, label="Q(S9,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[8][-1], "S9", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[9], color=color, label="Q(S10,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[9][-1], "S10", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[10], color=color, label="Q(S11,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[10][-1], "S11", ha='left', va='center', color='#555555')

color = '#dfdfdf'
ax2.plot(list(range(numEpisodes)), print_maxQvalue1[11], color=color, label="Q(S12,a)")
ax2.text(list(range(numEpisodes))[-1], print_maxQvalue1[11][-1], "S12", ha='left', va='center', color='#555555')

color = 'tab:blue'
ax2.set_ylabel("avg of maxQvalue", color=color)
ax2.plot(list(range(numEpisodes)), avg_maxQvalue1, color=color, label="Average of maxQ(S,a)")
ax2.tick_params(axis='y', labelcolor='black')

color = 'tab:green'
ax2.plot(list(range(numEpisodes)), avg_maxQvalue2, color=color, label="Average of maxQ(S,a): only exploration")
ax2.tick_params(axis='y', labelcolor='black')

# 그래프 출력
fig.tight_layout()

variable_x = mpatches.Patch(color='#dfdfdf',label='MaxQ in each state')
variable_y = mpatches.Patch(color='red',label='Epsilon')
variable_z = mpatches.Patch(color='blue',label='Average of maxQ in our system')
variable_a = mpatches.Patch(color='green',label='Average of maxQ in traditional methods')
plt.legend(handles=[variable_x, variable_y, variable_z, variable_a], loc ='upper center', frameon=False, fontsize = 10)

plt.show()
	
