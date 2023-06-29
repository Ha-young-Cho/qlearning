# sql 파일 뽑히면 수동으로 파일에 추가해야하는 것
# 초기 q_table 값들 insert해줘야함. (INSERT INTO q_table VALUES.....)
# 첫 episode_table record에 id 값 1로 설정해줘야함.

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

# 가우시안 분포로 reward 생성하는 함수
def generateGaussianReward(mean, stddev):
    # 랜덤 엔진 초기화
    generator = np.random.default_rng()
    distribution = generator.normal(mean, stddev)
    # 가우시안 분포를 따르는 랜덤 값을 생성하고 반환
    reward = distribution
    return reward

NUM_TERMINAL = 3
NUM_STATES = 9 # 6 + terminal state 3개
NUM_ACTION = 5
ACTION = [[0, 1, 3], [0, 1, 3], [2, 3], [3, 4], [3, 4], [1, 2]]
# print_maxQvalue = [[], [], [], [], [], []]

# 하이퍼파라미터 설정
alpha = 0.005 # 학습률
gamma = 0.99 # 할인인자
numEpisodes = 7000 # 에피소드 횟수 
kInitialEpsilon = 0.9 # 초기 입실론 값
kMinEpsilon = 0.05 # 입실론 값의 최소값
kDecayRate = 0.9995 # 입실론 값의 감소 비율(= 기존 앱실론 값의 99.95%로 감소)

policy = EpsilonGreedyPolicy(kInitialEpsilon, kMinEpsilon, kDecayRate) # 입실론 그리디 정책 객체 생성

print(f"\nalpha : {alpha}")
print(f"gamma : {gamma}")
print(f"num of episode : {numEpisodes}")
print(f"initial epsilon : {kInitialEpsilon}")
print(f"min epsilon : {kMinEpsilon}")
print(f"decay rate : {kDecayRate}\n")

# qtable 초기화
qTable = np.full((NUM_STATES, NUM_ACTION), -50.0)

print("Initial Q-table")
for i in range(NUM_STATES - NUM_TERMINAL):
    print(f"State {i+1}: {qTable[i]}")


#랜덤 이름 설정을 위한 변수
lastname = '김이박최조정곽배유오권홍강남문민방양서손송신심안윤임장전채한허황'
firstname1 = '가나다아하시이지서소오호우주기시이지'
firstname2 = '영현지서선연리하준우민미혜원진린주훈솜윤은희정빈규호섭'
alphabet_big = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet_small = 'abcdefghijklmnopqrstuvwxyz'

#sql 작성을 위한 base문
sql = []
base_member_table= "INSERT INTO member_table (id, member_password, member_name) VALUES "
base_patient = "INSERT INTO patient_table (id, patient_age, patient_gender, patient_name, member_id) VALUES "
base_patient_record = "INSERT INTO patient_record_table (patient_id, visit_date, state, reward, blood_pressure, blood, action, ecg, turn) VALUES "
base_final_patient_record = "INSERT INTO patient_record_table (patient_id, visit_date, state, turn) VALUES "
base_episode_table = "INSERT INTO episode_table (s1, s2, s3, s4, s5, s6, epsilon) VALUES "
base_q_table = "UPDATE q_table SET "

#의사 정보 설정
doctor_id = [11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888, 99999, 12121, 23232, 34343, 45454, 56565, 67676, 78787, 89898, 90909] #18명
for i in range(0, 18):
    query_doctor = ""
    doctor_pw = "" #10자리(id 5자리 + pw 5자리)
    doctor_name = ""
    for j in range(0, 5):
        index = random.randint(0, len(alphabet_small)-1);
        doctor_pw = doctor_pw + alphabet_small[index]
    doctor_pw = doctor_pw + str(doctor_id[i])
    lname_idx = random.randint(0, len(lastname)-1)
    fname1_idx = random.randint(0, len(firstname1)-1)
    fname2_idx = random.randint(0, len(firstname2)-1)
    doctor_name = lastname[lname_idx] + firstname1[fname1_idx] + firstname2[fname2_idx]
    query_doctor = base_member_table + '(' + str(doctor_id[i]) + ', "' + doctor_pw + '", "' + doctor_name + '");\n'
    sql.append(query_doctor)

#patient_record_table에서 사용할 변수 정리
visit_year = 2000
visit_month = 1
visit_day = 1
visit_temp = 0

for i in range(numEpisodes):
    
    #환자 정보 설정
    turn = 1
    query_patient=""
    patient_name=""
    patient_id = random.randint(1000000000, 9999999999) #10자리
    patient_age = random.randint(15, 90)
    patient_gender = random.choice(["M", "F"])
    lname_idx = random.randint(0, len(lastname)-1)
    fname1_idx = random.randint(0, len(firstname1)-1)
    fname2_idx = random.randint(0, len(firstname2)-1)
    idx = random.randint(0, 17)
    patient_name = lastname[lname_idx] + firstname1[fname1_idx] + firstname2[fname2_idx]
    query_patient = base_patient + '(' + str(patient_id) + ', ' + str(patient_age)+ ', "' + patient_gender + '", "'+ patient_name + '", ' + str(doctor_id[idx]) + ');\n'
    sql.append(query_patient)

    # 본격 qlearning 시작
    # 난수 발생하여 초기 상태 선택
    random_number = random.randint(10000, 99999)
    random.seed(random_number)
    currentState = random.randint(0, NUM_STATES - NUM_TERMINAL - 1)

    # 에피소드 실행 반복문. 터미널 state 만나면 종료
    while currentState < (NUM_STATES - NUM_TERMINAL):

        # 행동 선택 by 입실론 그리디
        chosen_action = policy.choose_action(qTable, currentState)
        
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
                reward = generateGaussianReward(0.0, 1.0)
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
        
        #patient_record_table 데이터 생성
        query_patient_record = ""
        
        visit_date = ""
        visit_temp = visit_temp + 1
        if visit_temp % 100 == 0:
            visit_day = visit_day + 1
            if visit_month in [1, 3, 5, 7, 9, 11]:
                if visit_day==32:
                    visit_day = 1
                    visit_month = visit_month + 1
            elif visit_month == 2:
                if visit_day==29:
                    visit_day = 1
                    visit_month = visit_month + 1
            elif visit_month in [4, 6, 8, 10]:
                if visit_day==31:
                    visit_day = 1
                    visit_month = visit_month + 1
            elif visit_month == 12:
                if visit_day==32:
                    visit_day = 1
                    visit_month = 1
                    visit_year = visit_year + 1
        visit_date = str(visit_year).zfill(4) + "-" + str(visit_month).zfill(2) + "-" + str(visit_day).zfill(2)

        blood_pressure = random.choice(["높음", "정상", "저하"])
        blood = random.choice(["높음", "정상", "저하"])
        ecg = random.randint(50, 120)
        query_patient_record = base_patient_record + '(' + str(patient_id) + ', "' + visit_date+ '", ' + str(currentState+1) + ', '+ str(reward) + ', "' + blood_pressure + '", "' + blood + '", ' + str(chosen_action+1) + ', ' + str(ecg) + ', ' + str(turn) + ');\n'
        sql.append(query_patient_record)

        #다음 상태로 이동
        currentState = nextState
        turn = turn + 1
        
        # 입실론 값 갱신
        policy.update_epsilon() 

        # maxQ(S,a) 값 갱신
        maxQvalue = 0.0
        print_maxQvalue = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for k in range(0, NUM_STATES-NUM_TERMINAL):
            if 2 <= k <= 5:
                temp_l = 2
            elif 0 <= k <= 1:
                temp_l = 3
            maxQvalue = qTable[k][ACTION[k][0]]
            for l in range(1, temp_l):
                if qTable[k][ACTION[k][l]] > maxQvalue:
                    maxQvalue = qTable[k][ACTION[k][l]]
            print_maxQvalue[k] = maxQvalue

        #episode_table 데이터 생성
        query_episode_table = ""
        query_episode_table = base_episode_table + '(' + str(print_maxQvalue[0])+ ', ' + str(print_maxQvalue[1]) + ', ' + str(print_maxQvalue[2]) + ', ' + str(print_maxQvalue[3])+ ', ' + str(print_maxQvalue[4]) + ', ' + str(print_maxQvalue[5]) + ', ' + str(policy.epsilon) + ');\n'
        sql.append(query_episode_table)

        # base_q_table 데이터 생성
        query_q_table11 = base_q_table + 'maxq = ' + str(qTable[0][0]) + ' WHERE id = 11;\n'
        query_q_table12 = base_q_table + 'maxq = ' + str(qTable[0][1]) + ' WHERE id = 12;\n'
        query_q_table14 = base_q_table + 'maxq = ' + str(qTable[0][3]) + ' WHERE id = 14;\n'
        query_q_table21 = base_q_table + 'maxq = ' + str(qTable[1][0]) + ' WHERE id = 21;\n'
        query_q_table22 = base_q_table + 'maxq = ' + str(qTable[1][1]) + ' WHERE id = 22;\n'
        query_q_table24 = base_q_table + 'maxq = ' + str(qTable[1][3]) + ' WHERE id = 24;\n'
        query_q_table33 = base_q_table + 'maxq = ' + str(qTable[2][2]) + ' WHERE id = 33;\n'
        query_q_table34 = base_q_table + 'maxq = ' + str(qTable[2][3]) + ' WHERE id = 34;\n'
        query_q_table44 = base_q_table + 'maxq = ' + str(qTable[3][3]) + ' WHERE id = 44;\n'
        query_q_table45 = base_q_table + 'maxq = ' + str(qTable[3][4]) + ' WHERE id = 45;\n'
        query_q_table54 = base_q_table + 'maxq = ' + str(qTable[4][3]) + ' WHERE id = 54;\n'
        query_q_table55 = base_q_table + 'maxq = ' + str(qTable[4][4]) + ' WHERE id = 55;\n'
        query_q_table62 = base_q_table + 'maxq = ' + str(qTable[5][1]) + ' WHERE id = 62;\n'
        query_q_table63 = base_q_table + 'maxq = ' + str(qTable[5][2]) + ' WHERE id = 63;\n'
        sql.append(query_q_table11)
        sql.append(query_q_table12)
        sql.append(query_q_table14)
        sql.append(query_q_table21)
        sql.append(query_q_table22)
        sql.append(query_q_table24)
        sql.append(query_q_table33)
        sql.append(query_q_table34)
        sql.append(query_q_table44)
        sql.append(query_q_table45)
        sql.append(query_q_table54)
        sql.append(query_q_table55)
        sql.append(query_q_table62)
        sql.append(query_q_table63)

    #terminal state 데이터 생성
    visit_date = ""
    visit_day = visit_day + 1
    if visit_month in [1, 3, 5, 7, 9, 11]:
        if visit_day==32:
            visit_day = 1
            visit_month = visit_month + 1
    elif visit_month == 2:
        if visit_day==29:
            visit_day = 1
            visit_month = visit_month + 1
    elif visit_month in [4, 6, 8, 10]:
        if visit_day==31:
            visit_day = 1
            visit_month = visit_month + 1
    elif visit_month == 12:
        if visit_day==32:
            visit_day = 1
            visit_month = 1
            visit_year = visit_year + 1
    visit_date = str(visit_year).zfill(4) + "-" + str(visit_month).zfill(2) + "-" + str(visit_day).zfill(2)
    query_final_patient_record = base_final_patient_record + '(' + str(patient_id) + ', "' + visit_date+ '", ' + str(currentState+1) + ', ' + str(turn) + ');\n'
    sql.append(query_final_patient_record)


#학습 결과 출력
print("Final Q-table")
for i in range(NUM_STATES - NUM_TERMINAL):
    print("State ", i + 1, ": ", end="")
    for j in range(NUM_ACTION):
        print(qTable[i][j], " | ", end="")
    print()
print("final epsilon : ", policy.epsilon)

#sql 파일로 뽑기
f = open('createPatient.sql', 'w')
for i, s in enumerate(sql):
    f.writelines(s)

f.close()