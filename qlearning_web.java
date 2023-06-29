import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class EpsilonGreedyPolicy {
    private double epsilon;
    private int[][] action = {{0,1,3}, {0,1,3}, {2,3}, {3,4}, {3,4}, {1,2}}

    public EpsilonGreedyPolicy(double epsilon) {
        this.epsilon = epsilon;
    }

    public int chooseAction(double[][] qTable, int state) {
        int numActions = action[state].length; //시행할 수 있는 action 개수
        Random rand = new Random();

        if (rand.nextDouble() < epsilon) { // exploration
            int randomIndex = rand.nextInt(numActions);
            return action[state][randomIndex];
        } 
        else { //exploitation
            int max_index = action[state][0];

            for (int i = 1; i < numActions; i++) {
                if (qTable[state][action[state][i]] > qTable[state][max_index]) {
                    max_index = action[state][i];
                }  
            }
            
            return max_index;
        }
    }

    public void updateEpsilon(double decayRate, double minEpsilon) {
        epsilon = Math.max(epsilon *= decayRate, minEpsilon);
    }
}

public class QLearning {
    private static final int NUM_TERMINAL = 3;
    private static final int NUM_STATES = 9; // state + terminal
    private static final int NUM_ACTIONS = 5;

    private static final int[][] ACTION = {
            {0, 1, 3},   // 상태 1에서 가능한 행동들
            {0, 1, 3},   // 상태 2에서 가능한 행동들
            {2, 3},   // 상태 3에서 가능한 행동들
            {3, 4},   // 상태 4에서 가능한 행동들
            {3, 4},   // 상태 5에서 가능한 행동들
            {1, 2}  // 상태 6에서 가능한 행동들
    };

    private static double[][] qTable;

    private static double generateGaussianReward(double mean, double stdDev) {
        Random rand = new Random();
        return mean + stdDev * rand.nextGaussian();
    }

    public static void main(String[] args) {
        qTable = new double[NUM_STATES][NUM_ACTIONS];

        //!!!!!!!!!!!!!!!!!!!!입력: db에서 qtable 값 불러와서 초기화해야함!!!!!!!!!!!!!!!!!
        //index 잘 확인하기. db에서 state1은 코드에서 state index 0임. 액션도 마찬가지!
        // Q-Table 초기화
        for (int i = 0; i < NUM_STATES; i++) {
            for (int j = 0; j < NUM_ACTIONS; j++) {
                qTable[i][j] = -50.0;
            }
        }

        //하이퍼파라미터
        double alpha = 0.005;              // 학습률
        double gamma = 0.99;              // 할인율
        double initialEpsilon = 0.9;     // 초기 입실론 값
        double minEpsilon = 0.05;        // 최소 입실론 값
        double decayRate = 0.9995;         // 입실론 감소 비율

        double reward = 0.0; 
        int nextState = 0;
        double randValue = 0.0;

        //!!!!!!!!!!!!!!!입력: 여기에 전달 받은 epsilon 값 넣기!!!!!!!!!!!!
        double epsilon = 0.9;


        EpsilonGreedyPolicy policy = new EpsilonGreedyPolicy(epsilon);

        // !!!!!!!!!!!!!!!!!!!입력: 여기에 html로부터 전달 받은 state를 넣어야 함!!!!!!!!!!!!!!!
        // !!!!!!!!!!!!!!!!!!!근데 index 차이 때문에 db값에서의 state - 1 값이 state에 들어가게 !!!!!!!!!!
        int state = new Random().nextInt(NUM_STATES);
    
        
        int action = policy.chooseAction(qTable, state);

        if (state == 0) { // S1
            if (action == 0) { // a1
                randValue = Math.random();
                if (randValue < 0.7) {
                    reward = generateGaussianReward(-40.0, 1.0);
                    nextState = 1; // s2
                } else {
                    reward = generateGaussianReward(-20.0, 0.5);
                    nextState = 2; // s3
                }
            } else if (action == 1) { // a2
                reward = generateGaussianReward(0.0, 1.0);
                nextState = 2; // s3
            } else if (action == 3) { // a4
                randValue = Math.random();
                if (randValue < 0.8) {
                    reward = generateGaussianReward(-30.0, 8.0);
                    nextState = 1; // s2
                } else {
                    reward = generateGaussianReward(-150.0, 3.0);
                    nextState = 8; // t3
                }
            }
        } else if (state == 1) { // s2
            if (action == 0) { // a1
                randValue = Math.random();
                if (randValue < 0.8) {
                    reward = generateGaussianReward(20.0, 1.0);
                    nextState = 6; // t1
                } else {
                    reward = generateGaussianReward(-80.0, 2.0);
                    nextState = 7; // t2
                }
            } else if (action == 1) { // a2
                reward = generateGaussianReward(-90.0, 1.0);
                nextState = 7; // t2
            } else if (action == 3) { // a4
                reward = generateGaussianReward(30.0, 3.0);
                nextState = 6; // t1
            }
        } else if (state == 2) { // s3
            if (action == 2) { // a3
                reward = generateGaussianReward(20.0, 2.0);
                nextState = 6; // t1
            } else if (action == 3) { // a4
                reward = generateGaussianReward(-80.0, 0.5);
                nextState = 0; // s1
            }
        } else if (state == 3) { // s4
            if (action == 3) { // a4
                randValue = Math.random();
                if (randValue < 0.9) {
                    reward = generateGaussianReward(0.0, 5.0);
                    nextState = 4; // s5
                } else {
                    reward = generateGaussianReward(-100.0, 1.0);
                    nextState = 5; // s6
                }
            } else if (action == 4) { // a5
                randValue = Math.random();
                if (randValue < 0.6) {
                    reward = generateGaussianReward(-10.0, 4.0);
                    nextState = 4; // s5
                } else if (randValue >= 0.6 && randValue < 0.9) {
                    reward = generateGaussianReward(-50.0, 1.0);
                    nextState = 3; // s4
                } else {
                    reward = generateGaussianReward(-80.0, 3.0);
                    nextState = 7; // t2
                }
            }
        } else if (state == 4) { // s5
            if (action == 3) { // a4
                randValue = Math.random();
                if (randValue < 0.7) {
                    reward = generateGaussianReward(-70.0, 2.0);
                    nextState = 7; // t2
                } else {
                    reward = generateGaussianReward(25.0, 2.0);
                    nextState = 6; // t1
                }
            } else if (action == 4) { // a5
                randValue = Math.random();
                if (randValue < 0.8) {
                    reward = generateGaussianReward(-10.0, 0.8);
                    nextState = 2; // s3
                } else {
                    reward = generateGaussianReward(-40.0, 0.5);
                    nextState = 4; // s5
                }
            }
        } else if (state == 5) { // s6
            if (action == 1) { // a2
                randValue = Math.random();
                if (randValue < 0.75) {
                    reward = generateGaussianReward(-10.0, 3.0);
                    nextState = 4; // s5
                } else {
                    reward = generateGaussianReward(0.0, 1.0);
                    nextState = 2; // s3
                }
            } else if (action == 2) { // a3
                reward = generateGaussianReward(-20.0, 4.0);
                nextState = 1; // s2
            }
        }

        // Q-Table 업데이트
        if(nextState < 6 ){
            double maxNextQValue = qTable[nextState][ACTION[nextState][0]];
            for(int j=1; j<ACTION[nextState].length; j++){
                if (qTable[nextState][ACTION[nextState][j]] > maxNextQValue) {
                    maxNextQValue = qTable[nextState][ACTION[nextState][j]];
                }
            }
        }
        else double maxNextQValue = 0.0;
        qTable[state][action] += alpha * (reward + gamma * maxNextQValue - qTable[state][action]);

        policy.updateEpsilon(decayRate, minEpsilon); //epsilon 업데이트

        //그래프를 위한 maxqvalue
        double maxqvalue1 = qTable[0][ACTION[0][0]];
        for(int j=1; j<ACTION[0].length; j++){
            if (qTable[0][ACTION[0][j]] > maxqvalue1) {
                maxqvalue1 = qTable[0][ACTION[0][j]];
            }
        }
        double maxqvalue2 = qTable[1][ACTION[1][0]];
        for(int j=1; j<ACTION[1].length; j++){
            if (qTable[1][ACTION[1][j]] > maxqvalue2) {
                maxqvalue2 = qTable[1][ACTION[1][j]];
            }
        }
        double maxqvalue3 = qTable[2][ACTION[2][0]];
        for(int j=1; j<ACTION[2].length; j++){
            if (qTable[2][ACTION[2][j]] > maxqvalue3) {
                maxqvalue3 = qTable[2][ACTION[2][j]];
            }
        }
        double maxqvalue4 = qTable[3][ACTION[3][0]];
        for(int j=1; j<ACTION[3].length; j++){
            if (qTable[3][ACTION[3][j]] > maxqvalue4) {
                maxqvalue4 = qTable[3][ACTION[3][j]];
            }
        }
        double maxqvalue5 = qTable[4][ACTION[4][0]];
        for(int j=1; j<ACTION[4].length; j++){
            if (qTable[4][ACTION[4][j]] > maxqvalue5) {
                maxqvalue5 = qTable[4][ACTION[4][j]];
            }
        }
        double maxqvalue6 = qTable[5][ACTION[5][0]];
        for(int j=1; j<ACTION[5].length; j++){
            if (qTable[5][ACTION[5][j]] > maxqvalue6) {
                maxqvalue6 = qTable[5][ACTION[5][j]];
            }
        }

        // 학습 결과 출력
        for (int i = 0; i < NUM_STATES - NUM_TERMINAL; i++) {
            for (int j = 0; j < NUM_ACTIONS; j++) {
                System.out.printf("Q[%d][%d]: %.2f\n", i+1, j+1, qTable[i][j]);
            }
        }

    }

    //!!!!!!!!!!!!DB에 반환할 내용 정리!!!!!!!!!!!
    //action -> patient_record_table
    //reward -> patient_record_table
    //nextState -> patient_record_table (다음 방문 회차에)
    //policy.epsilon -> episode_table
    //qtable -> q_table
    //maxqvalue1, maxqvalue2, maxqvalue3, maxqvalue4, maxqvalue5, maxqvalue6 -> episode_table

}
