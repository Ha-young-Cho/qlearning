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

    public void updateEpsilon(double decayRate) {
        epsilon = max(epsilon *= decayRate, minEpsilon);
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

    private static List<Double> episodeEpsilons = new ArrayList<>();
    private static List<Double> avgMaxQValues = new ArrayList<>();
    private static List<List<Double>> printMaxQValues = new ArrayList<>();
    private static List<Double> maxQValues = new ArrayList<>();

    private static double generateGaussianReward(double mean, double stdDev) {
        Random rand = new Random();
        return mean + stdDev * rand.nextGaussian();
    }

    public static void main(String[] args) {
        qTable = new double[NUM_STATES][NUM_ACTIONS];

        // Q-Table 초기화
        for (int i = 0; i < NUM_STATES; i++) {
            for (int j = 0; j < NUM_ACTIONS; j++) {
                qTable[i][j] = -50.0;
            }
        }

        double alpha = 0.005;              // 학습률
        double gamma = 0.99;              // 할인율
        int numEpisodes = 10000;          // 학습 반복 횟수
        double initialEpsilon = 0.9;     // 초기 입실론 값
        double minEpsilon = 0.05;        // 최소 입실론 값
        double decayRate = 0.9995;         // 입실론 감소 비율

        double reward = 0.0; 
        int nextState = 0;
        double randValue = 0.0;
        //double epsilon = initialEpsilon;
        int state = 0;
        int action = 0;
        double maxNextQvalue = 0.0;
        double maxQValueSum = 0.0;

        EpsilonGreedyPolicy policy = new EpsilonGreedyPolicy(initialEpsilon);

        // 학습 반복문
        for (int episode = 0; episode < numEpisodes; episode++) {

            // 에피소드 진행
            state = new Random().nextInt(NUM_STATES);
            while (state < NUM_TERMINAL) {
                action = policy.chooseAction(qTable, state);

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
                maxNextQValue = qTable[nextState][ACTION[nextState][0]];
                for(int j=1; j<ACTION[nextState].length; j++){
                    if (qTable[nextState][ACTION[nextState][j]] > maxNextQValue) {
                        maxNextQValue = qTable[nextState][ACTION[nextState][j]];
                    }
                }
                qTable[state][action] += alpha * (reward + gamma * maxNextQValue - qTable[state][action]);

                state = nextState;

                
                // maxQValueSum = 0.0;
                // maxQValues.add(maxNextQValue);
                // maxQValueSum += maxNextQValue;
            }

            episodeEpsilons.add(policy.epsilon);
            policy.updateEpsilon(decayRate);

            //avgMaxQValues.add(maxQValueSum / maxNextQValue.size());
            //printMaxQValues.add(maxNextQValue);
        }

        // 학습 결과 출력
        for (int i = 0; i < NUM_STATES - NUM_TERMINAL; i++) {
            for (int j = 0; j < NUM_ACTIONS; j++) {
                System.out.printf("Q[%d][%d]: %.2f\n", i+1, j+1, qTable[i][j]);
            }
        }

        // 입실론 값과 최대 Q값의 평균 그래프 출력 등의 추가 작업 가능
    }
}