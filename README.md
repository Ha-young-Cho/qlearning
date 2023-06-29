# qlearning
2023-1학기 정보통신종합설계 qlearning 코드

- qlearning_mdp1_sql.py : mdp1에 qlearning을 적용하여 n명의 환자 데이터를 생성하는 파이썬 파일
- qlearning_mdp1.py : mdp1에 qlearning을 실행시키는 파이썬 파일
- qlearning_mdp2.py : mdp2에 qlearning을 실행시키는 파이썬 파일
- qlearning.java : qlearning_mdp1.py을 java로 변환시킨 파일
- qlearning_web.java : qlearning.java를 웹에서 동작시키기 위해 변환시킨 파일 (DB 연동 등)
- createPatient.sql : qlearning_mdp1_sql.py로 생성된 환자 데이터로, DB에 삽입하기 위한 SQL 모음

### DB가 생성된 상태에서 sql 파일 뽑히면 수동으로 createPatient.sql 파일에 추가해야하는 것
초기 q_table 값들 insert해줘야함. (INSERT INTO q_table VALUES.....)
첫 episode_table record에 id 값 1로 설정해줘야함.
