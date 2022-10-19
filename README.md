컴퓨터비전응용 과제물<br>
컴퓨터비전을 이용한 마스크 착용 단속 프로그램 서버<br>

사용 라이브러리, 툴, 모델:<br>
Tensorflow(Keras),Opencv,cvlib(https://github.com/arunponnusamy/cvlib),Densenet,Mobilenet<br>
<br>
간단설명:
영상 스트림을 videoClient가 프레임별로 웹 소켓을 통해 메인 서버로 전송하면<br>
cvlib의 face detect를 통해 얼굴별로 프레임을 나누고, 해당 이미지에 잡힌 사람이<br>
마스크를 착용하였는지 여부를 판단 및 web page상에 표출하는 프로그램 <br><br><br>



npm과 node를 설치 후 진행.<br>
officerweb : 마스크 미착용자 고지용 클라이언트 웹(React)으로 npm i를 실행 해 필요한 파일들 다운로드 후 npm start로 웹 서버 시작.<br>
Server : 비전 프로세스에 필요한 서버 코드 폴더.<br>
socketServer: 웹소켓 중계 서버, (node.js)로 마찬가지로 npm i 후 npm start로 시작.<br>
videoClient.py : 비디오 송신용 프로그램. python videoClient.py로 시작.<br>
videoProcess.py : 비디오 처리 및 마스크 착용 여부 분류기(ML서버)<br>
modelLearning.py : 이미지 분류기 학습용 코드<br>
requirements.txt에 들어있는 모듈들을 pip install -r requirements.txt 로 설치.<br>
<br>
logs : 학습 과정 로그, tensorboard --logdir=./logs/fit로 tensorboard 실행 후 <br>
브라우저를 통해 localhost:6006에 접속하면 그래프 및 정보 확인 가능.<br>


https://bskyvision.com/1082  해당 소스코드를 참고하여 진행하였음.
