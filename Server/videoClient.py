import base64
import cv2
import socketio
import time
#영상 전송 서버(단속 현장)

#소켓 연결
videoSocket = socketio.Client()
videoSocket.connect('http://15.164.111.113:5555') #서버 주소

#영상 input
videoStream = cv2.VideoCapture(0)
videoStream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
videoStream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while(True):
    time.sleep(1) #1초마다 비디오 인풋전송
    status , frame = videoStream.read()
    #영상 읽어들여서 영상 입력이 있다면 전송
    if not status:
        print("영상 입력이 없음")
        break
    #이미지를 전송하기 위해 인코딩 처리.
    res , encodeframe = cv2.imencode('.jpg', frame, encode_param)
    cv2.imshow("video",frame)
    b64_encoded = base64.b64encode(encodeframe)
    #소켓 중계서버로 송신.
    videoSocket.emit('videoIncoming',b64_encoded)
videoSocket.disconnect()

