from io import BytesIO
from urllib import request
from PIL import Image
import cv2 as open_cv
import yaml
import socket
import pandas as pd
import numpy as np
import time
from datetime import datetime

from motion_detector import MotionDetector
from coordinates_generator import CoordinatesGenerator
from carnumber_detector import CarNumberDetector
from yolo_detector import YoloObjectDetector
from colors import *
from stream import *

host = '192.168.0.2'    # 호스트 IP 설정
port = 9999             # 포트번호 설정
video_file = open_cv.VideoCapture(VIDEO)
data_file = 'data/data.yml'

with open(data_file, "r") as data:
    points = yaml.safe_load(data)
df_data = pd.DataFrame(points)
df_data_index = list(df_data.index)

def main():
    server_sock = socket.socket(socket.AF_INET)
    server_sock.bind((host, port))
    while True:
        print("주차 공간 설정을 하시겠습니까? Y: Yes, N: No")
        key = input()
        if key == 'y':
            set_space()                         # 주차 공간 설정
            break
        else:
            break

    print("서버 연결 대기 중")
    server_sock.listen(5)
    client_sock, addr = server_sock.accept()

    print('Connected by', addr)
    data = client_sock.recv(1024)
    print(data.decode("utf-8"), len(data))

    while True:
        print("▶aParkings-AI◀\n　시작: S\n　종료: Q")
        key = input()
        if key == 's':
            np_flag, np_carnum = detect()                       # 차 여부 확인, 차번호 리턴
            df = make_dataframe()                               # 데이터 프레임 만들어서 리턴
            np_entered_time, np_elapsed_time = timer(np_flag)   # 들어간 시각, 진행된 시간 리턴
            np_fee = calc_fee(np_elapsed_time)                  # 진행된 시간으로 요금 계산해 리턴
            df = insert_dataframe(df, np_flag, np_carnum, np_entered_time, np_elapsed_time, np_fee)
            print(df)
            upload_to_app(df, client_sock)                      # 안드로이드 앱에 현재 값 업로드
            data = server_sock.recv(1024)
            print(data.decode(), len(data))
            #show_parkinglot(points)
        elif key == 'q':
            #close_opencv()
            close_sever(server_sock, client_sock)
            quit()

def upload_to_app(df, client_sock):
    # 안드로이드 앱 통신
    #df.to_json('test.json', orient='index')
    df_json = df.to_json(orient='index')
    client_sock.send(df_json.encode())

def close_sever(server_sock, client_sock):
    client_sock.close()
    server_sock.close()
    print("서버 접속 해제")

def close_opencv():
    open_cv.waitKey(0)
    open_cv.destroyAllWindows()
    print("OpenCV 종료")

def make_dataframe():
    index = df_data.index
    columns = ['available', 'car_number', 'entered_time', 'elapsed_time', 'fee']
    df = pd.DataFrame(index=index, columns=columns)
    
    return df

def insert_dataframe(df, np_flag, np_carnum, np_entered_time, np_elapsed_time, np_fee):
    for p in df_data_index:
        if np_flag[p] == 1:
            df.iloc[p, 0] = "불가능"
        else:
            df.iloc[p, 0] = "가능"
        df.iloc[p, 1] = np_carnum[p]
        df.iloc[p, 2] = np_entered_time[p]
        df.iloc[p, 3] = np_elapsed_time[p]
        df.iloc[p, 4] = np_fee[p]
    return df

def timer(np_flag):
    np_entered_time = np.array([])
    np_elapsed_time = np.array([])

    now = time.localtime()

    time_now = str(now.tm_year) + '-' + str(now.tm_mon) + '-' + str(now.tm_mday) + ' ' + str(now.tm_hour) + ':' + str(now.tm_min) + ':' + str(now.tm_sec)
    #entered_time = datetime.strptime('2021-06-23 14:30:01', '%Y-%m-%d %H:%M:%S')
    now_time = datetime.strptime(str(time_now), '%Y-%m-%d %H:%M:%S')

    for p in df_data_index:
        if np_flag[p] == 1:
            np_entered_time = np.append(np_entered_time, datetime.strptime(str(time_now), '%Y-%m-%d %H:%M:%S'))
            np_elapsed_time = np.append(np_elapsed_time, now_time - np_entered_time[p])
        else:
            np_entered_time = np.append(np_entered_time, 0).astype(np.int32)
            np_elapsed_time = np.append(np_elapsed_time, 0).astype(np.int32)

    return np_entered_time, np_elapsed_time

def calc_fee(np_elapsed_time):
    # 예시
    # 최초 1시간 1000원
    # 이후 10분당 500원
    now = time.localtime()
    np_fee = np.array([])
    fee = 1000
    for p in df_data_index:
        if np_elapsed_time[p] != 0:
            elapsed_time = datetime.strptime(str(str(now.tm_year) + '-' + str(now.tm_mon) + '-' + str(now.tm_mday) + ' ' + str(now.tm_hour) + ':' + str(now.tm_min) + ':' + str(now.tm_sec)), '%Y-%m-%d %H:%M:%S') - datetime.strptime('2021-07-02 11:30:01', '%Y-%m-%d %H:%M:%S') #str(np_elapsed_time[p])
            time_to_seconds = int(elapsed_time.total_seconds())
            if time_to_seconds > 3600:
                    time_to_seconds = time_to_seconds - 3600
                    tenmin = (time_to_seconds + 600) // 600
                    fee = fee + tenmin * 500
            np_fee = np.append(np_fee, fee)
        else:            
            np_fee = np.append(np_fee, 0).astype(np.int32)            
    
    return np_fee

def set_space():
    image_file = 'Setting Parking Space'

    if image_file is not None:
        with open(data_file, "w+") as points:
            generator = CoordinatesGenerator(image_file, points, COLOR_RED)
            generator.generate()

def detect():
    # 스트리밍 이미지 to RGB
    url = IMAGE
    res = request.urlopen(url).read()
    img = Image.open(BytesIO(res)).convert('RGB')
    img = np.array(img)
    img = img[:, :, ::-1].copy()

    np_flag = np.array([])
    np_carnum = np.array([])
    np_croped = np.array([])

    # yaml 파일 분석 후 임의 지정 주차 공간 numpy로 변환
    for p in df_data.index:
        for i in range(0, 4):
            for j in range(0, 2):
                np_croped = np.append(np_croped, df_data['coordinates'][p][i][j])

    np_croped = np.reshape(np_croped, (len(df_data_index), 4, 2))
    np_croped = np_croped.astype(np.int32)

    for p in df_data_index:
        # 임의 지정 주차 공간 크롭 후 차번호 인식
        rect = open_cv.boundingRect(np_croped[p])
        x, y, w, h = rect
        croped = img[y:y+h, x:x+w].copy()
        open_cv.imwrite("croped_" + str(p) + ".jpg", croped)
        #open_cv.imshow("croped_" + str(p), croped)
        
        carnumdetector = CarNumberDetector("croped_" + str(p) + ".jpg")
        chars = carnumdetector.detect()

        # 차번호를 가진 numpy
        np_carnum = np.append(np_carnum, chars)

        # 차 여부 확인
        yolo_img = open_cv.imread("croped_" + str(p) + ".jpg")
        yolodetector = YoloObjectDetector("croped_" + str(p) + ".jpg")
        yolo_img, flag = yolodetector.yolo(yolo_img, 416, 0.9, 0.5)

        np_flag = np.append(np_flag, flag)

        #open_cv.imshow("croped_" + str(p) + "_YOLO-416-0.9-0.5", yolo_img)
        #open_cv.imwrite("croped_" + str(p) + "_YOLO-416-0.9-0.5.jpg", yolo_img)

    return np_flag, np_carnum

def show_parkinglot(points):
    with open(data_file, "r"):
        detector = MotionDetector(video_file, points, 1)
        detector.detect_motion()

if __name__ == '__main__':
    main()