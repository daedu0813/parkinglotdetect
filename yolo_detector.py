from PIL import Image
import cv2
import numpy as np

class YoloObjectDetector:
    def __init__(self, image):
        self.image = image

    def yolo(self, frame, size, score_threshold, nms_threshold):
        # 클래스 리스트
        classes = ["person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

        # YOLO 네트워크 불러오기
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        # 이미지의 높이, 너비, 채널 받아오기
        height, width, channels = frame.shape

        # 네트워크에 넣기 위한 전처리
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)

        # 전처리된 blob 네트워크에 입력
        net.setInput(blob)

        # 결과 받아오기
        outs = net.forward(output_layers)

        # 각각의 데이터를 저장할 빈 리스트
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.1:
                    # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
        #print(f"boxes: {boxes}")
        #print(f"confidences: {confidences}")

        # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)
        
        # 후보 박스 중 선택된 박스의 인덱스 출력
        print(f"indexes: ", end='')
        #for index in indexes:
        #    print(index, end=' ')
        #print("\n============================== classes ==============================")
        flag = 0

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                class_name = classes[class_ids[i]]
                label = f"{class_name} {confidences[i]:.2f}"
                color = colors[class_ids[i]]

                # 사각형 테두리 그리기 및 텍스트 쓰기
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
                cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
                
                # 탐지된 객체의 정보 출력
                if (class_name == 'bus') or (class_name == 'car') or (class_name == 'truck'):
                    print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")
                    flag = 1
                    return frame, flag

        return frame, flag

    

# 이미지 경로
#img = "C:\\develop\\aps_ai\\test.jpg"

# 이미지 읽어오기
#frame = cv2.imread(img)

# 입력 사이즈 리스트 (Yolo 에서 사용되는 네크워크 입력 이미지 사이즈)
#size_list = [320, 416, 608]

#frame = yolo(frame=frame, size=size_list[1], score_threshold=0.8, nms_threshold=0.5)
#cv2.imshow("Output_Yolo", frame)
#cv2.waitKey(0)
#cv2.destroyAllWindows()