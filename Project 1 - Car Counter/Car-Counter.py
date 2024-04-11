import numpy as np
from ultralytics import YOLO
import cv2  # a powerful library for working with images in Python
import cvzone
import math
from sort import *

# Đọc video
cap = cv2.VideoCapture("../Videos/cars.mp4")
# using a pre-trained YOLOv8 model with weights(trọng số) loaded from the file # "yolov8l.pt"
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Each class name represents a category of object that the model is trained to detect (Phát hiện)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Đọc hình ảnh mask
mask = cv2.imread("mask2.png")

# Theo giỏi đối tượng  - Simple Online and Realtime Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# max_age=20: sau 20 khung hình không có đối tượng đang được theo giỏi sẽ xóa khỏi theo giỏi
# min_hits=3: sau 3 lần phát hiện thì mới theo giỏi
# iou_threshold=0.3 : độ tương đồng giữa 2 box tại đối tượng >= 3 để tiếp tục theo giỏi
limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()

    # lưu cái hình ảnh đã được lượt bỏ các phần không quan trọng
    imgRegion = cv2.bitwise_and(img,mask)
    # đọc hình ảnh
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # chồng hình ảnh lên
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    # Tạo mảng không chứa phần tử có 5 cột
    detections = np.empty((0, 5))

    # sử dụng model để nhận dạng hình ảnh
    # stream=True: báo cho model biết là dữ liệu dầu vào là 1 chuỗi dữ liệu liên tục (video)
    results = model(imgRegion, stream=True)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            # Tọa độ của box
            x1, y1, x2, y2 = box.xyxy[0]
            # Ép kiểu về int
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence - Độ chính sát
            conf = math.ceil((box.conf[0] * 100)) / 100
            # id Class Name
            cls = int(box.cls[0])
            # Lấy tên class hiện tại
            currentClass = classNames[cls]

            # check tên class hiện tại xem có phải là class đang cần hay không
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" and conf > 0.3:
                # Vẽ hình chủ nhật xung quanh đối tượng với vị trí bên dưới
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

                # Hiển thị tên class và độ chính xác lên
                # scale: font size, thickness: bold, offser: padding

                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                  scale=2, thickness=3, offset=5)

                # lưu thông tin của box
                currentArray = np.array([x1, y1, x2, y2, conf])
                # nối currentArray vào mãng detections
                detections = np.vstack((detections, currentArray))
    # Lưu kết quả của tối tượng được theo giỏi
    resultsTracker = tracker.update(detections)
    # Dùng cv2 để v đường kẻ
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # theo dõi đối tượng trong sort để lấy id và đếm
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # In ra thông tin của đối tượng đang theo giỏi
        print(result)
        w, h = x2 - x1, y2 - y1

        # Lấy trọng tâm của đối tượng (//: chia lấy phần nguyên)
        cx, cy = x1 + w // 2, y1 + h // 2
        # vẽ điểm trọng tâm
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        # Điểm trọng tâm cx nằm giữa x1, x2, điểm cy nằm dọc tầm 30 thì đếm
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # check if giá trị id có tồn tại chưa nếu chưa thì sẽ tiến hành thêm vào danh sách
            if totalCount.count(id) == 0:
                totalCount.append(id)
                # đổi màu line khi xe được đếm
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
    # show cái total lên img
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # show
    cv2.imshow("Image", img)

    cv2.waitKey(1)