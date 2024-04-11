import cv2

# Đọc hình ảnh
img = cv2.imread('example.jpg')

# Xác định tọa độ của điểm (100, 200)
x = 100
y = 100

# Vẽ một điểm ở tọa độ (100, 200) với màu đỏ và độ dày 5 pixel
cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

# Hiển thị hình ảnh
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
