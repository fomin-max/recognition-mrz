import cv2
import first as f
import second as s
import third as t

image = cv2.imread('examples/passport_03.jpg')
ref = cv2.imread('test1.png')

cv2.imshow("check", f.detect_mrz(image))

charNames = ['J', '1', 'S', 'A', '2', 'K', 'T', 'B', '3', 'U', 'L', 'C', '4', 'M', 'V', 'D', 'N', '5', 'W',
             'E', '6', '0', 'X', 'F', '7', 'Y', 'P', 'G', '8', 'Q', 'Z', 'H', '9', 'O', 'R', 'I', '<', '>']

image = f.detect_mrz(image)
zones = s.distribution(image)
cv2.imshow("0", zones[0])
cv2.imshow("1", zones[1])
print(t.recognition(zones[2], ref, charNames))
# print(t.recognition(zones[1], ref, charNames)) ошибка распознования буквы H -> J

# cv2.imshow("2", zones[2])


cv2.waitKey(0)