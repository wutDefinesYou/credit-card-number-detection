import argparse
from imutils import contours
import numpy as np
import cv2
import myutils

# 设置参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
# args = vars(ap.parse_args())
TEMPLATE_PATH = "./img/ocr_a_reference.png"
IMAGE_PATH = "./img/credit_card_05.png"

# 指定信用卡类型
FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}

# 绘图函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyWindow("img")
    cv2.waitKey(1)

# 读取一个模板图像
# template = cv2.imread(args["template"])
template = cv2.imread(TEMPLATE_PATH)
# cv_show("template", template)

# 灰度图
ref = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 二值图
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
# cv_show("binary", ref)

# 计算模板轮廓
# cv2.findContours()函数接受的参数为二值图，cv2.RETR_EXTERNAL只检测轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标

contours, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(ref.copy(), contours, -1, (0, 0, 255), 2)
# cv_show("template", template)
# print(f"shape of the contours: {len(contours)}")
contours = myutils.sort_contours(contours, method="left-to-right")[0]

digits = {}

# 遍历每一个轮廓
for index, contour in enumerate(contours):
    # 计算外接矩形并且重新缩放大小
    (x, y, w, h) = cv2.boundingRect(contour)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))

    # 字典：每个数字对应其模板
    digits[index] = roi

# cv_show("roi", np.hstack(tuple(digits.values())))

# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像，预处理
image = cv2.imread(IMAGE_PATH)
# cv_show("card", image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show("card gray", gray)

# 礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
# cv_show("card tophat", tophat)

# Sobel算子检测水平方向边缘，并进行归一化
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
gradX = gradX.astype("uint8")

# print(gradX.shape)
# cv_show("Sobel", gradX)

# 通过闭操作（先膨胀，再腐蚀）将数字连在一起
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# cv_show("morph close", gradX)

# THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为零
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show("threshold", thresh)

# 再进行一次闭操作将数字区域的空隙填充
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=sqKernel, iterations=1)
# cv_show("threshold closed", thresh)

# 计算轮廓
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cur_img = image.copy()
cv2.drawContours(cur_img, contours, -1, (0, 0, 255), 2)
# cv_show("card contours", cur_img)

# 筛选轮廓
locs = []
for index, contour in enumerate(contours):
    (x, y, w, h) = cv2.boundingRect(contour)
    ar = w / float(h)

    if h > 2.5 and ar < 4.0:
        if (w > 38 and w < 52) and (h > 8 and h < 22):
            locs.append((x, y, w, h))

# 从左至右四组轮廓
locs = sorted(locs, key=lambda x: x[0])

output = []

for index, (gX, gY, gW, gH) in enumerate(locs):
    groupOutput = []

    # 适当扩大每组轮廓范围五个单位
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # cv_show("group", group)

    # 转二值
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv_show("binary group", group)

    # 轮廓检测
    contours, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitContours = myutils.sort_contours(contours, method="left-to-right")[0]

    # 对于每一组轮廓中的每一个数的轮廓
    for cnt in digitContours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        scores = []

        # 每一个数字与每一个数字模板进行匹配
        for digit, digitROI in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            _, maxVal, _, _ = cv2.minMaxLoc(result)
            scores.append(maxVal)

        # 最相近的数字
        groupOutput.append(str(np.argmax(scores)))

    # 原输入图像上画出结果
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    output.extend(groupOutput)

print(" ".join(["".join(output[i:i + 4]) for i in range(0, len(output), 4)]))
cv_show("result", image)