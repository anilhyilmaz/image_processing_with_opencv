import cv2
import numpy as np
print("Package imported!")
#img = cv2.imread("mine/lenna.png")
#cv2.imshow("Output",img)  #Image captured
kernel = np.ones((5,5),np.uint8)
kernel1 = np.ones((10,10),np.uint8)

"""cap = cv2.VideoCapture("mine/sheep.mp4")
while True:
    success, img = cap.read()
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break"""     #Video captured and when q clicked it's quited

"""webcam = cv2.VideoCapture(0)
webcam.set(3,640)
webcam.set(4,480)
webcam.set(10,100)
while True:
    success, img = webcam.read()
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break""" #Webcam captured and when q clicked it's quited

"""img = cv2.imread("mine/lenna.png")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img,(7,7),0)
imgCanny = cv2.Canny(img,100,100)
imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)
imgDialation1 = cv2.dilate(imgCanny,kernel1,iterations=1)
imgEroded = cv2.erode(imgDialation,kernel,iterations=1)

cv2.imshow("Erode Image",imgEroded)
cv2.imshow("Dilate Image",imgDialation)
cv2.imshow("Dilate1 Image",imgDialation1)
imgCanny1 = cv2.Canny(img,150,100)
imgCanny2 = cv2.Canny(img,200,100)
cv2.imshow("Canny Image",imgCanny)
cv2.imshow("Canny1 Image",imgCanny1)
cv2.imshow("Canny2 Image",imgCanny2)
cv2.imshow("GaussianBlur Image",imgBlur)
cv2.imshow("Normal Image",img)
cv2.imshow("Gray Image",imgGray)"""  #Simple cv2 functions to change images

"""#RESIZING AND CROPPING
img = cv2.imread("mine/lambo.PNG")
imgResized = cv2.resize(img,(300,450))      #resized
imgCropped = img[:200,:300]                 #Taking crop
cv2.imshow("Cropped img",imgCropped)
print(img.shape)
cv2.imshow("Img",img)
cv2.imshow("Resized Img",imgResized)"""


#SHAPE AND TEXTING
"""img = np.zeros((512,512,3),np.uint8)
#img[200:300,100:300] = 255,0,0   # <--Belirli bir alanı istenen rgb'ye boyar
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3) # img.shape[1] noktasından img.shape[0] noktasına kadar 0,255,0 rgb rengini
# kullanarak 3 kalınlıklı çizgi çeker
cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)  #0,0 noktasından 250,350ye kadarye kadar 0,0,255 rengini
# kullanarak 2 kalınlıgında kare çizer.
cv2.circle(img,(400,50),30,(255,255,0),5) #(400,50) noktasında 30 yarıçaplı 255,255,0 renkli 5 kalıklı daire çizer
cv2.putText(img,"OPENCV",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),2) #300,200 noktasına cv2.FONT_HERSHEY_COMPLEX fontunu kullanarak
#1 yazı boyutunda  0,150,0 rengi ile 2 yazı kalınlıgında yazı yazar
cv2.imshow("My Img",img)
"""





"""cards = cv2.imread("mine/cards.jpg")
cv2.imshow("Cards",cards)
width,height = 250,350
pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutPut = cv2.warpPerspective(cards,matrix,(width,height))
cv2.imshow("imgOutPut",imgOutPut)
print(pts1)
print(pts2)""" #Fotograftan belli bir kesiti almak

"""def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
img = cv2.imread("mine/lenna.png")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img])) #yatak ve dikey şekilde istendigi kadar img'i sıralamak
cv2.imshow("ImgStack",imgStack)"""


"""img = cv2.imread("mine/lenna.png")
imghor = np.hstack((img,img))
imgver = np.vstack((img,img))
cv2.imshow("Vertical",imgver)
cv2.imshow("Horizontal",imghor)"""

"""def empty(a):
    pass
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
path = "mine/lambo.PNG"
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)
    #cv2.imshow("Orijinal lambo",img)
    #cv2.imshow("HSV Lambo",imgHSV)
    #cv2.imshow("Mask",mask)
    #cv2.imshow("Result", imgResult)
    imgStack = stackImages(0.6,([img,imgHSV],[mask,imgResult]))
    cv2.imshow("Final State", imgStack)
    cv2.waitKey(1)"""
"""#SHAPE DETECTİON
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x,y,w,h = cv2.boundingRect(approx)
            if objCor == 3: objectType = "Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio < 1.05: objectType = "Square"
                else: objectType = "Rectangle"
            elif objCor>4: objectType = "Circles"
            else: objectType = "None"
            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)




path = "mine/shapes.png"
img = cv2.imread(path)
imgContour = img.copy()


imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)
getContours(imgCanny)

imgBlank = np.zeros_like(img)
imgStack = stackImages(0.8 ,([img,imgGray,imgBlur],[imgCanny,imgContour,imgBlank]))
cv2.imshow("stack",imgStack)
cv2.waitKey(0)"""


