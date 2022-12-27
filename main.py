# Created by Ahmadreza Anaami
import cv2 ;
import numpy as np

print("----------image filter----------")

filepath = input("please enter your file path : ")

print()
print("hint !!!!! ")
print()
print("for blur Filter 'using average' enter → ' 1 ' ")
print("for blur Filter 'Gaussian Blurring' enter → ' 2 ' ")
print("for blur Filter 'destroy noise using median' enter → ' 3 ' ")
print("for blur Filter 'destroy noise using bilateralFilter' enter → ' 4 ' ")
print("for ' hsv_image 5 diffrent ☺ ' Filter  enter → ' 5 ' ")
print("for ' tv_60 ' Filter  enter → ' 6 ' ")
print("for ' emboss ' Filter  enter → ' 7 ' ")
print("for ' duo_tone ' Filter  enter → ' 8 ' ")
print("for ' sepiaimg ' Filter  enter → ' 9 ' ")
print("for ' ROI_selector ' Filter  enter → ' 10 ' ")





typeF = int(input("enter the type of filtering : "))



img = cv2.imread(filepath, 1)
size = (800 ,600)
img = cv2.resize(img  , size)

def blur_average(img):
    out = cv2.blur(img , (3,3))  
    cv2.imshow("real" , img)
    cv2.imshow("filter" , out)
    cv2.waitKey()
    cv2.destroyAllWindows()


def blur_GaussianBlur(img):
    out = cv2.GaussianBlur(img,(5,5),0)  
    cv2.imshow("real" , img)
    cv2.imshow("filter" , out)
    cv2.waitKey()
    cv2.destroyAllWindows()

def blur_medianBlur(img):
    out =  cv2.medianBlur(img,5)  
    cv2.imshow("real" , img)
    cv2.imshow("filter" , out)
    cv2.waitKey()
    cv2.destroyAllWindows()


def blur_bilateralFilter(img):
    out = cv2.bilateralFilter(img,9,50,50)
    cv2.imshow("real" , img)
    cv2.imshow("filter" , out)
    cv2.waitKey()
    cv2.destroyAllWindows()

def blur_bilateralFilter(img):
    size = (600 ,500)
    img = cv2.resize(img  , size)
    hsv_image=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV image',hsv_image)
    cv2.imshow('Hue channel',hsv_image[:,:,0])
    cv2.imshow('saturation channel',hsv_image[:,:,1])
    cv2.imshow('value channel',hsv_image[:,:,2])
    cv2.imshow("real" , img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def tv_60(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('val', 'image', 0, 255 , any)
    cv2.createTrackbar('threshold', 'image', 0, 100 , any)
 
    while True:
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.getTrackbarPos('threshold', 'image')
        val = cv2.getTrackbarPos('val', 'image')
        for i in range(height):
            for j in range(width):
                if np.random.randint(100) <= thresh:
                    if np.random.randint(2) == 0:
                        gray[i, j] = min(gray[i, j] + np.random.randint(0, val+1), 255) # adding noise to image and setting values > 255 to 255. 
                    else:
                        gray[i, j] = max(gray[i, j] - np.random.randint(0, val+1), 0) # subtracting noise to image and setting values < 0 to 0.
 
        cv2.imshow('Original', img)
        cv2.imshow('image', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def kernel_generator(size):
    kernel = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            if i < j:
                kernel[i][j] = -1
            elif i > j:
                kernel[i][j] = 1
    return kernel
 
def emboss(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('size', 'image', 0, 8, any)
    switch = '0 : BL n1 : BR n2 : TR n3 : BR'
    cv2.createTrackbar(switch, 'image', 0, 3, any)
 
    while True:
        size = cv2.getTrackbarPos('size', 'image')
        size += 2 # adding 2 to kernel as it a size of 2 is the minimum required.
        s = cv2.getTrackbarPos(switch, 'image')
        height, width = img.shape[:2]
        y = np.ones((height, width), np.uint8) * 128
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = kernel_generator(size) # generating kernel for bottom left kernel
        kernel = np.rot90(kernel, s) # switching kernel according to direction
        res = cv2.add(cv2.filter2D(gray, -1, kernel), y)
 
        cv2.imshow('Original', img)
        cv2.imshow('image', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def exponential_function(channel, exp):
    table = np.array([min((i**exp), 255) for i in np.arange(0, 256)]).astype("uint8") # generating table for exponential function
    channel = cv2.LUT(channel, table)
    return channel

def duo_tone(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('exponent', 'image', 0, 10, any)
    switch1 = '0 : BLUE n1 : GREEN n2 : RED'
    cv2.createTrackbar(switch1, 'image', 1, 2, any)
    switch2 = '0 : BLUE n1 : GREEN n2 : RED n3 : NONE'
    cv2.createTrackbar(switch2, 'image', 3, 3, any)
    switch3 = '0 : DARK n1 : LIGHT'
    cv2.createTrackbar(switch3, 'image', 0, 1, any)

    while True:
        exp = cv2.getTrackbarPos('exponent', 'image')
        exp = 1 + exp/100 # converting exponent to range 1-2
        s1 = cv2.getTrackbarPos(switch1, 'image')
        s2 = cv2.getTrackbarPos(switch2, 'image')
        s3 = cv2.getTrackbarPos(switch3, 'image')
        res = img.copy()
        for i in range(3):
            if i in (s1, s2): # if channel is present
                res[:, :, i] = exponential_function(res[:, :, i], exp) # increasing the values if channel selected
            else:
                if s3: # for light
                    res[:, :, i] = exponential_function(res[:, :, i], 2 - exp) # reducing value to make the channels light
                else: # for dark
                    res[:, :, i] = 0 # converting the whole channel to 0
        cv2.imshow('Original', img)
        cv2.imshow('image', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def sepia(img):
    res = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # converting to RGB as sepia matrix is for RGB
    res = np.array(res, dtype=np.float64)
    res = cv2.transform(res, np.matrix([[0.393, 0.769, 0.189],
                                        [0.349, 0.686, 0.168],
                                        [0.272, 0.534, 0.131]]))
    res[np.where(res > 255)] = 255 # clipping values greater than 255 to 255
    res = np.array(res, dtype=np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imshow("original", img)
    cv2.imshow("Sepia", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ROI_selector(img):
    roi = cv2.selectROI("cropped pic" , img , False)
    croped_img = img[ int(roi[1]):int(roi[1]+roi[3]) , int(roi[0]) : int(roi[0] + roi[2]) ]
    croped_img = cv2.resize( croped_img , size)
    cv2.imshow("cropped" , croped_img)
    cv2.waitKey()
    cv2.destroyAllWindows();


if(typeF == 1 ):
    blur_average(img)
if(typeF == 2 ):
    blur_GaussianBlur(img)
if(typeF == 3 ):
    blur_medianBlur(img)
if(typeF == 4 ):
    blur_bilateralFilter(img)
if(typeF == 5):
    blur_bilateralFilter(img)
if(typeF == 6):
    tv_60(img)
if(typeF == 7):
    emboss(img)
if(typeF == 8):
    duo_tone(img)
if(typeF == 9):
    sepia(img)
if(typeF == 10):
    ROI_selector(img)



























