import cv2
import numpy as np
import pytesseract
# 扫描文件四个顶点的收集程序
# 双击进行收集
#opencv鼠标事件

#points_collect鼠标响应函数
def points_collect(event,x,y,flags,param):
    dic_points = param    
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if len(dic_points['ps'])>=4:
            dic_points['ps']=[]  
            dic_points['ps'].append((x,y))
        else:
            dic_points['ps'].append((x,y))
    if event == cv2.EVENT_MOUSEMOVE:
        dic_points['p_move']=(x,y)  

def drawlines(img,dic_points):
    color = (0,255,0) 
    points = dic_points['ps'][:]
    points.append(dic_points['p_move'])
    
    if len(points)>0 and len(points)<5: 
        for i in range(len(points)-1): 
            cv2.circle(img,points[i],15,color,cv2.FILLED)
            cv2.line(img,points[i],points[i+1],color,6)
    elif len(points)>=5:
        for i in range(3):
            cv2.circle(img,points[i],15,color,cv2.FILLED)     
            cv2.line(img,points[i],points[i+1],color,6) 
        cv2.circle(img,points[3],15,color,cv2.FILLED)
        cv2.line(img,points[3],points[0],color,6) 


def reorder(points):
    points = np.array(points)
    ordered_points = np.zeros([4,2])
    # 将横纵坐标相加，
    # 最小为左上角，最大为右下角 
    add = np.sum(points,axis=1) 
    ordered_points[0] = points[np.argmin(add)]
    ordered_points[3] = points[np.argmax(add)]
    
    # 将横纵坐标相减 diff 为后减前 即 y-x
    # 最小为右上角，最大为左下角
    diff = np.diff(points,axis=1)
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[2] = points[np.argmax(diff)]
    return ordered_points

# 实现图像的仿射变换-图像校正
# ordered_points ： 需要变换的4个顶点
# size_wraped: 变换后 图像的大小 （w,h）    
def getWarp(img,ordered_points,size_wraped):
    
    w,h = size_wraped
    p1 = np.float32(ordered_points)
    # 目标图像坐标点
    p2 = np.float32([[0,0],[w,0],[0,h],[w,h]])#float32单精度浮点
    #要构建这个变换矩阵，你需要在输入图像上找 4 个点，以及他们在输出图像上对应的位置。
    # 这四个点中的任意三个都不能共线。
    # 这个变换矩阵可以由cv2.getPerspectiveTransform( ) 函数构建。然后把这个矩阵传给函数cv2.warpPerspective（）。
    # 计算仿射矩阵
    matrix = cv2.getPerspectiveTransform(p1, p2)
    
    # 进行仿射变换（透视变换：perspective）
    imgOutput = cv2.warpPerspective(img, matrix, (w, h))
    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(w,h))
    return imgCropped
    

if __name__ == "__main__":
    file_scan = "C:\\Users\\Forward\\Desktop\\python-ai-master\\Scanner\\shiyan4.jpg"
    size_wraped = (960,1020)
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    dic_points = {}
    dic_points["ps"]=[]
    dic_points["p_move"]=()
    cv2.namedWindow('image',cv2.WINDOW_NORMAL) 
    cv2.setMouseCallback('image', points_collect,param=dic_points)
    
    while True:
        img = cv2.imread(file_scan)
        drawlines(img,dic_points)
        cv2.imshow('image',img)
        key=cv2.waitKey(100)# & 0xFF  
        
        if key == ord('q'):
            break 
        if key == ord('w'):
            key = 0
            if len(dic_points['ps'])==4:
                ordered_points = reorder(dic_points['ps'])
                img_Warped = getWarp(img,ordered_points,size_wraped)
                cv2.imshow("ImageWarped",img_Warped)
                
                # 颜色转换，先把BGR的图像转换为RGB
                #需要注意的是，OpenCV中图像矩阵的顺序是 B,G,R。OpenCV 将颜色读取为 BGR（蓝绿色红色），但大多数计算机应用程序读取为 RGB（红绿蓝）
                imgWarped_RGB = cv2.cvtColor(img_Warped, cv2.COLOR_BGR2RGB)
                
                # 文字识别
                txt = pytesseract.image_to_string(imgWarped_RGB)
                print(txt)
                
                text_path='C:\\Users\\Forward\\Desktop\\python-ai-master\\Scanner\\ScannerTxt'+'.txt'
                #打开文件，在编写文件时设置编码格式，否则出现字符时会报错
                file_handle=open(text_path,mode='w',encoding= 'utf8')
                #将识别后的文本text写入指定文件
                file_handle.write(txt)
                #写完之后关闭文件
                file_handle.close()
                #创建列表result，用来存储生成文本文件的路径，文本文件内容
                result=[text_path]
                result.append(txt)
                
                
            
    