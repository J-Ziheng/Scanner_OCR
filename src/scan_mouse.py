import cv2
import numpy as np
import pytesseract

# 扫描文件四个顶点的收集程序
# 双击进行收集
#opencv鼠标事件

#points_collect鼠标响应函数
def points_collect(event,x,y,flags,param): #x,y鼠标xy坐标，flags鼠标状态，可以用参数对应值代替
    dic_points = param    #param是用户定义的传递到setMouseCallback函数的参数，作为setMouseCallback的参数
    if event == cv2.EVENT_LBUTTONDBLCLK:#触发事件为左键双击，左键按下，允许绘图，以及获取绘图坐标
        if len(dic_points['ps'])>=4:#如果大于等于4，[]为空，重新记录四个点的坐标位置
            dic_points['ps']=[]  #[]列表  {}字典
            dic_points['ps'].append((x,y))#给dic_points['ps']列表添加数据，数据即获取的绘图坐标
        else:
            dic_points['ps'].append((x,y))
            
    if event == cv2.EVENT_MOUSEMOVE:#触发事件为鼠标移动时
        dic_points['p_move']=(x,y)  #定义一个列表，在鼠标移动时获取坐标值，也就是说这里存放的是动点当前位置的坐标

def drawlines(img,dic_points):
    color = (0,255,0) #red和blue为0 green为255 为绿色
    
    # 已记录顶点复制，将鼠标标记好的四个顶点的坐标取出来,[:]代表截取列表，如[1:3]，截取下标为1--3元素，单纯‘：’就是全部提取
    points = dic_points['ps'][:]
    #追加移动动点,把鼠标移动的点坐标也加进来，append函数数组在末端添加元素
    points.append(dic_points['p_move'])
    
    if len(points)>0 and len(points)<5: #四个顶点+动点最多五个,if语句意思是鼠标连接完之前都要做什么
        for i in range(len(points)-1): #就是单纯地列表减1，当画到第三个点时，实际为四个，为了保证输出值为0 1 2 ，需要-1；注意下边是line语句是[i]和[i+1]
            cv2.circle(img,points[i],4,color,cv2.FILLED)#在img上画圆，points坐标，半径为4 颜色是0 255 0绿色，filled画一个实心的
            cv2.line(img,points[i],points[i+1],color,4)#将当前点和下一个点连接
            #这一个for循环是和动点连接的一个过程，在最后一个点出现前，保持和动点即鼠标不断移动的点连接
            
    elif len(points)>=5:
        #大于等于5就是，最后一个即第四个坐标点画完，带上动点为5个，已经完毕
        #接下来执行for[i]循环连接线条就行
        for i in range(3): #3是结束值，[0,1,2]
            cv2.circle(img,points[i],8,color,cv2.FILLED)     #img 后续要导入进来的图片的参数
            cv2.line(img,points[i],points[i+1],color,4)  #circle函数画圆，img为源图像，points[i]为画圆的圆心坐标，4为圆的半径，clolor为设定好的圆的颜色（python为BGR）
            #points[i]是当前点 points[i+1]是下一个点，两点连接，依次往下连接0 1 2 这三个点
            
        cv2.circle(img,points[3],8,color,cv2.FILLED)#point[3]为第四个点
        cv2.line(img,points[3],points[0],color,4) #这一句是吧0（起始）顶点和3（结束）顶点连接

# 将收集的四个顶点按照[左上，右上，左下，右下]，opencv的要求
# 顺序进行重新排列        
def reorder(points):
    points = np.array(points) #np.array()生成数组，二维数组对象的集合
    ordered_points = np.zeros([4,2]) #numpy.zeros 创建指定大小的数组，数组元素以0来填充
    #创建一个以o做填充的二维[4,2]数组（或者说4行2列的矩阵）oederd_ppoints
    
    # 将横纵坐标相加，
    # 最小为左上角，最大为右下角 
    add = np.sum(points,axis=1) #axis=1表示第二维度内元素间求和，第二维度就是如：[2,3]，[3,4]这样的内部求和，即2+2,3+4，得到的是[5,7]
    ordered_points[0] = points[np.argmin(add)]#argmin索引，根据求和得到的最大或最小值索引对应的数组（即坐标），并将数组值提供给等号左边的创建好的数组，放在下标为0的位置
    ordered_points[3] = points[np.argmax(add)]
    
    # 将横纵坐标相减 diff 为后减前 即 y-x
    # 最小为右上角，最大为左下角
    diff = np.diff(points,axis=1)
    ordered_points[1] = points[np.argmin(diff)]
    ordered_points[2] = points[np.argmax(diff)]
    #return传递并保存计算好的值
    return ordered_points

# 实现图像的仿射变换-图像校正
# ordered_points ： 需要变换的4个顶点
# size_wraped: 变换后 图像的大小 （w,h）    
def getWarp(img,ordered_points,size_wraped):
    w,h = size_wraped#定义变换后图像的宽度和高度
    
    # 源图像坐标点
    p1 = np.float32(ordered_points)
    
    # 目标图像坐标点
    p2 = np.float32([[0,0],[w,0],[0,h],[w,h]])#float32单精度浮点
    
    #要构建这个变换矩阵，你需要在输入图像上找 4 个点，以及他们在输出图像上对应的位置。
    # 这四个点中的任意三个都不能共线。
    # 这个变换矩阵可以由cv2.getPerspectiveTransform( ) 函数构建。然后把这个矩阵传给函数cv2.warpPerspective（）。
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(p1, p2)
    
    # 进行仿射变换（透视变换：perspective）
    imgOutput = cv2.warpPerspective(img, matrix, (w, h))
    #img原图，martix一个变换矩阵，（w,h)输出图像的尺寸大小width和hight
    
    # 对边界进行简单裁剪    crop：剪裁图片
    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]#python数组分片，取[y~y+h,x~x+w]这个下标范围（像素范围），构成一个新数组
    #shape[0]表示图片高度  shape[1]表示图片宽度  shape[2]表示图像的通道数
    imgCropped = cv2.resize(imgCropped,(w,h))
    #cv2.resize图像缩放函数，对应参数是（输入图像，x轴缩放系数，y轴缩放系数）
    return imgCropped
    


#在_main_部分里，实现回调函数       
if __name__ == "__main__":#python模拟的程序入口，程序运行后这句话之下的代码块运行
    # 需要扫描的文件
    file_scan = "C:\\Users\\Forward\\Desktop\\python-ai-master\\Scanner\\safe2.jpg"
    # 定义扫描后文件的大小（420*600大约是实际a4纸的比例*2  a4:297*210mm）
    #size_wraped = (960,1020)
    size_wraped = (420,600)
    # 定义tesseract的位置
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    # 记录坐标的字典dictionary
    dic_points = {}
    dic_points["ps"]=[]
    dic_points["p_move"]=()
    
    # 设置回调函数
    cv2.namedWindow('image',cv2.WINDOW_NORMAL) #创建一个不可改变大小的窗体（cv2.winwodnormal可改变大小）#,cv2.WINDOW_NORMAL
    #鼠标事件的实现函数（一个完整的鼠标事件由一个自定义的鼠标回调函数+实现鼠标回调函数的设置
    cv2.setMouseCallback('image', points_collect,param=dic_points)#实现鼠标回调函数到img的窗体
    #第一个参数image为窗体名称 指的是哪个窗体下执行的
    #第二个参数是points_collect鼠标响应函数，为鼠标回调函数的名称-传入函数名称，指的是传入整个函数声明，并非执行
    #param：响应函数传递的参数
    
    

    while True:
        img = cv2.imread(file_scan)
        
        drawlines(img,dic_points)#划线的操作
        
        cv2.imshow('image',img)
        
        #cv2.waitkey(millseconds) millseconds毫秒,表示等待millseconds毫秒后自动销毁
        key=cv2.waitKey(100) & 0xFF  #读取键值 
        
        if key == ord('q'):
            break
            
        if key == ord('w'):
            key = 0
            if len(dic_points['ps'])==4:
                
                # 图像仿射变换
                ordered_points = reorder(dic_points['ps'])
                img_Warped = getWarp(img,ordered_points,size_wraped)
                cv2.imshow("ImageWarped",img_Warped)
                
                # 颜色转换，先把BGR的图像转换为RGB
                #需要注意的是，OpenCV中图像矩阵的顺序是 B,G,R。OpenCV 将颜色读取为 BGR（蓝绿色红色），但大多数计算机应用程序读取为 RGB（红绿蓝）
                imgWarped_RGB = cv2.cvtColor(img_Warped, cv2.COLOR_BGR2RGB)
                
                # 文字识别
                txt = pytesseract.image_to_string(imgWarped_RGB,lang='chi_sim')
                
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
    