# Scanner_OCR 人工智能程序设计
## 一、开发说明
这是一款简单的图像文字识别程序——使用Python语言，基于**OpenCV**和**Tesseract-OCR**，实现了目标图像导入、手动截取、自动矫正和最终进行文字识别与输出这一系列功能。
>该程序是人工智能课程的个人结课设计，程序全部皆自主学习开发。
>
>于2021年末历时约30天开发与设计完成。  
>
>学习新领域知识对实际应用设计与开发的过程稍有曲折，程序总体虽然不大，但整体设计思路均已逐一实现。

#### 开发者：
&emsp;&emsp;&emsp;靳子恒 北科信息工程系计算机科学与技术

#### 开发技术：
&emsp;&emsp;&emsp;语言：python  

&emsp;&emsp;&emsp;工具：VsCode  

&emsp;&emsp;&emsp;环境：Windows + Python3.7 + OpenCV库 + numpy库 + Tesseract5.0


## 二、开发过程
### 1、设计思路
>如图所示:  
>- 首先通过 OpenCV 的 imread 和 imshow 函数读取图片并创造一个 GUI 窗口进行图像展示;  
>
>- 其次利用鼠标回调函数和鼠标事件（鼠标左键双击）实现鼠标双击绘图截选想要识别的文档部分;  
>
>- 再利用 OpenCV 的透视变换里的矩阵运算getPerspectivetransform 函数和 warpPerspective 函数共同完成图像的矩阵运算与转换; 
>
>- 与此同时对图像边界进行裁剪，以达到在图像截取过程中将不得已截取到的文档以外部分裁剪的目的;  
>
>- 最后使用谷歌的 Tesseract-OCR 引擎进行文字识别，与此同时，对识别内容进行保存，设定好存储路径，保存为 txt 文件。  

![设计思路](https://user-images.githubusercontent.com/92208322/221533403-30e9c6f3-5afb-4116-9cc4-885f7cf77a40.png)

### 2、实现过程
**1. 定义鼠标响应函数：收集源图像的四个顶点坐标点**
```python
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

```
>解析：
>
>&emsp;&emsp;*定义一个鼠标回调函数，利用鼠标响应进行图像截取这一事件的触发。图像截取为调用 OpenCV 绘图函数在下一部分，此处是定义 event 事件为鼠标左键双击 LBUTTONDBLCLK 事件。*
>
>&emsp;&emsp;*当鼠标左键双击后，开始描点绘图截取图像矫正部分并进行鼠标坐标点的记录。因图像截取为四个坐标点，截取梯形图，因此定义名为dic_points[]的列表变量,在此函数部分对鼠标双击事件开始后鼠标坐标点的收集。*
>
>&emsp;&emsp;*在绘图过程中，考虑到鼠标点第一个点之后需要显示连线，因此鼠标在移动过程中的坐标也要纳入 dic_points 列表内，为下一步点与点之间连线做好准备，移动的坐标即动点，利用 append 函数将move 动点坐标附加到 dic_points 列表内。*
>
>&emsp;&emsp;*其中，函数参数 param 作用是将定义好的鼠标坐标列表dic_points 赋给 param，在主函数 setmouseback 部分进行参数回调。*


**2. 确认选点并绘图**
```python
def drawlines(img,dic_points):
    color = (0,255,0) #red和blue为0 green为255 为绿色
    
    # 已记录顶点复制，将鼠标标记好的四个顶点的坐标取出来,[:]代表截取列表，如[1:3]，截取下标为1--3元素，单纯‘：’就是全部提取
    points = dic_points['ps'][:]
    #追加移动动点,把鼠标移动的点坐标也加进来，append函数数组在末端添加元素
    points.append(dic_points['p_move'])
    
    if len(points)>0 and len(points)<5: #四个顶点+动点最多五个,if语句意思是鼠标连接完之前都要做什么
        for i in range(len(points)-1): #就是单纯地列表减1，当画到第三个点时，实际为四个，为了保证输出值为0 1 2 ，需要-1；注意下边是line语句是[i]和[i+1]
            cv2.circle(img,points[i],4,color,cv2.FILLED)#在img上画圆，points坐标，半径为4 颜色是0 255 0绿色，filled画一个实心的
            cv2.line(img,points[i],points[i+1],color,1)#将当前点和下一个点连接
            #这一个for循环是和动点连接的一个过程，在最后一个点出现前，保持和动点即鼠标不断移动的点连接
            
    elif len(points)>=5:
        #大于等于5就是，最后一个即第四个坐标点画完，带上动点为5个，已经完毕
        #接下来执行for[i]循环连接线条就行
        for i in range(3): #3是结束值，[0,1,2]
            cv2.circle(img,points[i],4,color,cv2.FILLED)     #img 后续要导入进来的图片的参数
            cv2.line(img,points[i],points[i+1],color,1)  #circle函数画圆，img为源图像，points[i]为画圆的圆心坐标，4为圆的半径，clolor为设定好的圆的颜色（python为BGR）
            #points[i]是当前点 points[i+1]是下一个点，两点连接，依次往下连接0 1 2 这三个点
            
        cv2.circle(img,points[3],4,color,cv2.FILLED)#point[3]为第四个点
        cv2.line(img,points[3],points[0],color,1) #这一句是吧0（起始）顶点和3（结束）顶点连接

```
>解析：
>
>&emsp;&emsp;*此函数就是上一步中所述的 OpenCV 绘图函数，实现过程中起到描点连线，框选想要截选图像部分的功能。该功能是在第一步鼠标响应函数的基础上，当鼠标左键双击，触发事件，开始绘图并记录坐标。*
>
>如图所示：
>
>![人工智能基础程序设计报告 pdf_和另外_5_个页面_-_个人_-_Microsoft​_Edgemsedge](https://user-images.githubusercontent.com/92208322/221545172-17995cd4-b6be-4a44-8a93-41921bef5447.png)

>
>&emsp;&emsp;*OpenCV 描点连线颜色 color 设置为（0,255,0）纯绿色。Circle函数画圆，然后用 Filled 函数进行填充，已达到画点的目的。Line函数用于画线，画线函数内利用 point(i)和 point(i+1)函数实现当前点与下一个点的连接，当然此处也间接解释了为什么第一步鼠标回调里要附加（append）鼠标动点坐标到 dic_points[]列表里。*
>
>&emsp;&emsp;*在函数 if-elif 循环部分是考虑到以下情况设置 0-5 区间范围和>=5 的判定值：1.除去鼠标四个固定顶点外还要考虑动点，动点算作第五个坐标点，在鼠标截选图像至要截选最后一个时已经有了四个点（3 顶点 1 动点），此时执行的是 if 语句内的代码；当画第四个
顶点的时候，实际有 5 个点，这时候执行 elif 语句，将第四个点和初始点之间建立连接。2.当描完四个顶点并连线后，如果对框选部分不满意，那么再次双击，此时和上一部鼠标响应函数有关，动点+四个顶点+额外双击时，坐标数已经大于 4，那么执行清空数组语句dic_points[‘ps’]=[]，重新开始绘图。*
>

**3. 将绘图中收集到的四个顶点进行排序**
```python
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

```
>解析：
>
>&emsp;&emsp;*此处对收集好的四个顶点坐标排序是为下一步透视变化做好准备，之所以要坐标排序是因为，我们在截选图像的时候顺序不一样，可能逆时针可能顺时针，可能从左下开始描点，可能从右上开始描点等等，计算机并不知道这些坐标的顺序，没有个很正的图像的标准，也就无法对截取的图像进行矫正。因此要对坐标进行一个固定的位置排序，排序原理如下图所示：*
>
>如图所示：
>![人工智能基础程序设计报告 pdf_和另外_5_个页面_-_个人_-_Microsoft​_Edgemsedge](https://user-images.githubusercontent.com/92208322/221546021-521cfea5-66ac-43e9-b6b1-ddb00ad6a83e.png)
>
>&emsp;&emsp;*由此图应该更好理解，坐标排序的依据。定义了两个算法：add 和 diff，前者是计算二维数组（存放好的四个顶点坐标）内x+y 的值，后者是计算二维数组内 x-y 的值。计算好后，利用argmin 和 argmax 函数对计算出的最大最小值的 x y 进行索引。*
>
>&emsp;&emsp;*左上角 X+Y 一定是最小的，argmin（add）将索引的那个坐标重新赋给新的数组 orded_point[]里，并放在下标为 0 的位置，以此类推，其他不再赘述。*
>
>&emsp;&emsp;*其中 zero（[4,2]）是创建一个用 0 填充的二维数组，即：[[0,0],[0,0],[0,0],[0,0]],为上述坐标排序做前提准备。*


**4. 使用透视变换进行图像矫正,并对图像边缘裁剪**

```python
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
    # 计算仿射矩阵
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

```
>解析：
>
>&emsp;&emsp;*此处是图像校正函数 getwarp。在前三步的基础上，获得源坐标点和目标坐标点，再利用 getPersepectiveTransgorm 函数进行透视矩阵的运算，运算完成后，图像用计算好的结果进行矫正，即 warpPerspective 函数。*
>
>&emsp;&emsp;*此时图像已经矫正完成，再对图像边界进行裁剪，以达到在图像截取过程中不得已截取到文档以外部分的裁剪的目的。用到的函数：imgOutput[20:x-20,20:y-20],他表示对图像截取 x 轴上从第 20像素开始到 x-20 像素的那一部分，y 轴同理，以此达到图像边界各裁剪 20 像素的目的。*
>
>&emsp;&emsp;*在原代码中还有 shape[0]和 shape[1]函数，前者表示获取图像的高度，后者表示图像的宽度，如果参数是 2 表示图像的通道数。最后 resize 函数将图像矫正后的最终成品赋给变量 imgCropped，供主函数调用。*


**5. 主函数部分：调用前四步定义好的函数，OpenCV 进行图像路径读取、实现文字识别与识别内容的存储**
```python
#在_main_部分里，实现回调函数       
if __name__ == "__main__":#python模拟的程序入口，程序运行后这句话之下的代码块运行
    # 需要扫描的文件路径
    file_scan = "C:\\Users\\Forward\\Desktop\\Scanner\\camera2.jpg"
    # 定义扫描后文件的大小（420*600大约是实际a4纸的比例*2  a4:297*210mm）
    size_wraped = (420,600)
    # 定义tesseract的位置
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    # 记录坐标的字典dictionary
    dic_points = {}
    dic_points["ps"]=[]
    dic_points["p_move"]=()
    
    # 设置回调函数
    cv2.namedWindow('image',cv2.WINDOW_NORMAL) #创建一个不可改变大小的窗体（cv2.winwodnormal可改变大小）
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
                txt = pytesseract.image_to_string(imgWarped_RGB, lang='chi_sim')
                print(txt)

```
>解析：
>
>&emsp;&emsp;*此处是主函数部分，size_wrapped 是设置第四部我们想要设置的矫正后的图像的宽高比。第二句是导入谷歌的 tesseract语言识别库，是谷歌已经训练好的语言识别包，正因是谷歌的 ocr，因此对中文的识别率并不高。在下边用pytesseract.imag_to_string 语句实现为图像的文字识别。*
>
>&emsp;&emsp;*其中 waitkey 函数是设置 OpenCV 展示图像函数执行后展示图像的 GUI 的滞留时间，单位是 ms。但后又加了一句& 0xFF，它的作用是不以时间为 GUI 窗口滞留的标准，而是以按键来判断，下边接着，如果是 q 按键则 break 关闭窗口，如果是 w 实现文字识别。*
>
>&emsp;&emsp;*最后一部分就是创建一个保存识别内容的路径，文字识别后将识别内容进行一个存储，存储格式为 txt 文件，命名为 ScannerTxt。*
>
>&emsp;&emsp;*至此，程序语句全部执行完毕！*
>

## 三、相关算法和理解

诸如拉伸、收缩、扭曲、旋转是图像的几何变换，在三维视觉技术中大量应用到这些变换，又分为仿射变换和透视变换。

**我的部分理解：**
因为图像的变化主要涉及到像素坐标的处理，在图像的变化开始时，获取坐标准备矩阵运算，在图像的变化的过程中，像素坐标传输到矩阵运算算法里，不断地计算和定位，最终呈现出图像拉伸、扭曲等效果。 当然，图像的几何变换实际情况更复杂，应用场景更多，涉及到的算法也更多，难度也更大，需求也更复杂！

这两个算法通常一起配合使用，OpenCV库里提供了函数，但依旧需要了解原理，才能更好的使用函数，因为涉及到像素坐标的选择、处理等。

### 透视变换算法
>
>&emsp;&emsp;透视变换（Perspective Transformation）就是将二维的图片投影至一个三维的视平面上，然后再转换到二维坐标下，所以也称为投影映射（Projective Mapping）或者透射变换。简单来说就是二维→三维→二维的一个过程。其背后是复杂的**数学矩阵运算**进行像素坐标解析，在很多计算机视觉领域会用到此算法。
>

### 仿射变换算法
>
>&emsp;&emsp;仿射变换利用cvWarpAffine解决稠密仿射变换，用cvTransform解决稀疏仿射变换。仿射变换可以将矩形转换成平行四边形，它可以将矩形的边压扁但必须保持边是平行的，也可以将矩形旋转或者按比例变化。透视变换提供了更大的灵活性，一个透视变换可以将矩阵转变成梯形。当然，平行四边形也是梯形，所以仿射变换是透视变换的子集。
>


### 注意--OpenCV像素排列转换
>
>&emsp;&emsp;OpenCV中图像矩阵的顺序是 BGR。OpenCV 将颜色读取为 BGR（蓝绿色红色），但大多数计算机应用程序读取为 RGB（红绿蓝），常见的硬件屏幕色彩输>出顺序也是RGB。
>因此就产生了新问题，图片经OpenCV处理时必须由RGB排列方式转换为BGR排列方式。而OpenCV里已经提供了比较方便的转换办法,使用opencv自带函数转换图像的R通道和B通道。
>
>```python
>RGB –  BGR
> img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
> 
>BGR- RGB
> img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
>```
>


## 四、心得体会

1. 在学习过程中，让我对 java、c 和 python 不同高级语言间的关系和相似之处感受的更加深刻。

2. 了解了将高级语言应用到实际场景的一个代码实现的过程

3. 对 python 整个环境的配置过程（包括 linux）中和不同高级语言编写工具的练习（vscode、linux的 vim、pycharm）使用上有了新的认识并且熟练掌握，也增强了环境配置和理解的能力，结合操作系统知识，对我自身整体知识框架有了新的加强和体会。

4. 对人工智能-计算机视觉领域里文字识别的微浅研究和学习，让我窥见了人工智能的“智能”之处和算法之难，背后都是强大的高难数学知识支撑，以及感受到 python 之简洁。


