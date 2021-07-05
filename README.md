# LabelTool

遥感影像样本制作工具，结合专业软件ENVI使用。

## 1. 使用条件
首先利用ENVI进行兴趣区选取，利用ROI工具对目标影像.tif制作多边形的.xml或.tif文件，即**标注文件**。
然后使用该工具完成其他批量制作任务。

## 2. 目前功能与使用方法

### 基本参数：
- f：选择功能
- i：设置影像的路径或目录
- l：设置标注文件的路径或目录
- s：设置保存路径或目录

### f 选择功能参数

目前主要包括三个简单功能：批量裁剪、制作VOC格式的目标检测样本、样本切分。

功能参数对应选值：  
- bc：批量裁剪, 利用多个标注文件将.tif影像裁剪。
- c：裁剪, 针对一个标注文件裁剪对应的.tif影像。
- bs：批量制作样本, 利用多个标注文件批量制作VOC目标检测样本。
- s：制作VOC样本。
- d：切分样本，将样本分为训练集、验证集或测试集。


### 功能介绍

1. bc 批量裁剪：采用一批标注文件将他们对应的.tif影像进行裁剪。
    - 其他参数介绍：
        - mask：是否利用标注文件对裁剪的影像进行掩膜，默认为True，即是。
        - size：设置固定的裁剪结果的大小，需要输入两个int型参数，默认为None，即不进行限制。
        - center：是否以多边形中心进行裁剪，默认为False，即否，size为None时，该参数无效。
        - count：每一个多边形输出裁剪结果的数量，默认为1，center为True时，该参数将默认为1。
        - bias：设置裁剪时基于中心的随机偏移量取值范围，需要输入两个float型参数，默认为(0, 100)，center为True时，该参数无效。
        - pos：是否保存每一个结果在原图的位置，默认为False，即否，size为None时，该参数无效。
    - 基本使用方法：
    ```
    >>> python sample.py -f bc -i [image_dir] -l [label_dir] -s [save_dir]
    ```
    其他参数有默认设置，详细可查看help。
2. c 裁剪：即根据一个标注文件将对应的遥感影像裁剪成多个多边形。
    - 其他参数介绍：
        与批量裁剪时所用参数相同。
    - 基本使用方法：
    ```
    >>> python sample.py -f c -i [image_path] -l [label_path] -s [save_path]
    ```
    其他参数有默认设置，详细可查看help。
3. bs 批量制作样本：即根据一批标注文件，制作VOC格式的目标检测样本。
    - 其他参数介绍：
        - d：可视化结果的保存路径或目录，默认为None，即不输出可视化。
        - sp：是否对标注文件进行分离，如果分离则每个多边形将作为一个对象裁剪，否则每个region作为一个对象裁剪，默认为True，即分离。
    - 基本使用方法：
    ```
    >>> python sample.py -f bs -i [image_dir] -l [label_dir] -s [save_dir]
    ```
    其他参数有默认设置，详细可查看help。
4. s 制作VOC样本：根据标注文件，将其制作为VOC格式的目标检测样本。
    - 其他参数介绍：
        - d：可视化结果的保存路径或目录，默认为None，即不输出可视化。
        - sp：是否对标注文件进行分离，如果分离则每个多边形将作为一个对象裁剪，否则每个region作为一个对象裁剪，默认为True，即分离。
    - 基本使用方法：
    ```
    >>> python sample.py -f s -i [image_path] -l [label_path] -s [save_path]
    ```
    其他参数有默认设置，详细可查看help。
5. d 切分样本：将样本分为训练集、验证集或测试集。
    - 其他参数介绍：
        - name：设置类别名称。
        - types：样本集类型，输入两个类型名，默认三个为('train', 'val', 'test')。
        - pct：不同类别样本集所占比例，输入两个float型数值，与types对应，默认为(0.6, 0.3, 0.1)。
        - shuffle：是否乱序，默认为True，即是。
    - 基本使用方法：
    ```
    >>> python sample.py -f d -i [image_path] -s [save_path] --name [catetory]
    ```
    其他参数有默认设置，详细可查看help。

### 其他

该程序设计了一个新的类——EnviLabel，用于执行与ENVI标注文件相关的一些操作以及获取标注文件的一些基本信息。
在EviLabel实现过程中，隐藏了一个小小的创新点，即采用一种新的提取连通区的方式从二值图中提取多边形，具体详见label.py -> EnviLabel -> extract_from_img(self, path) -> # extraction以及utils -> convex_hell.py -> array2vector(array)。

EnviLabel中还有一些内容需要完善，主要集中在对地理坐标的适配上，后期可能会完善。

希望这个小小程序能对诸君有所帮助。
交流邮箱：haowang_rs@foxmail.com