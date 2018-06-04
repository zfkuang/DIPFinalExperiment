###运行环境

​	Python3，tensorflow



### 已有函数

​	util.py中的uploadData可以自动生成所需的所有数据（前提是training文件夹存在）

​	layer.py中的函数可以方便建神经层



### 统一格式

​	所有测试所用文件都放在data文件夹下	

​	对于每个baseline算法，请在baseline文件夹下创建算法名称为名的python文件，并在其中完成算法（如需要多个文件请使用文件夹，通用的函数请写在util.py中）

​	完成算法后，请在main.py中提供一个使用该算法求解的接口函数：

​	def "算法名" (trainData, trainLabel, testData, testLabel)





（可选）可以先从网上下载fc7.npy, label.npy，放在根目录下，以后会再用到