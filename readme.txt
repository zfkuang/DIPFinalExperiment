文件说明：
alexnet.py --AlexNet的tensorflow网络结构定义文件
bvlc_alexnet.npy --用1000类base classes预训练好的AlexNet模型参数
fc7.npy --1000类base classes的fc7层特征采样，每个类别约64个样本
label.npy  --对应fc7.npy，为每个样本的标签
base_classes.txt --1000类基本类的名称
training文件夹 --包含50个新类的训练图片，每个类别10张

由于fc7.npy文件比较大，故和label.npy一起放在了百度云
链接：https://pan.baidu.com/s/1WydGFkP3zWBRwIBWxXPzFA 密码：91ek



======================
特别说明：由于预训练的模型输入是以BGR进行存储的，故在训练时需要对输入图像做一些处理
参考代码如下：
        VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
         # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
