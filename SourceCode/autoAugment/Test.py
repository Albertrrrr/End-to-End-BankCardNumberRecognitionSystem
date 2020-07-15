import PIL
from autoaugment import ImageNetPolicy
import os

#原路地址
path = "C:\\Users\\hjnui\\Desktop\\BankSet"

for filename in os.listdir(path):
    image = PIL.Image.open(os.path.join(path, filename))
    policy = ImageNetPolicy()
    count = 1
    for i in range(10)[1:80]:
         transformed = policy(image)
         (filepath, tempfilename) = os.path.split(filename)
         (shotname, extension) = os.path.splitext(tempfilename)
         ImageName = "".join(shotname)
        # print(ImageName)
         #保存地址
         path_save = "C:\\Users\\hjnui\\Desktop\\BankCardData\\" + ImageName
         print("在:" + path_save + " 写入完成 目前为止是 第 " + str(count) + "项目 第" + str(i) + "个")
         if(i < 10):
            transformed.save(path_save + str(i) + ".jpg")
         else:
            path_save = path_save[0:len(path_save)]
            transformed.save(path_save + str(i) + ".jpg")
    count = count + 1
