import os

path = r'F:\datasets\coco128\labels\val'  # 需改的txt文件的路径
filenames = os.listdir(path)  # 将所有txt文件的文件名用filename储存起来

for filename in filenames:  # 遍历所有txt文件
    position = path + '\\' + filename  # 获取绝对路径 '\\'有一个斜杠是转义符
    # print(position) #此行代码不注释可用于检查绝对路径是否正确

    with open(position) as f:

        lines = f.readlines()  # 以行为单位读取txt文件中的内容
        s = [line[:-1].split(' ') for line in lines]  # 以空格为标志分割txt中每行的内容，并以列表的方式储存在s中
        # print(s) #取消注释可以看是否成功分行分元素储存

        for i in range(len(s)):  # 遍历s的每一行
            # print(s[i][0]) #取消注释可看每一行的第一个元素是多少

            s[i][0] = '0'  # 如果每一行的第一个元素为1，也就是分类的标签为1，则将其修改成0
                # print(s) #取消注释可查看是否成功将第一列全修改为0
                # print(len(s[i]))

            with open(position, 'w', encoding='utf-8') as f:  # 将修改完的s写入到txt文件当中
                for j in range(len(s)):  # 此处采用先按行读取，再按列一个一个元素读取
                    for k in range(len(s[j])):
                        f.write((s[j][k] + ' '))  # 写入的每一个元素都以空格隔开，因此需要+' '
                    f.write("\n")  # 写完一行之后需要换行