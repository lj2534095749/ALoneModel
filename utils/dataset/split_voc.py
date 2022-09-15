import os
import random

trainval_percent = 0.8                          # 训练集+验证集总占比
train_percent = 0.875                           # 训练集在trainval_percent里的train占比，0.875*0.8=0.7，因此训练集在总样本中占比70%
VOCdevkit_path = 'VOCdevkit'                    # 数据集文件路径
random.seed(0)                                  # 设定种子，使得程序能够复现

print("Generate txt in ImageSets.")
xmlfilepath = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')           # 标签文件路径
saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')       # 训练集、验证集、测试集txt文件的所在路径
temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num = len(total_xml)                            # 获得数据集样本的总数量
list = range(num)                               # 获得数据集样本的索引
tv = int(num * trainval_percent)                # 验证集+训练集样本的总数量
tr = int(tv * train_percent)                    # 训练集样本的数量
trainval = random.sample(list, tv)              # 训练集+验证集样本索引构成的列表
train = random.sample(trainval, tr)             # 训练集样本索引构成的列表
# random.sample(list, tv) 表示从list中生成一个长度为tv新列表，新列表中的元素从list中取样获得
# 而list是一个range对象，表示数据集的索引

print("train and val size", tv)
print("train size", tr)

ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'  # total_xml[i][:-4]之所以只到-4，是因为最后4位是 .xml，这个我们暂时不需要
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
print("Generate txt in ImageSets done.")
