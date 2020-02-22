#数据处理

本程序实现3fold的数据集

数据宽度为157, 长度见下文

data.py实现了index,time,attribute以及相应的label
7447 + 7447 + 7448 = 22342

data.py实现了index,attribute,value以及相应的label
12332 + 12332 + 12332 = 36996

load.py用于查看numpy数据长度, 即二元组条数

如果想要生成新的3fold数据，只需python data.py以及source_data中的原始数据
