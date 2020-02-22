import json
import numpy as np
import random


def load_attribute_value(fold=True):
    data = json.load(open('../source_data/assignment_training_data_word_segment.json', 'rb'))
    # 主数据集
    index_attr_val = [[3 for col in range(157)] for row in range(3000)]
    attr_val = []

    for idx in range(3000):
        index_attr_val[idx][0: len(data[idx]['indexes'])] = data[idx]['indexes']

        # Attribute和value中的每个元素进行join,
        val_len = len(data[idx]['values'])
        attr_val_for_sent = [[0 for col in range(3)] for row in
                             range(len(data[idx]['attributes']) * val_len)]
        for i in range(len(data[idx]['attributes'])):
            for j in range(val_len):
                attr_val_for_sent[val_len * i + j][0:2] = [data[idx]['attributes'][i], data[idx]['values'][j]]
                # 在result的每行的[0:2]中查找，如果找到,则 1，否则 0
                for K in data[idx]['results']:
                    if attr_val_for_sent[val_len * i + j][0: 2] == K[1: 3]:
                        attr_val_for_sent[val_len * i + j][2] = 1
                # warning
                attr_val.append(index_attr_val[idx].copy())
                attr_val[-1][154: 157] = attr_val_for_sent[val_len * i + j]
        # 将生成的列表修改在第154:157上
    random.shuffle(attr_val)

    if fold:
        # 3 fold 7447 + 7447 + 7448
        fold1 = attr_val[0:12332].copy()
        fold2 = attr_val[12332:24664].copy()
        fold3 = attr_val[24664:36996].copy()

        # fold1 & fold2 拼合
        part1_train = fold1 + fold2
        part2_train = fold1 + fold3
        part3_train = fold2 + fold3

        # numpy
        part1_train_np = np.array(part1_train, dtype=np.int32)
        part2_train_np = np.array(part2_train, dtype=np.int32)
        part3_train_np = np.array(part3_train, dtype=np.int32)
        part1_test_np = np.array(fold3, dtype=np.int32)
        part2_test_np = np.array(fold2, dtype=np.int32)
        part3_test_np = np.array(fold1, dtype=np.int32)

        np.save("part1_train.npy", part1_train_np)
        np.save("part2_train.npy", part2_train_np)
        np.save("part3_train.npy", part3_train_np)
        np.save("part1_test.npy", part1_test_np)
        np.save("part2_test.npy", part2_test_np)
        np.save("part3_test.npy", part3_test_np)

    np.save("train_av.npy", attr_val)

    print(len(attr_val))
    return attr_val


if __name__ == "__main__":
    load_attribute_value()
