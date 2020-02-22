import json
import numpy as np
import random


def load_time_attribute(fold=True):
    data = json.load(open('../source_data/assignment_training_data_word_segment.json', 'rb'))
    # 主数据集
    index_time_attr = [[3 for col in range(157)] for row in range(3000)]
    time_attr = []

    for idx in range(3000):
        index_time_attr[idx][0: len(data[idx]['indexes'])] = data[idx]['indexes']

        # time中的每个元素和Attribute中的每个元素进行join,
        attr_len = len(data[idx]['attributes'])
        time_attr_for_sent = [[0 for col in range(3)] for row in
                              range(len(data[idx]['times']) * attr_len)]
        for i in range(len(data[idx]['times'])):
            for j in range(len(data[idx]['attributes'])):
                time_attr_for_sent[attr_len * i + j][0:2] = [data[idx]['times'][i], data[idx]['attributes'][j]]
                # 在result的每行的[0:2]中查找，如果找到,则 1，否则 0
                for K in data[idx]['results']:
                    if time_attr_for_sent[attr_len * i + j][0: 2] == K[0: 2]:
                        time_attr_for_sent[attr_len * i + j][2] = 1
                # warning
                time_attr.append(index_time_attr[idx].copy())
                time_attr[-1][154: 157] = time_attr_for_sent[attr_len * i + j]
        # 将生成的列表修改在第154:157上

    random.shuffle(time_attr)

    if fold:
        # 3 fold 7447 + 7447 + 7448
        fold1 = time_attr[0:7447].copy()
        fold2 = time_attr[7447:14894].copy()
        fold3 = time_attr[14894:22342].copy()

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

    np.save("train_ta.npy", time_attr)

    print(len(time_attr))
    return time_attr


if __name__ == "__main__":
    load_time_attribute()
