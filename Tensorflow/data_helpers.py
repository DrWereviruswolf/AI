import numpy as np


def load_k_fold_data(path, fold):
    raw_data_tra = np.load(path + "part{}_train.npy".format(fold))
    raw_data_val = np.load(path + "part{}_test.npy".format(fold))

    x_text_tra = raw_data_tra[:, :-3]
    x_position_tra = raw_data_tra[:, -3:-1]
    y_tra = raw_data_tra[:, -1]
    x_text_val = raw_data_val[:, :-3]
    x_position_val = raw_data_val[:, -3:-1]
    y_val = raw_data_val[:, -1]

    return x_text_tra, x_position_tra, y_tra, x_text_val, x_position_val, y_val


def load_train_data(path):
    raw_data = np.load(path)

    x_text = raw_data[:, :-3]
    x_position = raw_data[:, -3:-1]
    y = raw_data[:, -1]

    return x_text, x_position, y


def batch_iter(x_text, x_position, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_size = len(x_text)
    num_batches_per_epoch = int((len(x_text) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if y is not None:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_x_text = x_text[shuffle_indices]
                shuffled_x_position = x_position[shuffle_indices]
                shuffled_y = y[shuffle_indices]
            else:
                shuffled_x_text = x_text
                shuffled_x_position = x_position
                shuffled_y = y
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield [shuffled_x_text[start_index:end_index],
                       shuffled_x_position[start_index:end_index],
                       shuffled_y[start_index:end_index]]
        else:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_x_text = x_text[shuffle_indices]
                shuffled_x_position = x_position[shuffle_indices]
            else:
                shuffled_x_text = x_text
                shuffled_x_position = x_position
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield [shuffled_x_text[start_index:end_index],
                       shuffled_x_position[start_index:end_index]]


def load_test_data(data):
    len_index = 154
    index_attr_val = [[3 for col in range(len_index + 3)] for row in range(len(data))]
    attr_val = []
    index_time_attr = [[3 for col in range(len_index + 3)] for row in range(len(data))]
    time_attr = []

    for idx in range(len(data)):
        index_time_attr[idx][0: min(len_index, len(data[idx]['indexes']))] = data[idx]['indexes'][0: min(len_index, len(data[idx]['indexes']))]

        # time中的每个元素和Attribute中的每个元素进行join,
        attr_len = len(data[idx]['attributes'])
        time_attr_for_sent = [[0 for col in range(3)] for row in
                              range(len(data[idx]['times']) * attr_len)]
        for i in range(len(data[idx]['times'])):
            for j in range(len(data[idx]['attributes'])):
                time_attr_for_sent[attr_len * i + j][0:2] = [data[idx]['times'][i], data[idx]['attributes'][j]]
                time_attr_for_sent[attr_len * i + j][2] = idx
                time_attr.append(index_time_attr[idx].copy())
                time_attr[-1][-3:] = time_attr_for_sent[attr_len * i + j]

    for idx in range(len(data)):
        index_attr_val[idx][0: min(len_index, len(data[idx]['indexes']))] = data[idx]['indexes'][0: min(len_index, len(data[idx]['indexes']))]

        # Attribute和value中的每个元素进行join,
        val_len = len(data[idx]['values'])
        attr_val_for_sent = [[0 for col in range(3)] for row in
                             range(len(data[idx]['attributes']) * val_len)]
        for i in range(len(data[idx]['attributes'])):
            for j in range(val_len):
                attr_val_for_sent[val_len * i + j][0:2] = [data[idx]['attributes'][i], data[idx]['values'][j]]
                attr_val_for_sent[val_len * i + j][2] = idx
                # warning
                attr_val.append(index_attr_val[idx].copy())
                attr_val[-1][-3:] = attr_val_for_sent[val_len * i + j]

    time_attr = np.array(time_attr)
    attr_val = np.array(attr_val)
    return time_attr[:, :-3], time_attr[:, -3:-1], time_attr[:, -1], attr_val[:, :-3], attr_val[:, -3:-1], attr_val[:, -1]


if __name__ == "__main__":
    pass
