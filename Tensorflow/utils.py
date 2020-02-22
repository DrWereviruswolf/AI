import numpy as np

# input form: data['time_attr'][time, attr, probability]
# input form: data['attr_val'][attr, val, probability]
data_sample = {'time_attr': np.array([[1, 10, 0.8], [2, 10, 0.9], [3, 10, 0.6], [1, 2, 0.6]], dtype=int),
        'attr_val': np.array([[10, 15, 0.9], [10, 17, 0.7], [2, 18, 0.6]], dtype=int)}


def Two2Three(data):
    time_attr = data['time_attr']
    attr_val = data['attr_val']
    if time_attr.shape[0] == 0 or attr_val.shape[0] == 0:
        return np.array([[]])
    ind_ta = np.lexsort((time_attr[:, 0], time_attr[:, 2], time_attr[:, 1]))
    time_attr = time_attr[ind_ta]
    ind_av = np.lexsort((attr_val[:, 1], attr_val[:, 2], attr_val[:, 0]))
    attr_val = attr_val[ind_av]
    time_attr_val = np.array([], dtype=int)
    attr_set = set()
    attr_set |= set(time_attr[:, 1])
    attr_set |= set(attr_val[:, 0])
    for attr in attr_set:
        ind_ta_attr = np.where(time_attr[:, 1] == attr)
        time_attr_now = time_attr[ind_ta_attr]
        ind_av_attr = np.where(attr_val[:, 0] == attr)
        attr_val_now = attr_val[ind_av_attr]
        minlen = min(len(time_attr_now), len(attr_val_now))
        if minlen == 0:
            continue
        time_attr_now = time_attr_now[-minlen:]
        attr_val_now = attr_val_now[-minlen:]
        ind_ta_now = np.lexsort((time_attr_now[:, 1], time_attr_now[:, 2], time_attr_now[:, 0]))
        time_attr_now = time_attr_now[ind_ta_now]
        ind_av_now = np.lexsort((attr_val_now[:, 0], attr_val_now[:, 2], attr_val_now[:, 1]))
        attr_val_now = attr_val_now[ind_av_now]
        time_attr_val_now = np.concatenate((time_attr_now[:, :2], attr_val_now[:, 1, None]), axis=1)
        if len(time_attr_val) == 0:
            time_attr_val = time_attr_val_now
        else:
            time_attr_val = np.concatenate((time_attr_val, time_attr_val_now), axis=0)
    return time_attr_val


if __name__ == "__main__":
    time_attr_val = Two2Three(data_sample)
    print(time_attr_val)
    print(time_attr_val.tolist())
