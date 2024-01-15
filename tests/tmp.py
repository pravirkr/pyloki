

def get_data_arr(fold_segment):
    n_accel, n_period = fold_segment.shape[:2]
    data_arr = np.zeros((n_accel*n_period,) + fold_segment.shape[2:])  
    k = 0
    for i in range(n_accel):
        for j in range(n_period):
            data_arr[k] = fold_segment[i, j]
            k += 1
    return data_arr
