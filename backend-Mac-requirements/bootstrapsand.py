from scipy.stats import bootstrap
import numpy as np

def custom_stat(values):
    # print(np.array(values).shape)
    v0 = values[0]
    v1 = values[1]
    return np.std(v0) + np.std(v1)

data0 = np.random.normal(5, 1, size=100)
data1 = np.random.normal(6, 2, size=100)

data = list(zip((data0, data1)))
print(data)
res = bootstrap((data,), custom_stat, n_resamples=3)