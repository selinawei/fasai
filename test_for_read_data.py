import numpy as np

# 加载NPZ文件
data = np.load('/home/yuanqw/Wlx/SAI_data/operated_npz_file_with_frame/0002.npz')

data = np.load('/home/yuanqw/Wlx/SAI_data/occ_test/complex/0000.npz')

# data是一个类似字典的对象，所以你可以使用键来访问其中的数组
# 例如，如果你知道一个数组的键名是'array1'，你可以这样访问它：
# array1 = data['array1']

# 打印数组内容
# print(array1)

# 遍历所有数组
for key in data:
    print(f"Key: {key}")
    print("Array:", data[key].shape)
    if key == 'occ_aps_ts':
        formatted_arr = np.vectorize(lambda x: format(x, '.7f'))(data[key]) 
        print(formatted_arr)
    if key =='occ_free_aps_ts':
        formatted_arr = np.vectorize(lambda x: format(x, '.7f'))(data[key]) # .4f保留四位小鼠
        print(formatted_arr)
# 不要忘记关闭文件
data.close()
