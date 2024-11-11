from datareader import Datareader
from datareader.feasai import FEASAIDataset

import torch



def collate_fn(data):  # 这里的data是一个list， list的元素是元组，元组构成为(self.data, self.label)
	# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,data[索引到数据index][索引到data或者label][索引到channel]
    data.sort(key=lambda x: len(x[0][0]), reverse=False)  # 按照数据长度升序排序
    data_list = []
    label_list = []
    min_len = len(data[0][0][0]) # 最短的数据长度 
    for batch in range(0, len(data)): #
        data_list.append(data[batch][0][:, :min_len])
        label_list.append(data[batch][1])
    data_tensor = torch.tensor(data_list, dtype=torch.float32)
    label_tensor = torch.tensor(label_list, dtype=torch.float32)
    data_copy = (data_tensor, label_tensor)
    return data_copy

class DatasetFactory:
    def __init__(self):
        self.FEASAI = FEASAIDataset

    def get(self,name,*args)->Datareader:
        if not hasattr(self,name): 
            raise NotImplementedError(f'Dataset types {name} are not supported')
        return getattr(self,name)(*args)
