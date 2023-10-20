from mindspore.dataset import GeneratorDataset,Dataset
class DataLoader():
    def __init__(self,data,column_names=['image','label'],batch_size=1,shuffle=False,num_workers=None,drop_last=False):
        self.data=data
        self.column_names=column_names
        if isinstance(self.data,Dataset)==False :
            self.data = GeneratorDataset(self.data, column_names=self.column_names)
        self.column_names=column_names
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.num_workers=num_workers
        self.drop_last=drop_last

    def __call__(self, *args, **kwargs):
        if self.shuffle:
            self.data=self.data.shuffle(self.data.get_dataset_size())
        self.data=self.data.batch(batch_size=self.batch_size,drop_remainder=self.drop_last,num_parallel_workers=self.num_workers)
        # self.data= self.data.create_dict_iterator()
        return self.data
    def __iter__(self):
        return self.data.create_dict_iterator()
