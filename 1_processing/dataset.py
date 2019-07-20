import scipy.io as sio
import json
import os


with open('dataset.json', encoding='utf8') as _fp:
    DS_CONF = json.load(_fp)
DS_DIR = DS_CONF['path']


class SpectralDataset:
    def __init__(self, dataset_name: str):
        def find_key(_ds):
            print(_ds.keys())
            for _k in _ds.keys():
                if not _k.startswith('_'):
                    return _k
        dataset_name = dataset_name.lower()
        if dataset_name not in DS_CONF['dataset']:
            raise ValueError(dataset_name + ' not found in dataset.')
        self.conf = DS_CONF['dataset'][dataset_name]
        self.name = dataset_name
        self._img_raw = sio.loadmat(DS_DIR + self.conf['file'])
        self.image = self._img_raw[self.conf.get('img_key', find_key(self._img_raw))]
        if 'gt' in self.conf:
            self.type = 'gt'
        elif 'obj_key' in self.conf:
            self.type = 'obj'
        else:
            self.type = 'none'
        
        if self.type == 'gt':
            self._gt_raw = sio.loadmat(DS_DIR + self.conf['gt'])
            self.ground_truth = self._gt_raw[find_key(self._gt_raw)]
        elif self.type == 'obj':
            self.objects = self._img_raw[self.conf['obj_key']]
        

if __name__ == '__main__':
    for k in DS_CONF['dataset']:
        print(k, '*'.join(list(map(str, SpectralDataset(k).image.shape))))