import scipy.io as sio
import json

with open('dataset.json', encoding='utf8') as _fp:
    DS_CONF = json.load(_fp)
DS_DIR = DS_CONF['path']


class SpectralDataset:
    def __init__(self, dataset_name: str):
        dataset_name = dataset_name.lower()
        if dataset_name not in DS_CONF['dataset']:
            raise ValueError(dataset_name + ' not found in dataset.')
        conf = DS_CONF['dataset'][dataset_name]
        fp_img = conf['file'] + ('_corrected' if conf['corrected'] else '') + '.mat'
        fp_gt = conf['file'] + '_gt.mat'
        key_img = conf['key'] + ('_corrected' if conf['corrected'] else '')
        key_gt = conf['key'] + '_gt'
        self.name = dataset_name
        self.image = sio.loadmat(DS_DIR + fp_img)[key_img]
        self.ground_truth = sio.loadmat(DS_DIR + fp_gt)[key_gt]
        self.is_corrected = conf['corrected']
        

if __name__ == '__main__':
    for k in DS_CONF['dataset']:
        print(k, '*'.join(list(map(str, SpectralDataset(k).image.shape))))