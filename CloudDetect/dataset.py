from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from .util import read_off

class CloudDataset(Dataset):
    def __init__(self, metadata, preprocessor, root, class_mapper, device, one_hot = False):
        self.metadata = metadata
        self.preprocessor = preprocessor
        self.root = root
        self.class_mapper = class_mapper
        self.one_hot = one_hot
        self.device = device
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data_instance = self.metadata.iloc[idx]
        class_item = self.class_mapper[data_instance['class']]
        if self.one_hot:
            class_item = torch.nn.functional.one_hot(torch.tensor(class_item), num_classes=len(self.class_mapper))
        file = self.root + data_instance['object_path']
        verts, faces = read_off(open(file))
        return {'data':self.preprocessor(verts),'category':class_item}



def assign_val_indices(df, target_name, n_splits):
    skf = StratifiedKFold(n_splits=n_splits)
    df['kfold'] = None
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df[target_name].values)):
        df.loc[val_idx, 'kfold'] = fold