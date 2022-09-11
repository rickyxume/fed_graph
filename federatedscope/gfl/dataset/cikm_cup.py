import logging

import torch
import os
from torch_geometric.data import InMemoryDataset

logger = logging.getLogger(__name__)

class CIKMCUPDataset(InMemoryDataset):
    name = 'CIKM22Competition'

    def __init__(self, root):
        super(CIKMCUPDataset, self).__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def processed_file_names(self):
        return ['pre_transform.pt', 'pre_filter.pt']

    def __len__(self):
        return len([
            x for x in os.listdir(self.processed_dir)
            if not x.startswith('pre')
        ])

    def _load(self, idx, split):
        try:
            data = torch.load(
                os.path.join(self.processed_dir, str(idx), f'{split}.pt'))
        except:
            data = None
        return data

    def process(self):
        pass

    def __getitem__(self, idx):
        data = {}
        for split in ['train', 'val', 'test']:
            split_data = self._load(idx, split)
            if split_data:
                data[split] = split_data
        return data


def load_cikmcup_data(config):
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data

    # Build data
    logger.info(f'Loading CIKMCUP data from {os.path.abspath(os.path.join(config.data.root, "CIKM22Competition"))}.')
    dataset = CIKMCUPDataset(config.data.root)
    # config.merge_from_list(['federate.client_num', len(dataset)])

    if len(dataset) == 0:
        raise FileNotFoundError(f'Cannot load CIKMCUP data from {os.path.abspath(os.path.join(config.data.root, "CIKM22Competition"))}, please check if the directory is correct.')

    data_dict = {}
    # Build DataLoader dict
    for client_idx in range(1, config.federate.client_num + 1):
        logger.info(f'Loading CIKMCUP data for Client #{client_idx}.')
        dataloader_dict = {}
        tmp_dataset = []
        
        trn_graphs = dataset[client_idx]['train']
        val_graphs = dataset[client_idx]['val']
        test_graphs = dataset[client_idx]['test']

        if 'train' in dataset[client_idx]:
            # 'SubMix','GraphCrop'没有配适
            augment_type = config.data.augment
            if augment_type in ['NodeSam','MotifSwap','DropEdge', 'DropNode', 'ChangeAttr', 'AddEdge', 'NodeAug']:
                from federatedscope.gfl.dataset.augment import Augment
                logger.info(f'Using {augment_type} data augmentation for Client #{client_idx}.')
                # Augment training graph data
                trn_graphs = [Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y) for graph in trn_graphs]
                trn_augment = Augment(trn_graphs, augment_type)
                trn_aug_graphs = trn_augment(torch.randperm(len(trn_graphs)),as_batch=False)
                # merge augmented data into original training set
                trn_graphs.extend(trn_aug_graphs)
            elif augment_type == '' or  augment_type is None:
                trn_graphs = [Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y) for graph in trn_graphs]
            else:
                pass

            if config.data.use_aug_val_in_training_set:
                logger.info(f'Using augmented val_data in training set for Client #{client_idx}.')
                val_graphs = [Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y=graph.y) for graph in val_graphs]
                val_augment = Augment(val_graphs, augment_type)
                val_aug_graphs = val_augment(torch.randperm(len(val_graphs)),as_batch=False)
                trn_graphs.extend(val_aug_graphs)

            dataloader_dict['train'] = DataLoader(trn_graphs,
                                                  config.data.batch_size,
                                                  shuffle=config.data.shuffle)
            tmp_dataset += trn_graphs
        if 'val' in dataset[client_idx]:
            dataloader_dict['val'] = DataLoader(val_graphs,
                                                config.data.batch_size,
                                                shuffle=False)
            tmp_dataset += val_graphs
        if 'test' in dataset[client_idx]:
            dataloader_dict['test'] = DataLoader(test_graphs,
                                                 config.data.batch_size,
                                                 shuffle=False)
            tmp_dataset += test_graphs
        if tmp_dataset:
            dataloader_dict['num_label'] = 0

        data_dict[client_idx] = dataloader_dict

    return data_dict, config
