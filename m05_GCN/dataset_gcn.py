import os, json, torch, dgl, yaml

from dgl.data import DGLDataset

import graph.json
import config_files

class GCNDataset(DGLDataset):

    def __init__(self):

        super().__init__(name='my_gcn_dataset')

        data_cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'data_gcn.yaml')
        with open(data_cfg_path, 'rb') as f:
            data_cfg = yaml.load(f, Loader=yaml.FullLoader)

        ifc_classes = data_cfg['ifc_classes']
        type_to_idx = {cls_name: idx for idx, cls_name in enumerate(ifc_classes)}
        num_classes = len(ifc_classes)

        path_to_graph_raw =  os.path.dirname(graph.json.__file__)
        graph_files = [
            f for f in os.listdir(path_to_graph_raw)
            if f.endswith("_with_latent.json")
        ]

        self.graphs = []

        for file_name in graph_files:
            #Get data from json
            with open(os.path.join(path_to_graph_raw, file_name), 'r') as f:
                data = json.load(f)
            #Nodes
            node_features = []
            targets = []
            nodes_sorted = sorted(data['nodes'], key=lambda x: x['index'])
            for node in nodes_sorted:
                type_name = node['type']
                type_idx = type_to_idx[type_name]
                one_hot_vector = torch.nn.functional.one_hot(torch.tensor(type_idx), num_classes=num_classes).float()
                latent_code = torch.tensor(node['latent_code'], dtype=torch.float32)
                full_feature = torch.cat([one_hot_vector, latent_code], dim=0)
                node_features.append(full_feature)
                targets.append(latent_code)
            # Feature list convert
            node_features = torch.stack(node_features)
            targets = torch.stack(targets) 
            # Edges
            edges = data['edges']
            src, dst = zip(*edges)
            src = torch.tensor(src)
            dst = torch.tensor(dst)
            g = dgl.graph((src, dst))
            g = dgl.to_bidirected(g)
            # g = dgl.add_self_loop(g)
            g.ndata['feat'] = node_features
            g.ndata['target'] = targets 
            # Adding graph
            self.graphs.append(g)

        self.batched_graph = dgl.batch(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

def main():
    data = GCNDataset()
    print(data.__getitem__(0))

if __name__=='__main__':
    main()