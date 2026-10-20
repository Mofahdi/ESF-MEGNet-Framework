import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from glob import glob

import dgl 
from train.megnet import MEGNet
from process.graph_converter import Icoxx2Graph
from process.dataset import ModMGLDataset, MGLDataLoader, collate_fn_graph
from dgl.data.utils import split_dataset
from _training_dgl import ModelLightningModule
#from train.model_utils import load_model

import torch

from pytorch_lightning.loggers import CSVLogger
#from lightning.loggers import CSVLogger
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


converter = Icoxx2Graph(bond_length_cutoff=5, node_feats_type='Z') # 'rand_embd': 118, one hot 'Z': 118

best_ckpt_dir=os.path.join (os.getcwd(), 'best_megCkpts')
best_ckpt_name='best'
max_epochs=500
patience=300
device='cuda'
split_idx_avail=True
batch_size=64
csv_filename = 'log_mean_300msd_lobsData.csv'
col_name = 'log_msd_mean_300'



df=pd.read_csv(csv_filename)

# convert the raw dataset into MEGNetDataset
ph_dataset = ModMGLDataset(
        name='MGLDataset_ICOXX',
        structures_path=list(df.struc_dir.values),
        labels={"target": list(df[col_name].values.tolist())},
        converter=converter,
        icoxxlists_path=list(df.ICOHP_dir.values),
        ids=list(df.id.values),
        save_cache=False, # True 
        save_dir='.'
    )

if split_idx_avail:
    train_idx_filename = glob("train_idx_*.npy")[0]
    val_idx_filename = glob('val_idx_*.npy')[0]
    test_idx_filename = glob('test_idx_*.npy')[0]
    train_idx = np.load(train_idx_filename).tolist()
    val_idx = np.load(val_idx_filename).tolist()
    test_idx = np.load(test_idx_filename).tolist()

    from dgl.data.utils import Subset
    train_data=Subset(ph_dataset, train_idx)
    val_data=Subset(ph_dataset, val_idx)
    test_data=Subset(ph_dataset, test_idx)
    #train_data, val_data, test_data=ph_dataset[train_idx], ph_dataset[val_idx], ph_dataset[test_idx]
else:
    train_data, val_data, test_data = split_dataset(
    ph_dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
    )
#    pass

train_loader, val_loader, _ = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_graph,
        batch_size=batch_size,
        num_workers=0,
    )

model = MEGNet(
        dim_node_embedding = 118, # change to 118 or 92
        dim_edge_embedding = 50,
        dim_state_embedding = 2,
        ntypes_state = None,
        nblocks = 3,
        hidden_layer_sizes_input = (64, 32),
        hidden_layer_sizes_conv = (64, 64, 32),
        hidden_layer_sizes_output = (32, 16),
        nlayers_set2set = 1,
        niters_set2set = 2,
        activation_type = 'softplus2',
        #activation_type = "swish",
        is_classification = False,
        include_state = True,
        dropout = 0.0,
        #element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion = None,
        cutoff = 5.0,
        gauss_width = 0.5,
        #**kwargs,
    )
lit_module = ModelLightningModule(model=model, loss_params=None, loss='MAE')

checkpoint_callback = ModelCheckpoint(
   dirpath=best_ckpt_dir,
   monitor="val_MAE",
   filename= best_ckpt_name, #"-{epoch:02d}-{val_MAE:.4f}-{val_RMSE:.4f}-{train_MAE:.4f}-{train_RMSE:.4f}",
   save_top_k=1,
   mode="min",
)
early_stopping = EarlyStopping(monitor="val_MAE", patience=patience, mode="min", verbose=False); # True
logger = CSVLogger("logs", name="MEGNet_training")
trainer = pl.Trainer(max_epochs=max_epochs, accelerator=device, logger=logger, callbacks=[checkpoint_callback, early_stopping],) #callbacks=[checkpoint_callback, early_stopping] #callbacks=[checkpoint_callback]
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


best_ckpt_path=os.path.join(best_ckpt_dir, best_ckpt_name)
def get_best_module(lit_module, best_ckpt_path, device):
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    new_lit_module=copy.deepcopy(lit_module)
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k] = v
    new_lit_module.load_state_dict(new_state_dict)
    return new_lit_module


metrics = pd.read_csv("logs/MEGNet_training/version_0/metrics.csv", index_col='epoch')
metrics["train_MAE"].dropna().plot()
metrics["val_MAE"].dropna().plot()
metrics["val_MAE"].min()
metrics["train_MAE"].min()
print(metrics["val_MAE"].min(), metrics["train_MAE"].min())
import matplotlib.pyplot as plt
#plt.plot(range(len(metrics["train_MAE"].dropna())), metrics["train_MAE"].dropna())
plt.ylabel('MAE')
plt.legend()
plt.savefig('train_val.jpg')




from dgl.dataloading import GraphDataLoader
test_loader = GraphDataLoader(test_data, shuffle=False, collate_fn=collate_fn_graph, batch_size=1, num_workers=0,)

ids=[]; labels_list=[]; preds_best_list=[]
for gr in test_loader:
    id_, g, lattice, state_attr, label=gr
    pred_best_model=get_best_module(lit_module, best_ckpt_path+'.ckpt', 'cpu').model(g, state_attr)
    preds_best_list.append(pred_best_model.squeeze().detach().numpy().tolist())
    labels_list.append(label.squeeze().detach().numpy().tolist());
    ids.extend(id_)


from torch.nn import L1Loss
test_mae=L1Loss()
print('testing set MAE:', test_mae(torch.tensor(labels_list, dtype=torch.float64), torch.tensor(preds_best_list, dtype=torch.float64)))

res_df=pd.DataFrame({'id': ids, 'label': labels_list, 'preds_best': preds_best_list})
res_df.to_csv('results.csv', index=False)


from sklearn.metrics import r2_score
coefficient_of_dermination = r2_score(labels_list, preds_best_list)
print('R2=', coefficient_of_dermination)




import json
results = {
    "train_MAE_min": float(metrics["train_MAE"].min()),
    "val_MAE_min": float(metrics["val_MAE"].min()),
    "test_MAE": float(test_mae(
        torch.tensor(labels_list, dtype=torch.float64),
        torch.tensor(preds_best_list, dtype=torch.float64)
    )),
    "R2": float(r2_score(labels_list, preds_best_list))
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)






