"""Modifed script to construct a dataset of DGL graphs with bond strengths"""
from __future__ import annotations
import os
import json
import dgl
import numpy as np
import torch
import warnings

from tqdm import trange
from pymatgen.core import Structure
from dgl.data.utils import load_graphs, save_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset

#import matgl
#from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph

from graph_converter import Icoxx2Graph

warnings.simplefilter("ignore")

float_np = np.float32
float_th = torch.float32

int_np = np.int32
int_th = torch.int32

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


def compute_pair_vector_and_distance(g: dgl.DGLGraph):
    """Calculate bond vectors and distances using dgl graphs.

    Args:
    g: DGL graph

    Returns:
    bond_vec (torch.tensor): bond distance between two atoms
    bond_dist (torch.tensor): vector from src node to dst node
    """
    dst_pos = g.ndata["pos"][g.edges()[1]] + g.edata["pbc_offshift"]
    src_pos = g.ndata["pos"][g.edges()[0]]
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


class ModMGLDataset(DGLDataset):
    """Create a dataset including dgl graphs."""
    def __init__(
        self,
        filename: str = "dgl_graph.bin",
        filename_lattice: str = "lattice.pt",
        filename_line_graph: str = "dgl_line_graph.bin",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.json",
        filename_ids: str = "ids.json",
        include_line_graph: bool = False,
        converter: Icoxx2Graph | None = None,
        threebody_cutoff: float | None = None,
        directed_line_graph: bool = False,
        structures_path: list | None = None,
        icohplists_path: list | None = None,
        icobilists_path: list | None = None,
        doscars_path: list | None = None, 
        chgcars_path: list | None = None, 
        
        labels: dict[str, list] | None = None,
        ids: list | None=None, 
        name: str = "ModMGLDataset",
        graph_labels: list[int | float] | None = None,
        save_cache: bool = True,
        raw_dir: str | None = None,
        save_dir: str | None = None,
    ):
        """
        Args:
            filename: file name for storing dgl graphs.
            filename_lattice: file name for storing lattice matrixs.
            filename_line_graph: file name for storing dgl line graphs.
            filename_state_attr: file name for storing state attributes.
            filename_labels: file name for storing labels.
            include_line_graph: whether to include line graphs.
            converter: dgl graph converter.
            threebody_cutoff: cutoff for three body.
            directed_line_graph (bool): Whether to create a directed line graph (CHGNet), or an
                undirected 3body line graph (M3GNet)
                Default: False (for M3GNet)
            structures_path: list of structure paths.
            icoxxlists_path: list of Icohplist paths.
            labels: targets, as a dict of {name: list of values}.
            name: name of dataset.
            graph_labels: state attributes.
            save_cache: whether to save the processed dataset. The dataset can be reloaded from save_dir
                Default: True
            raw_dir : str specifying the directory that will store the downloaded data or the directory that already
                stores the input data.
                Default: ~/.dgl/
            save_dir : directory to save the processed dataset. Default: same as raw_dir.
        """
        self.filename = filename
        self.filename_lattice = filename_lattice
        self.filename_line_graph = filename_line_graph
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.filename_ids = filename_ids
        self.include_line_graph = include_line_graph
        self.converter = converter
        self.structures_path = structures_path or []
        self.icohplists_path = icohplists_path or []
        self.icobilists_path = icobilists_path or []
        self.doscars_path = doscars_path or []
        self.chgcars_path = chgcars_path or []

        self.labels = labels or {}
        for k, v in self.labels.items():
            self.labels[k] = v.tolist() if isinstance(v, np.ndarray) else v
        self.ids=ids
        self.threebody_cutoff = threebody_cutoff
        self.directed_line_graph = directed_line_graph
        self.graph_labels = graph_labels
        self.save_cache = save_cache
        super().__init__(name=name, raw_dir=raw_dir, save_dir=save_dir)
    
    def has_cache(self) -> bool:
        """Check if the dgl_graph.bin exists or not."""
        files_to_check = [
            self.filename,
            self.filename_lattice,
            self.filename_state_attr,
            self.filename_labels,
        ]
        if self.include_line_graph:
            files_to_check.append(self.filename_line_graph)
        return all(os.path.exists(os.path.join(self.save_path, f)) for f in files_to_check)
        
    def process(self):
        """Convert Pymatgen structure into dgl graphs with icoxx as edge attributes."""
        num_graphs = len(self.structures_path)
        graphs = []
        lattices = []
        line_graphs = []
        state_attrs = []
        
        for idx in trange(num_graphs):
            path_to_structure = self.structures_path[idx]  # type: ignore
            path_to_icohplist = self.icohplists_path[idx]
            path_to_icobilist = self.icobilists_path[idx]
            path_to_doscar = self.doscars_path[idx] if isinstance(self.doscars_path, list) else self.doscars_path
            path_to_chgcar = self.chgcars_path[idx] if isinstance(self.chgcars_path, list) else self.chgcars_path
            graph, lattice, state_attr = self.converter.get_graph(path_to_structure=path_to_structure, 
                                                                  path_to_icohplist=path_to_icohplist, 
                                                                  path_to_icobilist=path_to_icobilist,
                                                                    path_to_doscar = path_to_doscar,
                                                                    path_to_chgcar = path_to_chgcar,
                                                                    asym_ind = False,)  # type: ignore)  # type: ignore
            #print(path_to_structure)
            graphs.append(graph)
            lattices.append(lattice)
            state_attrs.append(state_attr)
            #graph.ndata["pos"] = torch.tensor(Structure.from_file(path_to_structure).cart_coords)
            graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist

        all_chg = torch.cat([g.ndata['chg'] for g in graphs], dim=0)
        self.chg_mean = all_chg.mean(dim=0)
        self.chg_std = all_chg.std(dim=0, unbiased=False)  
        del all_chg
        all_dos = torch.cat([g.ndata['dos_centers'] for g in graphs], dim=0)
        self.dos_mean = all_dos.mean(dim=0)
        self.dos_std = all_dos.std(dim=0, unbiased=False)  
        del all_dos

        self.graphs = graphs
        
        for g in self.graphs:
            g.ndata['norm_chg'] = (g.ndata['chg'] - self.chg_mean) / self.chg_std
            g.ndata['norm_dos_centers'] = (g.ndata['dos_centers'] - self.dos_mean) / self.dos_std

            if self.include_line_graph:
                line_graph = create_line_graph(graph, self.threebody_cutoff, directed=self.directed_line_graph)  # type: ignore
                for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                    line_graph.ndata.pop(name)
                line_graphs.append(line_graph)

        if self.graph_labels is not None:
            state_attrs = torch.tensor(self.graph_labels).long()
        else:
            state_attrs = torch.tensor(np.array(state_attrs), dtype=default_dtype)
        
        self.lattices = lattices
        self.state_attr = state_attrs

        if self.include_line_graph:
            self.line_graphs = line_graphs
            return self.ids, self.graphs, self.lattices, self.line_graphs, self.state_attr
        return self.ids, self.graphs, self.lattices, self.state_attr
    
    def save(self):
        """Save dgl graphs and labels to self.save_path."""
        if self.save_cache is False:
            return

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.labels:
            with open(os.path.join(self.save_path, self.filename_labels), "w") as file:
                json.dump(self.labels, file)
        if self.ids:
            with open(os.path.join(self.save_path, self.filename_ids), "w") as file:
                json.dump(self.ids, file)
            
        save_graphs(os.path.join(self.save_path, self.filename), self.graphs)
        torch.save(self.lattices, os.path.join(self.save_path, self.filename_lattice))
        torch.save(self.state_attr, os.path.join(self.save_path, self.filename_state_attr))
        if self.include_line_graph:
            save_graphs(os.path.join(self.save_path, self.filename_line_graph), self.line_graphs)

    def load(self):
        """Load dgl graphs from files."""
        self.graphs, _ = load_graphs(os.path.join(self.save_path, self.filename))
        self.lattices = torch.load(os.path.join(self.save_path, self.filename_lattice))
        if self.include_line_graph:
            self.line_graphs, _ = load_graphs(os.path.join(self.save_path, self.filename_line_graph))
        self.state_attr = torch.load(os.path.join(self.save_path, self.filename_state_attr))
        with open(os.path.join(self.save_path, self.filename_labels)) as f:
            self.labels = json.load(f)
        with open(os.path.join(self.save_path, self.filename_ids)) as f:
            self.ids = json.load(f)

    def __getitem__(self, idx: int):
        """Get graph and label with idx."""
        items = [
            self.ids[idx],
            self.graphs[idx],
            self.lattices[idx],
            self.state_attr[idx],
            {k: torch.tensor(v[idx], dtype=default_dtype) for k, v in self.labels.items()},
        ]
        if self.include_line_graph:
            items.insert(3, self.line_graphs[idx])
        return tuple(items)

    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)


def collate_fn_graph(batch: list, include_line_graph: bool = False, multiple_values_per_target: bool = False) -> tuple:
    """Merge a list of dgl graphs to form a batch."""
    line_graphs = None
    if include_line_graph:
        ids, graphs, lattices, line_graphs, state_attr, labels = map(list, zip(*batch, strict=False))
    else:
        ids, graphs, lattices, state_attr, labels = map(list, zip(*batch, strict=False))
    g = dgl.batch(graphs)
    labels = (
        torch.vstack([next(iter(d.values())) for d in labels])  # type:ignore[assignment]
        if multiple_values_per_target
        else torch.tensor([next(iter(d.values())) for d in labels], dtype=default_dtype)  # type:ignore[assignment]
    )
    state_attr = torch.stack(state_attr)  # type:ignore[assignment]
    lat = lattices[0] if g.batch_size == 1 else torch.squeeze(torch.stack(lattices))
    if include_line_graph:
        l_g = dgl.batch(line_graphs)
        return ids, g, lat, l_g, state_attr, labels
    return ids, g, lat, state_attr, labels


def MGLDataLoader(
    train_data: dgl.data.utils.Subset,
    val_data: dgl.data.utils.Subset,
    collate_fn: Callable | None = None,
    test_data: dgl.data.utils.Subset = None,
    **kwargs,
) -> tuple[GraphDataLoader, ...]:
    """Dataloader for MatGL training.

    Args:
        train_data (dgl.data.utils.Subset): Training dataset.
        val_data (dgl.data.utils.Subset): Validation dataset.
        collate_fn (Callable): Collate function.
        test_data (dgl.data.utils.Subset | None, optional): Test dataset. Defaults to None.
        **kwargs: Pass-through kwargs to dgl.dataloading.GraphDataLoader. Common ones you may want to set are
            batch_size, num_workers, use_ddp, pin_memory and generator.

    Returns:
        tuple[GraphDataLoader, ...]: Train, validation and test data loaders. Test data
            loader is None if test_data is None.
    """
    if collate_fn is None:
        if "forces" not in train_data.dataset.labels:
            collate_fn = collate_fn_graph

    train_loader = GraphDataLoader(train_data, shuffle=True, collate_fn=collate_fn, **kwargs)
    val_loader = GraphDataLoader(val_data, shuffle=False, collate_fn=collate_fn, **kwargs)
    if test_data is not None:
        test_loader = GraphDataLoader(test_data, shuffle=False, collate_fn=collate_fn, **kwargs)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


if __name__=='__main__':
    import pandas as pd
    #converter = Icoxx2Graph(element_types=elem_list,bond_length_cutoff=4.5)
    converter = Icoxx2Graph(bond_length_cutoff=5, node_feats_type='cgcnn')
    df_ph=pd.read_csv('log_mean_300msd_lobsData.csv')
    #print(df_ph.prop.values.tolist()); exit()
    # convert the raw dataset into MEGNetDataset
    ph_dataset = ModMGLDataset(
        name='MGLDataset_ICOXX_OG',
        structures_path=list(df_ph.struc_dir.values),
        labels={"log_msd_mean_300": list(df_ph.log_msd_mean_300.values.tolist())},
        #labels=list(df_ph.prop.tolist()),
        converter=converter,
        icohplists_path=list(df_ph.ICOHP_dir.values),
        icobilists_path=list(df_ph.ICOBI_dir.values),
        ids=list(df_ph.id.values),
        #clear_processed=True,
        save_cache=False,
        save_dir='.'
    )

    print(ph_dataset[0:2])
    exit()
    train_data, val_data, test_data = split_dataset(
    ph_dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
    )
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_graph,
        batch_size=32,
        num_workers=0,
    )

    #for gr in val_loader:
     #   g, lattice, state_attr, label=gr
      #  print(g)

    #name='MGLDataset_ICOXX_OG',
    #structures_path=list(df_ph.struc_dir.values),
    #labels={"last_phdos_peak": list(df_ph.prop.values)},
    #labels=list(df_ph.prop.values),
    #converter=converter,
    #icoxxlists_path=list(df_ph.icoxx_lists.values),
    #clear_processed=True,
    #save_dir='.'
    #save_cache: bool = True,
