from __future__ import annotations
import dgl
import numpy as np
import torch
#import matgl
import warnings
from tqdm import trange

from ase import Atoms, Atoms
from ase.neighborlist import neighbor_list
from ase.io import read
from jarvis.core.specie import Specie
from jarvis.core.specie import get_node_attributes

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.lobster import Grosspop

#from lobsterpy.featurize.core import FeaturizeIcoxxlist
#from matgl.graph.converters import GraphConverter

from test_nbrs import icoxx_struc_fxn

# checklist:
# 1- check missing interactions are zero and skip if they are not
# 2- 

warnings.simplefilter("ignore")

float_np = np.float32
float_th = torch.float32

int_np = np.int32
int_th = torch.int32

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type.

    Arguments
    ---------
    emb_size: int
        Atom embeddings size
    """

    def __init__(self, emb_size, num_elements, sparse=False):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = torch.nn.Embedding(num_elements + 1, emb_size, sparse=sparse)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Forward pass of the atom embedding layer.

        Returns
        -------
        h: torch.Tensor, shape=(nAtoms, emb_size)
            Atom embeddings.
        """
        h = self.embeddings(Z)
        return h


class Icoxx2Graph:
    """Construct a DGL graph from Pymatgen Structure and Icohplist objects."""
    def __init__(
        self,
        #element_types: tuple[str, ...],
        bond_length_cutoff: float | None = None,
        node_feats_type: str='Z', 
    ):
        """
        Args:
        ----------
        element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
            constructed with the same dimensionality of features.
        bond_length_cutoff: bond length cutoff radius for graph representation
        """
        #self.element_types = tuple(element_types)
        self.bond_length_cutoff = bond_length_cutoff
        self.node_feats_type = node_feats_type
    
    def get_graph(self, 
                  path_to_structure: str,
                  path_to_icoxxlist: str, 
                  path_to_grosspop: str | None = None,
                  ) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Get a DGL graph from an input Structure and ICOXXLIST files.
        
        :param path_to_structure: path to structure 
        :param path_to_icohplist: path to icohplist.lobster
        :return:
            g: DGL graph
            lat: lattice for periodic systems
            state_attr: state features
        """

        structure=read(path_to_structure)
        #pmg_struc=Structure.from_file(path_to_structure)

        mapped_icoxx_data_reverse, missing_interactions_reverse=icoxx_struc_fxn(CONTCAR_path=path_to_structure, 
                                                                            r_cutoff = self.bond_length_cutoff, 
                                                                            ICOXX_path = path_to_icoxxlist, 
                                                                            test_bond_len=False, missing_check=False, 
                                                                            missing_cutoff=4, missing_icoxx=0)

        src_ids = []
        dst_ids = []
        icoxxs = []
        images = []
        for _, src_dst_dist_img_ic in enumerate(mapped_icoxx_data_reverse):
            src=src_dst_dist_img_ic[0][0] 
            dst=src_dst_dist_img_ic[0][1] 

            img = np.array(src_dst_dist_img_ic[2])
            icoxx= src_dst_dist_img_ic[-1]

            src_ids.append(src)
            dst_ids.append(dst)
            icoxxs.append(icoxx)
            images.append(img)

        if path_to_grosspop:
            grosspop= Grosspop(filename=path_to_grosspop) 
            pair_n_electrons = np.array(
                    [
                        grosspop.list_dict_grosspop[at1]["Loewdin GP"]["total"]
                        + grosspop.list_dict_grosspop[at2]["Loewdin GP"]["total"]
                        for at1, at2 in zip(src_ids, dst_ids)
                    ]
                    )
            norm_bond_strength=np.asarray(icoxxs)/pair_n_electrons
            #print(type(icoxxs), type(pair_n_electrons), norm_bond_strength.dtype)
            #print(icoxxs/pair_n_electrons); exit()
        lattice_matrix=np.expand_dims(structure.cell.array, axis=0)
        g, lat, state_attr = self.get_graph_from_processed_structure(
            structure,
            src_ids,
            dst_ids,
            images,
            lattice_matrix,
            #element_types,
            structure.get_scaled_positions(),
            icoxxs,
        )
        if path_to_grosspop:
            g.edata["norm_bond_strengths"]=torch.as_tensor(norm_bond_strength, dtype=default_dtype)

        return g, lat, state_attr


    def get_graph_from_processed_structure(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        #element_types,
        frac_coords,
        icoxxs,
    ) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Construct a dgl graph from processed structure and bond strength information.

        Args:
            structure: Input crystals or molecule of pymatgen structure or molecule types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            element_types: Element symbols of all atoms in the structure.
            frac_coords: Fractional coordinates of all atoms in the structure. Note: Cartesian coordinates for molecule
            is_atoms: whether the input structure object is ASE atoms object or not.

        Returns:
            DGLGraph object, state_attr

        """
        u, v = torch.tensor(src_id), torch.tensor(dst_id)
        g = dgl.graph((u, v), num_nodes=len(structure))

        # TODO: Need to check if the variable needs to be double or float, now use float
        lattice = torch.tensor(np.array(lattice_matrix), dtype=default_dtype)

        if self.node_feats_type.lower()=='z':
            #one_hot_atomic
            node_features = torch.nn.functional.one_hot(
                            torch.from_numpy(structure.get_atomic_numbers().copy()-1).long(), num_classes=118,
                            ).to(dtype=default_dtype)
        elif self.node_feats_type.lower()=='orb':
            #atom_emb=AtomEmbedding(emb_size=92, num_elements=118, sparse=False)
            atom_emb=AtomEmbedding(emb_size=118, num_elements=118, sparse=False)
            node_features = atom_emb(torch.from_numpy(structure.get_atomic_numbers().copy()-1).long()).detach()

        elif self.node_feats_type.lower()=='cgcnn':
            sps_features = []
            for ii, s in enumerate(structure.get_chemical_symbols().copy()):
                feat = list(get_node_attributes(s, atom_features='cgcnn'))
                sps_features.append(feat)
            node_features = torch.tensor(sps_features, dtype=default_dtype)
        
        else:
            raise ValueError("""use one of the following feature types: "Z", "orb", or "cgcnn" ignoring the case 
                    "Z": one hot encoding atomic number
                    "orb": node features similar node features to orb orb
                    "cgcnn": features from CGCNN""")

        g.ndata["node_feats"] = node_features
        positions = torch.as_tensor(structure.positions.copy(), dtype=default_dtype);
        g.ndata["pos"] = positions
        #g.ndata["positions"] = positions
        # TODO: Need to check if the variable needs to be double or float, now use float
        g.ndata["frac_coords"] = torch.tensor(frac_coords, dtype=default_dtype)

        # Note: pbc_ offshift and pos needs to be float64 to handle cases where bonds are exactly at cutoff
        pbc_offset = torch.tensor(images, dtype=default_dtype)
        g.edata["pbc_offset"] = pbc_offset
        g.edata["bond_strengths"] = torch.tensor(icoxxs, dtype=default_dtype) #* -1

        edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[torch.tensor(src_id)]
        #print(torch.tensor(src_id).dtype); print(torch.tensor(images, dtype=default_dtype).dtype, lattice[edge_batch].dtype, positions.dtype)
        edge_vec = (positions[torch.tensor(dst_id)]
                - positions[torch.tensor(src_id)]
                + torch.einsum('ni,nij->nj', torch.tensor(images, dtype=default_dtype), lattice[edge_batch]))
        g.edata["edge_vec"] = edge_vec

        state_attr = np.array([0.0, 0.0]).astype(np.float64) #float_np)
        return g, lattice, state_attr



if __name__=='__main__':
    g, lattice, state_attr=Icoxx2Graph(bond_length_cutoff=5, node_feats_type='cgcnn').get_graph(path_to_structure='CONTCAR.gz',
                path_to_icoxxlist='ICOHPLIST.lobster.gz', path_to_grosspop='GROSSPOP.lobster.gz',
            )    
    print(g,) #g.edges()) 
    print(g.num_nodes(), g.num_edges(), g.nodes(), g.ndata['node_feats'].shape, g.ndata['node_feats'].dtype)
    print(g.edata['bond_strengths'].shape, g.edata['norm_bond_strengths'].shape)


    exit()
    g, lattice, state_attr=Icoxx2Graph(bond_length_cutoff=5, node_feats_type='cgcnn').get_graph(path_to_structure='CONTCAR_CH',
                path_to_icoxxlist='ICOHPLIST.lobster', 
            )
    print(g,) #g.edges()) 
    print(g.num_nodes(), g.num_edges(), g.nodes(), g.ndata['node_feats'].shape, g.ndata['node_feats'].dtype)
    #print(g.edata['bond_strengths'])

    def get_element_list(train_structures: list[Structure]) -> tuple[str, ...]:
        """Get the tuple of elements in the training set for atomic features.

        Args:
            train_structures: pymatgen Molecule/Structure object

        Returns:
            Tuple of elements covered in training set
        """
        elements: set[str] = set()
        for s in train_structures:
            elements.update(s.composition.get_el_amt_dict().keys())
        return tuple(sorted(elements, key=lambda el: Element(el).Z))

#pmg_struc=Structure.from_file('CONTCAR')
#els=get_element_list(train_structures=[pmg_struc, Structure.from_file('CONTCAR_NaCl')]);
#print(els)
#node_type = (
#            np.array([els.index(site.specie.symbol) for site in Structure.from_file('CONTCAR_NaCl')])
#        )
#element_to_index = {elem: idx for idx, elem in enumerate(els)}  
#print(element_to_index, node_type)


#CH_z=read('CONTCAR_CH').get_atomic_numbers()-1
#one_hot_atomic = torch.nn.functional.one_hot(
 #       torch.from_numpy(CH_z), num_classes=118
  #      )
#print(one_hot_atomic.shape)

#print(torch.from_numpy(CH_z).dtype)
#exit()
#atomic_number_rep = torch.from_numpy(CH_z).long()
#print(atomic_number_rep.dtype)


#atom_emb=AtomEmbedding(emb_size=256, num_elements=118, sparse=False)
#atomic_embedding = atom_emb(atomic_number_rep)
#print(atomic_embedding.dtype)
#exit()
