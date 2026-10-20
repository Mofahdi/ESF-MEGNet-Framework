"""Implementation of Modified MatErials Graph Network (MEGNet) model.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from dgl.nn import Set2Set
from torch import nn

from process.dataset import compute_pair_vector_and_distance

from ._bond import BondExpansion
from ._activations import ActivationFunction
from .layers import MLP, EdgeSet2Set, MEGNetBlock, EmbeddingBlock
#from matgl.utils.io import IOMixIn
from .model_utils import IOMixIn


if TYPE_CHECKING:
    import dgl
    from process.graph_converter import Icoxx2Graph

logger = logging.getLogger(__file__)


class ESF_MEGNet(nn.Module, IOMixIn):
    """DGL implementation of MEGNet."""

    __version__ = 1

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        ntypes_state: int | None = None,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        is_classification: bool = False,
        include_state: bool = True,
        dropout: float = 0.0,
        #element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion: BondExpansion | None = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    ):
        """Useful defaults for all arguments have been specified based on MEGNet formation energy model.

        Args:
            dim_node_embedding: Dimension of node embedding.
            dim_edge_embedding: Dimension of edge embedding.
            dim_state_embedding: Dimension of state embedding.
            ntypes_state: Number of state types.
            nblocks: Number of blocks.
            hidden_layer_sizes_input: Architecture of dense layers before the graph convolution
            hidden_layer_sizes_conv: Architecture of dense layers for message and update functions
            nlayers_set2set: Number of layers in Set2Set layer
            niters_set2set: Number of iterations in Set2Set layer
            hidden_layer_sizes_output: Architecture of dense layers for concatenated features after graph convolution
            activation_type: Activation used for non-linearity
            is_classification: Whether this is classification task or not
            layer_node_embedding: Architecture of embedding layer for node attributes
            layer_edge_embedding: Architecture of embedding layer for edge attributes
            layer_state_embedding: Architecture of embedding layer for state attributes
            include_state: Whether the state embedding is included
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according to
                a Bernoulli distribution. Defaults to 0, i.e., no dropout.
    ########element_types: Elements included in the training set
            bond_expansion: Gaussian expansion for edge attributes
            cutoff: cutoff for forming bonds
            gauss_width: width of Gaussian function for bond expansion
            **kwargs: For future flexibility. Not used at the moment.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        self.cutoff = cutoff
        self.bond_expansion = BondExpansion(
            initial=0.0, final=cutoff+1, num_centers=dim_edge_embedding, width=gauss_width
        )
        self.bond_ICOHP_expansion = nn.Sequential(nn.Linear(1, dim_edge_embedding),
                                    nn.SiLU(),)
        self.bond_ICOBI_expansion = nn.Sequential(nn.Linear(1, dim_edge_embedding),
                                    nn.SiLU(),)
        
        self.node_plus_feats_dim=32
        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input] #  dim_edge_embedding*2
        edge_icoxx_dims = [dim_edge_embedding*2, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]

        self.node_plus_feats = nn.Sequential(nn.Linear(1, self.node_plus_feats_dim), nn.LayerNorm(self.node_plus_feats_dim), 
                                             nn.SiLU(), nn.Linear(32, 16), nn.SiLU()) #nn.Dropout(0.1))
        self.nodes_fusion=nn.Sequential(nn.Linear(node_dims[-1]+16, node_dims[-1]), 
                                        nn.SiLU(),) # nn.Dropout(0.1))

        activation: nn.Module = ActivationFunction(activation_type) #ActivationFunction.type[activation_type]

        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            #ntypes_node=len(self.element_types),
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.edge_encoder_len = MLP(edge_dims, activation, activate_last=True) # edge_dims*2
        self.edge_encoder_icoxx = MLP(edge_icoxx_dims, ActivationFunction('swish'), activate_last=True) # edge_dims*2
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]
        block_args = {
            "conv_hiddens": hidden_layer_sizes_conv,
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        # first block
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)] + [  # type: ignore
            MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args)  # type: ignore
            for _ in range(nblocks - 1)
        ]

        self.blocks = nn.ModuleList(blocks)

        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}
        self.edge_s2s_len = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)
        self.edge_s2s_icoxx = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)
        #self.edge_s2s = EdgeSet2Set(2*dim_blocks_out, **s2s_kwargs)
        self.node_s2s = Set2Set(dim_blocks_out, **s2s_kwargs)

        self.output_proj = MLP(
            # S2S cats q_star to output producing double the dim
            #dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            dims=[2 * 3 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation,
            activate_last=False,
        )

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.is_classification = is_classification
        self.include_state_embedding = include_state

    def forward(self, g: dgl.DGLGraph, state_attr: torch.Tensor | None = None, **kwargs):
        """Forward pass of MEGnet. Executes all blocks.

        Args:
            g (dgl.DGLGraph): DGL graphs
            state_attr (torch.Tensor): State attributes
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            Prediction
        """
        g = g.local_var()
        node_attr = g.ndata["node_feats"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist
        bond_len_attr = self.bond_expansion(g.edata["bond_dist"]); #[num_edges, dim_edge_emb] 

        #bond_icoxx_exp = self.bond_ICOXX_expansion(g.edata["norm_bond_strengths"]); #[num_edges, dim_edge_emb]
        #print(g.edata["bond_strengths"].shape, g.edata["bond_strengths"]); exit()
        #coxx_attr = g.edata["bond_strengths"].unsqueeze(dim=-1)
        #bond_icoxx_attr = self.bond_ICOXX_expansion(coxx_attr).squeeze(); #[num_edges, dim_edge_emb]
        cohp_attr = g.edata["ICOHPs"].unsqueeze(dim=-1)
        cobi_attr = g.edata["ICOBIs"].unsqueeze(dim=-1)
        bond_icohp_attr = self.bond_ICOHP_expansion(cohp_attr).squeeze(); #[num_edges, dim_edge_emb]
        bond_icobi_attr = self.bond_ICOBI_expansion(cobi_attr).squeeze(); #[num_edges, dim_edge_emb]
        bond_icoxx_attr = torch.cat([bond_icohp_attr, bond_icobi_attr], -1); #[num_edges, 2*dim_edge_emb]
        
        node_feat, edge_len_feat, state_feat = self.embedding(node_attr=node_attr, edge_attr=bond_len_attr, state_attr=state_attr)
        edge_len_feat = self.edge_encoder_len(edge_len_feat); #print(edge_feat.shape); exit()
        edge_icoxx_feat = self.edge_encoder_icoxx(bond_icoxx_attr) # [edges, hidden_layer_sizes_input[-1]]
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)

        #node_plus_feat=self.node_plus_feats(torch.hstack([g.ndata['norm_chg'], g.ndata['norm_dos_centers']])) #g.ndata['norm_asym_ind_icohp'], g.ndata['norm_asym_ind_icobi']]))
        node_plus_feat=self.node_plus_feats(g.ndata['chg'])
        node_feat=torch.hstack([node_feat, node_plus_feat])
        node_feat=self.nodes_fusion(node_feat)

        for block in self.blocks:
            output = block(graph=g,
                        edge_len_feat = edge_len_feat,
                        edge_icoxx_feat = edge_icoxx_feat,
                        node_feat = node_feat,
                        state_feat = state_feat,)
            edge_len_feat, edge_icoxx_feat, node_feat, state_feat = output

        #print(node_feat.shape, torch.hstack([edge_len_feat, edge_icoxx_feat]).shape)
        node_vec = self.node_s2s(g, node_feat)

        #edge_vec = self.edge_s2s(g, edge_len_feat)
        edge_vec_len = self.edge_s2s_len(g, edge_len_feat)
        edge_vec_coxx = self.edge_s2s_icoxx(g, edge_icoxx_feat)
        #edge_vec = self.edge_s2s(g, torch.hstack([edge_len_feat, edge_icoxx_feat]))

        #print(node_vec.shape, edge_vec.shape, edge_len_feat.shape, torch.cat([edge_len_feat, edge_icoxx_feat], -1).shape, torch.hstack([edge_len_feat, edge_icoxx_feat]).shape)
        node_vec = torch.squeeze(node_vec)
        #edge_vec = torch.squeeze(edge_vec)
        edge_vec_len = torch.squeeze(edge_vec_len)
        edge_vec_coxx = torch.squeeze(edge_vec_coxx)
        state_feat = torch.squeeze(state_feat)

        vec = torch.hstack([node_vec, edge_vec_len, edge_vec_coxx, state_feat])

        if self.dropout:
            vec = self.dropout(vec)  # pylint: disable=E1102

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        return torch.squeeze(output)

