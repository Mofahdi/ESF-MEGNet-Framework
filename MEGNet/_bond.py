"""Generate bond features based on spherical bessel functions or gaussian expansion."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

#from matgl.layers._basis import GaussianExpansion

class GaussianExpansion(nn.Module):
    """Gaussian Radial Expansion.

    The bond distance is expanded to a vector of shape [m], where m is the number of Gaussian basis centers.
    """

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 4.0,
        num_centers: int = 20,
        width: None | float = 0.5,
    ):
        """
        Args:
            initial: Location of initial Gaussian basis center.
            final: Location of final Gaussian basis center
            num_centers: Number of Gaussian Basis functions
            width: Width of Gaussian Basis functions.
        """
        super().__init__()
        self.centers = nn.Parameter(torch.linspace(initial, final, num_centers), requires_grad=False)  # type: ignore
        if width is None:
            self.width = 1.0 / torch.diff(self.centers).mean()
        else:
            self.width = torch.as_tensor(width)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.centers = nn.Parameter(self.centers, requires_grad=False)

    def forward(self, bond_dists):
        """Expand distances.

        Args:
            bond_dists :
                Bond (edge) distances between two atoms (nodes)

        Returns:
            A vector of expanded distance with shape [num_centers]
        """
        diff = bond_dists[:, None] - self.centers[None, :]
        return torch.exp(-self.width * (diff**2))
    


class BondExpansion(nn.Module):
    """Expand pair distances into a set of spherical bessel or gaussian functions."""

    def __init__(
        self,
        cutoff: float = 5.0,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 100,
        width: float = 0.5,
    ) -> None:
        """
        Args:
            max_l (int): order of angular part
            max_n (int): order of radial part
            cutoff (float): cutoff radius
            rbf_type (str): type of radial basis function .i.e. either "SphericalBessel", "ExpNorm" or 'Gaussian'
            smooth (bool): whether apply the smooth version of spherical bessel functions or not
            initial (float): initial point for gaussian expansion
            final (float): final point for gaussian expansion
            num_centers (int): Number of centers for gaussian expansion.
            width (float): width of gaussian function.
        """
        super().__init__()

        self.cutoff = cutoff
        self.num_centers = num_centers
        self.width = width
        self.initial = initial
        self.final = final

        self.rbf = GaussianExpansion(initial, final, num_centers, width)  # type:ignore[assignment]

    def forward(self, bond_dist: torch.Tensor):
        """Forward.

        Args:
        bond_dist: Bond distance

        Return:
        bond_basis: Radial basis functions
        """
        bond_basis = self.rbf(bond_dist)
        return bond_basis
