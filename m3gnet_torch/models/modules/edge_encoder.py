"""
Ref:
    - https://github.com/mir-group/nequip
    - https://www.nature.com/articles/s41467-022-29939-5
"""

import torch
from torch import nn


class SmoothBesselBasis(nn.Module):

        def __init__(self, r_max, max_n=10):
            r"""Smooth Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
            This is an orthogonal basis with first
            and second derivative at the cutoff
            equals to zero. The function was derived from the order 0 spherical Bessel
            function, and was expanded by the different zero roots
            Ref:
                https://arxiv.org/pdf/1907.02374.pdf
            Args:
                r_max: torch.Tensor distance tensor
                max_n: int, max number of basis, expanded by the zero roots
            Returns: expanded spherical harmonics with derivatives smooth at boundary
            Parameters
            ----------
            """
            super(SmoothBesselBasis, self).__init__()
            self.max_n = max_n
            n = torch.arange(0, max_n).float()[None, :]
            PI = 3.1415926535897
            SQRT2 = 1.41421356237
            fnr = (
                    (-1) ** n
                    * SQRT2
                    * PI
                    / r_max ** 1.5
                    * (n + 1)
                    * (n + 2)
                    / torch.sqrt(2 * n ** 2 + 6 * n + 5)
            )
            en = n ** 2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
            dn = [torch.tensor(1.0).float()]
            for i in range(1, max_n):
                dn.append(1 - en[0, i] / dn[-1])
            dn = torch.stack(dn)
            self.register_buffer("dn", dn)
            self.register_buffer("en", en)
            self.register_buffer("fnr_weights", fnr)
            self.register_buffer("n_1_pi_cutoff", ((torch.arange(0, max_n).float() + 1) * PI/ r_max).reshape(1, -1))
            self.register_buffer("n_2_pi_cutoff", ((torch.arange(0, max_n).float() + 2) * PI/ r_max).reshape(1, -1))
            self.register_buffer("r_max", torch.tensor(r_max))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Evaluate Smooth Bessel Basis for input x.

            Parameters
            ----------
            x : torch.Tensor
                Input
            """
            x_1 = x.unsqueeze(-1) * self.n_1_pi_cutoff
            x_2 = x.unsqueeze(-1) * self.n_2_pi_cutoff
            fnr = self.fnr_weights * (torch.sin(x_1) / x_1 + torch.sin(x_2) / x_2)
            gn = [fnr[:, 0]]
            for i in range(1, self.max_n):
                gn.append(1 / torch.sqrt(self.dn[i]) * (fnr[:, i] + torch.sqrt(self.en[0, i] / self.dn[i - 1]) * gn[-1]))
            return torch.transpose(torch.stack(gn), 1, 0)