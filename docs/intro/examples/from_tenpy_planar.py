"""Two-site DMRG EffectiveH as a PlanarLinearOperator."""

import cyten as ct
from cyten.testing import random_tensor


class TwoSiteEffectiveH(ct.PlanarLinearOperator):
    """Effective Hamiltonian on two MPS sites.

    The operator is the planar network::

        |        .---       ---.
        |        |    |   |    |
        |       LP----W0--W1---RP
        |        |    |   |    |
        |        .---       ---.

    and ``matvec(theta)`` contracts a two-site wave function into the open
    ket legs.
    """

    op_diagram = ct.PlanarDiagram(
        tensors='Lp[vR*, wR, vR], W0[wL, p, wR, p*], W1[wL, p, wR, p*], Rp[vL*, vL, wL]',
        definition=(
            'Lp:vR* -> vL, Lp:wR @ W0:wL, Lp:vR -> vL*, '
            'W0:p -> p0, W0:wR @ W1:wL, W0:p* -> p0*, '
            'W1:p -> p1, W1:wR @ Rp:wL, W1:p* -> p1*, '
            'Rp:vL* -> vR, Rp:vL -> vR*'
        ),
        dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], w=['wL', 'wR'], d=['p', 'p*']),
    )
    matvec_diagram = op_diagram.add_tensor(
        tensor='theta[vL, p0, p1, vR]',
        extra_definition='theta:vL @ Lp:vR, theta:p0 @ W0:p*, theta:p1 @ W1:p*, theta:vR @ Rp:vL',
        extra_dims=dict(chi=['vL', 'vR'], d=['p0', 'p1']),
    )

    def __init__(self, Lp, W0, W1, Rp):
        super().__init__(
            op_diagram=self.op_diagram,
            matvec_diagram=self.matvec_diagram,
            op_tensors=dict(Lp=Lp, W0=W0, W1=W1, Rp=Rp),
            vec_name='theta',
        )


sym = ct.U1().as_Symmetry()
theta = random_tensor(sym, 4, labels=['vL', 'p0', 'p1', 'vR'], max_multiplicity=2, max_blocks=2)
vL, p0, p1, vR = theta.legs
Lp = random_tensor(sym, [vL, None, vL.dual], labels=['vR*', 'wR', 'vR'], max_multiplicity=2, max_blocks=2)
W0 = random_tensor(
    sym,
    [p0, None, p0.dual, Lp.get_leg('wR').dual],
    labels=['p', 'wR', 'p*', 'wL'],
    max_multiplicity=2,
    max_blocks=2,
)
W1 = random_tensor(
    sym,
    [p1, None, p1.dual, W0.get_leg('wR').dual],
    labels=['p', 'wR', 'p*', 'wL'],
    max_multiplicity=2,
    max_blocks=2,
)
Rp = random_tensor(sym, [vR, vR.dual, W1.get_leg('wR').dual], labels=['vL*', 'vL', 'wL'])

Heff = TwoSiteEffectiveH(Lp=Lp, W0=W0, W1=W1, Rp=Rp)
op = ct.planar_permute_legs(Heff.to_tensor(), codomain=['vL', 'p0', 'p1', 'vR'])
assert op.codomain_labels == ['vL', 'p0', 'p1', 'vR']
assert op.domain_labels == ['vL*', 'p0*', 'p1*', 'vR*']

H_theta = Heff.matvec(theta)
assert H_theta.codomain_labels == ['vL', 'p0', 'p1', 'vR']
assert ct.almost_equal(ct.compose(op, theta), H_theta)
