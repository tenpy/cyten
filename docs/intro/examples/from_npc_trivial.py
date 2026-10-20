"""TeNPy ``np_conserved`` trivial-array example, rewritten for LASyTen.

Compare with ``examples/userguide/a_npc_arrays_triv.py`` in TeNPy.
"""

import numpy as np

import lasyten as lasy

leg = lasy.ElementarySpace.from_trivial_sector(2)
M = lasy.tensor([[0.0, 1.0], [1.0, 0.0]], codomain=[leg], domain=[leg], labels=['i', 'j'])
v = lasy.tensor([3.0, 4.0 + 1.0j], codomain=[leg], labels=['i'])
print('|v> =', v.to_numpy())
# |v> = [3.+0.j 4.+1.j]

# Contract as maps: M.domain == v.codomain.
M_v = M @ v
print('M|v> =', M_v.to_numpy())
# M|v> = [4.+1.j 3.+0.j]

# Same contraction by explicit legs (the tensordot analogue).
M_v2 = lasy.tdot(M, v, 'j', 'i')
np.testing.assert_allclose(M_v.to_numpy(), M_v2.to_numpy())

# Frobenius inner product  <v|M|v> = Tr[v^dagger @ (M @ v)].
print('<v|M|v> =', lasy.inner(v, M_v).to_numpy())
# <v|M|v> = (24+0j)
np.testing.assert_allclose(lasy.inner(v, M_v).to_numpy(), 24.0 + 0.0j)
