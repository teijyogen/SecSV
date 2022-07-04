# distutils: language = c++
#cython: boundscheck=False, wraparound=False, nonecheck=False

import numpy as np
cimport numpy as np
cimport cython
from cython.operator cimport dereference as deref
cimport random128

ctypedef long double float_type

# @cython.boundscheck(False)
def random_array(size, n_bits):

    cdef unsigned int N = size
    cdef unsigned int N_bits = n_bits
    cdef rng_sampler[float_type] *rng_p = new rng_sampler[float_type]()
    cdef rng_sampler[float_type] rng = deref(rng_p)
    cdef np.ndarray[float_type, ndim=1] result = np.zeros(N, dtype=np.float128)

    # for i in range(N):
    #     result[i] = rng.uniform(N_bits)
    rng.uniform_array(<float_type*> result.data, N_bits, N)

    return result