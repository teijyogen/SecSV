

cdef extern from "boost/multiprecision/random.hpp" namespace "boost::random" nogil:
    # random number generator
    cdef cppclass mt19937:
        #init
        mt19937() nogil
        #attributes

        #methods
        seed(unsigned long)


cdef extern from "rng_wrapper.hpp" nogil:
    # wrapper to distributions ...
    cdef cppclass rng_sampler[result_type]:
        #init
        rng_sampler(mt19937) nogil
        rng_sampler()  nogil
        # methods (gamma and exp are using rate param)
        result_type uniform(unsigned int) nogil
        result_type uniform_array(result_type*, unsigned int, unsigned int) nogil


ctypedef mt19937 rng
