#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/random.hpp>

namespace mp = boost::multiprecision;
using namespace boost::random;

template<typename float_t=mp::number<mp::cpp_dec_float<0>>>
class rng_sampler {

public:
    typedef float_t result_type;
    typedef mp::cpp_int random_type;

    rng_sampler(mt19937& in_R) : R(in_R) {}
    rng_sampler() {
        R = mt19937();
        R.seed(std::clock());
    }

    result_type uniform(unsigned int n_bits) {
        uniform_int_distribution<random_type> ui(0, mp::cpp_int(1) << n_bits);
        return ui(R).convert_to<result_type>();
    }

    result_type uniform_array(result_type *arr, unsigned int n_bits, unsigned int size) {
        uniform_int_distribution<random_type> ui(0, mp::cpp_int(1) << n_bits);
        for (unsigned int i=0; i < size; ++i){
            arr[i] = ui(R).convert_to<result_type>();
        }
//        return arr;
    }

private:
    mt19937 R;

};




