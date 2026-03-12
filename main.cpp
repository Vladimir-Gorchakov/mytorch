#include <iostream>
#include <cassert>
#include "tensor.hpp"
#include "tensor_iterator.hpp"

template<typename T>
void addition(std::span<T*> ptrs) {
    (*ptrs[2]) = (*ptrs[1]) + (*ptrs[0]);
}


int main() {
    // Tensor test
    torch::Tensor<float> t({2,3,4});

    assert(t.dim() == 3);
    assert(t.size(1) == 3);

    // TensorIterator test
    torch::Tensor<float> A{{2,2}};
    torch::Tensor<float> B{{2,2}};
    torch::Tensor<float> C{{1,2}};

    C[{0,0}] = 5;
    C[{0,1}] = 6;

    B[{0,0}] = 1;
    B[{0,1}] = 2;
    B[{1,0}] = 3;
    B[{1,1}] = 4;

    float* ptr = B.data_ptr();

    for (size_t i = 0; i < 4; i++) {
        assert((*(ptr + i) == (float)i + 1));
    }


    torch::iterator::TensorIterator<float> iter{{&B, &C}, {&A}};
    torch::shape_t shape{2, 2};

    assert((shape == iter.shape()));

    iter.run(addition<float>);

    assert((A[{0,0}] == 6));
    assert((A[{0,1}] == 8));
    assert((A[{1,0}] == 8));
    assert((A[{1,1}] == 10));

    std::cout << "All tests passed!" << std::endl;
}
