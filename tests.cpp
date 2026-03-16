#include <catch2/catch_test_macros.hpp>

#include "tensor.hpp"
#include "tensor_iterator.hpp"

template <typename T>
void addition(std::span<T*> ptrs) {
    (*ptrs[2]) = (*ptrs[1]) + (*ptrs[0]);
}

TEST_CASE("Tensor test") {
    torch::Tensor<float> t({2, 3, 4});

    REQUIRE(t.ndim() == 3);
    REQUIRE(t.size(1) == 3);
}

TEST_CASE("TensorIterator test") {
    torch::Tensor<float> A{{2, 2}};
    torch::Tensor<float> B{{2, 2}};
    torch::Tensor<float> C{{1, 2}};

    C[{0, 0}] = 5;
    C[{0, 1}] = 6;

    B[{0, 0}] = 1;
    B[{0, 1}] = 2;
    B[{1, 0}] = 3;
    B[{1, 1}] = 4;

    float* ptr = B.data_ptr();

    for (size_t i = 0; i < 4; i++) {
        REQUIRE((*(ptr + i) == (float)i + 1));
    }

    torch::iterator::TensorIterator<float> iter{{&B, &C}, {&A}};
    torch::shape_t shape{2, 2};

    REQUIRE((shape == iter.shape()));

    iter.run(addition<float>);

    CHECK((A[{0, 0}] == 6));
    CHECK((A[{0, 1}] == 8));
    CHECK((A[{1, 0}] == 8));
    CHECK((A[{1, 1}] == 10));
}
