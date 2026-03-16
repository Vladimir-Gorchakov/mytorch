#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "tensor.hpp"
#include "tensor_iterator.hpp"

using size_t = std::size_t;

TEST_CASE("Tensor test") {
    torch::Tensor<int> t{{2, 3, 4}};

    REQUIRE(t.ndim() == 3);
    REQUIRE(t.size(1) == 3);
}

TEST_CASE("TensorIterator test") {
    torch::Tensor<float> A{{2, 3}};
    torch::Tensor<float> B{{2, 3}};
    torch::Tensor<float> C{{1, 3}};

    torch::iterator::TensorIterator<float> iter{{&B, &C}, {&A}};
    torch::shape_t shape{2, 3};

    REQUIRE((shape == iter.shape()));

    auto srides_C = iter.compute_strides(C.strides(), C.shape());
    auto srides_B = iter.compute_strides(B.strides(), B.shape());

    REQUIRE(srides_C.size() == 2u);
    REQUIRE(srides_B.size() == 2u);

    REQUIRE(srides_C == torch::shape_t{0, 1});
    REQUIRE(srides_B == torch::shape_t{3, 1});
}

TEST_CASE("Tensor operations test") {
    torch::Tensor<int> t1{{2, 3, 4}};
    torch::Tensor<int> t2{{2, 3, 4}};

    for (size_t i = 0; i < t1.numel(); i++) {
        t1[i] = t2[i] = (int)i;
    }

    torch::Tensor<int> res1 = t1 * t2;

    for (size_t i = 0; i < t1.numel(); i++) {
        CHECK(t1[i] * t2[i] == res1[i]);
    }

    torch::Tensor<int> res2 = t1 * 2;
    for (size_t i = 0; i < t1.numel(); i++) {
        CHECK(t1[i] * 2 == res2[i]);
    }
}
