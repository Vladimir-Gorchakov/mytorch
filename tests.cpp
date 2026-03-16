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
        t1[i] = t2[i] = (int)i + 1;
    }

    torch::Tensor<int> res1 = t1 * t2;

    for (size_t i = 0; i < t1.numel(); i++) {
        CHECK(t1[i] * t2[i] == res1[i]);
    }

    torch::Tensor<int> res2 = t1 * 2;

    for (size_t i = 0; i < t1.numel(); i++) {
        CHECK(t1[i] * 2 == res2[i]);
    }

    torch::Tensor<int> res3 = t1 + t2;

    for (size_t i = 0; i < t1.numel(); i++) {
        CHECK(t1[i] + t2[i] == res3[i]);
    }

    torch::Tensor<int> res4 = t1 + 2;

    for (size_t i = 0; i < t1.numel(); i++) {
        CHECK(t1[i] + 2 == res4[i]);
    }

    torch::Tensor<int> res5 = t1 / t2;

    for (size_t i = 0; i < t1.numel(); i++) {
        CHECK(t1[i] / t2[i] == res5[i]);
    }

    torch::Tensor<int> res6 = t1 / 2;

    for (size_t i = 0; i < t1.numel(); i++) {
        CHECK(t1[i] / 2 == res6[i]);
    }
}

TEST_CASE("Tensor numel") {
    REQUIRE(torch::Tensor<int>{{2, 3, 4}}.numel() == 24u);
    REQUIRE(torch::Tensor<int>{{5}}.numel() == 5u);
    REQUIRE(torch::Tensor<int>{{1, 1, 1}}.numel() == 1u);
}

TEST_CASE("Tensor strides are row-major") {
    torch::Tensor<int> t{{2, 3, 4}};
    REQUIRE(t.strides() == torch::shape_t{12, 4, 1});

    torch::Tensor<int> t2{{5}};
    REQUIRE(t2.strides() == torch::shape_t{1});
}

TEST_CASE("Tensor empty") {
    torch::Tensor<float> empty{};
    REQUIRE(empty.empty());
    REQUIRE(empty.ndim() == 0u);

    torch::Tensor<float> nonempty{{2, 3}};
    REQUIRE_FALSE(nonempty.empty());
}

TEST_CASE("Tensor is_contiguous") {
    torch::Tensor<int> t{{2, 3, 4}};
    REQUIRE(t.is_contiguous());
}

TEST_CASE("Tensor 1D") {
    torch::Tensor<int> t{{5}};
    REQUIRE(t.ndim() == 1u);
    REQUIRE(t.size(0) == 5u);
    REQUIRE(t.numel() == 5u);

    for (size_t i = 0; i < 5; i++) t[i] = (int)i * 3;
    REQUIRE(t[4] == 12);
    REQUIRE(t({4}) == 12);
}

TEST_CASE("Tensor multi-dim indexing via operator()") {
    torch::Tensor<int> t{{3, 4}};
    for (size_t i = 0; i < t.numel(); i++) t[i] = (int)i;

    // Row-major: (row, col) -> row*4 + col
    REQUIRE(t({0, 0}) == 0);
    REQUIRE(t({0, 3}) == 3);
    REQUIRE(t({2, 0}) == 8);
    REQUIRE(t({2, 3}) == 11);

    t({1, 2}) = 99;
    REQUIRE(t[6] == 99);
}

TEST_CASE("Tensor operator() bounds checking") {
    torch::Tensor<int> t{{2, 3}};

    REQUIRE_THROWS_AS(t({2, 0}), std::out_of_range);  // row out of range
    REQUIRE_THROWS_AS(t({0, 3}), std::out_of_range);  // col out of range
    REQUIRE_THROWS_AS(t({0, 0, 0}), std::out_of_range);  // rank mismatch
    REQUIRE_THROWS_AS(t({0}), std::out_of_range);  // rank mismatch (too few)
}

TEST_CASE("Tensor broadcast operations") {
    torch::Tensor<float> a{{2, 3}};
    torch::Tensor<float> b{{1, 3}};

    // a = [[1,2,3],[4,5,6]], b = [[1,2,3]]
    for (size_t i = 0; i < a.numel(); i++) a[i] = (float)(i + 1);
    for (size_t i = 0; i < b.numel(); i++) b[i] = (float)(i + 1);

    torch::Tensor<float> result = a + b;

    REQUIRE(result.shape() == torch::shape_t{2, 3});
    // row 0: 1+1, 2+2, 3+3
    CHECK(result[0] == 2.0f);
    CHECK(result[1] == 4.0f);
    CHECK(result[2] == 6.0f);
    // row 1: 4+1, 5+2, 6+3
    CHECK(result[3] == 5.0f);
    CHECK(result[4] == 7.0f);
    CHECK(result[5] == 9.0f);
}

TEST_CASE("TensorIterator incompatible shapes throw") {
    torch::Tensor<float> a{{2, 3}};
    torch::Tensor<float> b{{2, 4}};

    REQUIRE_THROWS(a + b);
}
