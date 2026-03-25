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

    for (size_t i = 0; i < 5; i++)
        t[i] = (int)i * 3;
    REQUIRE(t[4] == 12);
    REQUIRE(t({4}) == 12);
}

TEST_CASE("Tensor multi-dim indexing via operator()") {
    torch::Tensor<int> t{{3, 4}};
    for (size_t i = 0; i < t.numel(); i++)
        t[i] = (int)i;

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

    REQUIRE_THROWS_AS(t({2, 0}), std::out_of_range);    // row out of range
    REQUIRE_THROWS_AS(t({0, 3}), std::out_of_range);    // col out of range
    REQUIRE_THROWS_AS(t({0, 0, 0}), std::out_of_range); // rank mismatch
    REQUIRE_THROWS_AS(t({0}), std::out_of_range); // rank mismatch (too few)
}

TEST_CASE("Tensor broadcast operations") {
    torch::Tensor<float> a{{2, 3}};
    torch::Tensor<float> b{{1, 3}};

    // a = [[1,2,3],[4,5,6]], b = [[1,2,3]]
    for (size_t i = 0; i < a.numel(); i++)
        a[i] = (float)(i + 1);
    for (size_t i = 0; i < b.numel(); i++)
        b[i] = (float)(i + 1);

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

// ── compound assignment operators ────────────────────────────────────────────

TEST_CASE("operator+= tensor") {
    torch::Tensor<int> a{{2, 3}};
    torch::Tensor<int> b{{2, 3}};
    for (size_t i = 0; i < a.numel(); i++) {
        a[i] = (int)i + 1;
        b[i] = (int)i * 2;
    }

    std::vector<int> expected(a.numel());
    for (size_t i = 0; i < a.numel(); i++)
        expected[i] = a[i] + b[i];

    a += b;

    for (size_t i = 0; i < a.numel(); i++)
        CHECK(a[i] == expected[i]);
}

TEST_CASE("operator+= scalar") {
    torch::Tensor<int> a{{2, 3}};
    for (size_t i = 0; i < a.numel(); i++)
        a[i] = (int)i;

    a += 10;

    for (size_t i = 0; i < a.numel(); i++)
        CHECK(a[i] == (int)i + 10);
}

TEST_CASE("operator*= tensor") {
    torch::Tensor<int> a{{2, 3}};
    torch::Tensor<int> b{{2, 3}};
    for (size_t i = 0; i < a.numel(); i++) {
        a[i] = (int)i + 1;
        b[i] = (int)i + 2;
    }

    std::vector<int> expected(a.numel());
    for (size_t i = 0; i < a.numel(); i++)
        expected[i] = a[i] * b[i];

    a *= b;

    for (size_t i = 0; i < a.numel(); i++)
        CHECK(a[i] == expected[i]);
}

TEST_CASE("operator*= scalar") {
    torch::Tensor<int> a{{2, 3}};
    for (size_t i = 0; i < a.numel(); i++)
        a[i] = (int)i + 1;

    a *= 3;

    for (size_t i = 0; i < a.numel(); i++)
        CHECK(a[i] == ((int)i + 1) * 3);
}

TEST_CASE("operator/= tensor") {
    torch::Tensor<int> a{{2, 3}};
    torch::Tensor<int> b{{2, 3}};
    for (size_t i = 0; i < a.numel(); i++) {
        a[i] = ((int)i + 1) * 6;
        b[i] = (int)i + 1;
    }

    std::vector<int> expected(a.numel());
    for (size_t i = 0; i < a.numel(); i++)
        expected[i] = a[i] / b[i];

    a /= b;

    for (size_t i = 0; i < a.numel(); i++)
        CHECK(a[i] == expected[i]);
}

TEST_CASE("operator/= scalar") {
    torch::Tensor<int> a{{2, 3}};
    for (size_t i = 0; i < a.numel(); i++)
        a[i] = ((int)i + 1) * 4;

    a /= 2;

    for (size_t i = 0; i < a.numel(); i++)
        CHECK(a[i] == ((int)i + 1) * 2);
}

TEST_CASE("compound assignment returns reference to self") {
    torch::Tensor<int> a{{2, 2}};
    torch::Tensor<int> b{{2, 2}};
    a.fill_(4);
    b.fill_(2);

    REQUIRE(&(a += b) == &a);
    REQUIRE(&(a *= b) == &a);
    REQUIRE(&(a /= b) == &a);
    REQUIRE(&(a += 1) == &a);
    REQUIRE(&(a *= 2) == &a);
    REQUIRE(&(a /= 2) == &a);
}

TEST_CASE("operator+= with broadcasting") {
    torch::Tensor<int> a{{2, 3}};
    torch::Tensor<int> b{{1, 3}};
    for (size_t i = 0; i < a.numel(); i++)
        a[i] = (int)i + 1;
    for (size_t i = 0; i < b.numel(); i++)
        b[i] = (int)(i + 1) * 10;

    // a = [[1,2,3],[4,5,6]], b = [[10,20,30]]
    a += b;

    CHECK(a({0, 0}) == 11);
    CHECK(a({0, 1}) == 22);
    CHECK(a({0, 2}) == 33);
    CHECK(a({1, 0}) == 14);
    CHECK(a({1, 1}) == 25);
    CHECK(a({1, 2}) == 36);
}

// ── transpose ────────────────────────────────────────────────────────────────

TEST_CASE("Transpose swaps shape and strides") {
    torch::Tensor<int> t{{3, 4}};
    auto t_T = t.transpose(0, 1);

    REQUIRE(t_T.shape() == torch::shape_t{4, 3});
    REQUIRE(t_T.strides() == torch::shape_t{1, 4});
}

TEST_CASE("Transpose 3D swaps correct dims") {
    torch::Tensor<int> t{{2, 3, 4}};
    // original strides: {12, 4, 1}
    auto t_T = t.transpose(0, 2);

    REQUIRE(t_T.shape() == torch::shape_t{4, 3, 2});
    REQUIRE(t_T.strides() == torch::shape_t{1, 4, 12});
}

TEST_CASE("Transpose makes tensor non-contiguous") {
    torch::Tensor<int> t{{3, 4}};
    REQUIRE(t.is_contiguous());

    auto t_T = t.transpose(0, 1);
    REQUIRE_FALSE(t_T.is_contiguous());
}

TEST_CASE("Transpose shares storage — write visible through both views") {
    torch::Tensor<int> t{{3, 4}};
    for (size_t i = 0; i < t.numel(); i++)
        t[i] = (int)i;

    auto t_T = t.transpose(0, 1);

    // t(1,2) == 6;  after transpose that element is at t_T(2,1)
    REQUIRE(t({1, 2}) == t_T({2, 1}));
}

TEST_CASE("Transpose out of range throws") {
    torch::Tensor<int> t{{3, 4}};
    REQUIRE_THROWS_AS(t.transpose(0, 2), std::out_of_range);
    REQUIRE_THROWS_AS(t.transpose(5, 0), std::out_of_range);
}

// ── contiguous ───────────────────────────────────────────────────────────────

TEST_CASE("contiguous on already-contiguous tensor returns same-storage view") {
    torch::Tensor<int> t{{3, 4}};
    auto c = t.contiguous();

    REQUIRE(c.is_contiguous());
    REQUIRE(c.data_ptr() == t.data_ptr());
}

TEST_CASE("contiguous on transposed tensor returns contiguous copy") {
    torch::Tensor<int> t{{3, 4}};
    for (size_t i = 0; i < t.numel(); i++)
        t[i] = (int)i;

    auto t_T = t.transpose(0, 1);
    auto c = t_T.contiguous();

    REQUIRE(c.is_contiguous());
    REQUIRE(c.shape() == torch::shape_t{4, 3});

    // Every element must match the transposed logical view
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 3; j++)
            REQUIRE(c({i, j}) == t_T({i, j}));
}

// ── reshape ──────────────────────────────────────────────────────────────────

TEST_CASE("Reshape preserves numel and values") {
    torch::Tensor<int> t{{2, 6}};
    for (size_t i = 0; i < t.numel(); i++)
        t[i] = (int)i;

    torch::shape_t new_shape{3, 4};
    auto r = t.reshape(new_shape);

    REQUIRE(r.shape() == torch::shape_t{3, 4});
    REQUIRE(r.numel() == 12u);

    for (size_t i = 0; i < r.numel(); i++)
        REQUIRE(r[i] == (int)i);
}

TEST_CASE("Reshape to 1D") {
    torch::Tensor<int> t{{3, 4}};
    for (size_t i = 0; i < t.numel(); i++)
        t[i] = (int)i;

    torch::shape_t flat{12};
    auto r = t.reshape(flat);

    REQUIRE(r.shape() == torch::shape_t{12});
    REQUIRE(r.ndim() == 1u);
    REQUIRE(r[11] == 11);
}

TEST_CASE("Reshape incompatible numel throws") {
    torch::Tensor<int> t{{2, 6}};
    torch::shape_t bad{3, 5}; // 15 != 12
    REQUIRE_THROWS(t.reshape(bad));
}

// ── fill_ ───────────────────────────────────────────────────────────────────

TEST_CASE("fill_ sets all elements to value") {
    torch::Tensor<int> t{{2, 3}};
    t.fill_(42);

    for (size_t i = 0; i < t.numel(); i++)
        CHECK(t[i] == 42);
}

TEST_CASE("fill_ works on 1D tensor") {
    torch::Tensor<float> t{{5}};
    t.fill_(3.14f);

    for (size_t i = 0; i < t.numel(); i++)
        CHECK(t[i] == 3.14f);
}

TEST_CASE("fill_ works on transposed (non-contiguous) tensor") {
    torch::Tensor<int> t{{3, 4}};
    for (size_t i = 0; i < t.numel(); i++)
        t[i] = (int)i;

    auto t_T = t.transpose(0, 1); // shape {4,3}, non-contiguous
    t_T.fill_(7);

    // all elements of the transposed view should be 7
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 3; j++)
            CHECK(t_T({i, j}) == 7);

    // original tensor shares storage, so it should also be all 7s
    for (size_t i = 0; i < t.numel(); i++)
        CHECK(t[i] == 7);
}

TEST_CASE("fill_ returns reference to self") {
    torch::Tensor<int> t{{2, 2}};
    auto& ref = t.fill_(5);
    REQUIRE(&ref == &t);
}

// ── autodiff ────────────────────────────────────────────────────────────────

TEST_CASE("autodiff single value test") {
    torch::Tensor<float> a{{1}};
    a.fill_(2.);

    auto b = a * a;
    auto c = b * b;
    c.backward();

    REQUIRE((*(a.grad_))[0] == 32.);
}
