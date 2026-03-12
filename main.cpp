#include <iostream>
#include <cassert>
#include "tensor.hpp"


int main() {
    torch::Tensor<float> t({2,3,4});

    torch::iterator::Operand<float>{t};

    assert(t.dim() == 3);
    assert(t.size(1) == 3);

    std::cout << "All tests passed!" << std::endl;
}
