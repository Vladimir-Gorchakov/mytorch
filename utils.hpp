#pragma once
#include <sstream>
#include <vector>

namespace torch::utils {
template <typename T>
std::string vec_to_string(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i)
            oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}
} // namespace torch::utils
