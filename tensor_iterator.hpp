#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include "tensor.hpp"
#include "utils.hpp"

namespace torch::exception {
class shape_exception;
}

namespace torch::iterator {
using size_t = std::size_t;
using shape_t = std::vector<size_t>;

template <typename T>
class Operand {
  public:
    T* data_ptr_;

    shape_t strides_;

    Operand(T* data_ptr, shape_t strides)
        : data_ptr_(data_ptr),
          strides_(strides) {}
};

template <typename T>
class TensorIterator {
  private:
    shape_t shape_;

    std::vector<Operand<const T>> inputs_;
    std::vector<Operand<T>> outputs_;

  public:
    TensorIterator(const std::vector<const Tensor<T>*>& inputs,
                   const std::vector<Tensor<T>*>& outputs);

    size_t numel() const;

    static shape_t broadcast_shape(const std::vector<const Tensor<T>*>& inputs);
    static shape_t broadcast_shape(const std::vector<const Tensor<T>*>& inputs, const shape_t& shape);

    shape_t compute_strides(shape_t strides, shape_t shape) const;

    shape_t shape();

    void _move_ptrs(std::vector<T*>& ptrs, std::vector<shape_t*>& strides,
                    int step, size_t dim);

    void _move_ptrs(std::vector<const T*>& ptrs, std::vector<shape_t*>& strides,
                    int step, size_t dim);

    void run(std::function<void(std::span<const T*>, std::span<T*>)> kernel);
};

template <typename T>
TensorIterator<T>::TensorIterator(const std::vector<const Tensor<T>*>& inputs,
                                  const std::vector<Tensor<T>*>& outputs) {
    if (inputs.empty()) {
        assert((!outputs.empty()));
        shape_ = outputs[0]->shape();
    } else {
        shape_ = broadcast_shape(inputs);
    }

    for (auto& tensor : inputs) {
        inputs_.emplace_back(tensor->data_ptr(),
                             compute_strides(tensor->strides(),
                                             tensor->shape()));
    }

    for (auto& tensor : outputs) {
        if (!(shape_ == tensor->shape())) {
            throw exception::shape_exception{
                std::format("Shape exception: output tensor shape must be {} "
                            "but got {}",
                            utils::vec_to_string(shape_),
                            utils::vec_to_string(tensor->shape()))};
        }

        outputs_.emplace_back(tensor->data_ptr(), tensor->strides());
    }
}

template <typename T>
size_t TensorIterator<T>::numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1u,
                           std::multiplies<size_t>());
}

template <typename T>
shape_t TensorIterator<T>::broadcast_shape(
    const std::vector<const Tensor<T>*>& inputs) {
    size_t shape_size = 0;
    for (auto tensor : inputs) {
        shape_size = std::max(shape_size, tensor->shape().size());
    }

    shape_t shape(shape_size);
    std::fill(shape.begin(), shape.end(), 1);

    return broadcast_shape(inputs, shape);
}

template <typename T>
shape_t TensorIterator<T>::broadcast_shape(const std::vector<const Tensor<T>*>& inputs,
                                           const shape_t& shape) {
    for (size_t i = 1; i <= shape.size(); ++i) {
        for (auto tensor : inputs) {
            if (i > tensor->shape().size()) {
                continue;
            }

            size_t t_i = tensor->shape().size() - i;
            size_t s_i = shape.size() - i;

            if (shape[s_i] != 1 && tensor->shape()[t_i] != 1 &&
                shape[s_i] != tensor->shape()[t_i]) [[unlikely]] {
                throw exception::shape_exception{
                    std::format("Shape exception: output tensor shape at index "
                                "{} "
                                "shuld be the same but got {} and {}",
                                i, shape[s_i], tensor->shape()[t_i])};
            }

            shape[s_i] =
                (tensor->shape()[t_i] == 1) ? shape[s_i] : tensor->shape()[t_i];
        }
    }

    return shape;
}

template <typename T>
shape_t TensorIterator<T>::compute_strides(shape_t strides,
                                           shape_t shape) const {
    shape_t new_strides(shape_.size());

    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i <
            shape_.size() -
                strides
                    .size()) { // Гарантированно strides.size() <= shape_.size()
            continue;
        }

        size_t corrected_idx = i - (shape_.size() - strides.size());

        new_strides[i] =
            shape[corrected_idx] != 1u ? strides[corrected_idx] : 0;
    }

    return new_strides;
}

template <typename T>
shape_t TensorIterator<T>::shape() {
    return shape_;
}

template <typename T>
void TensorIterator<T>::_move_ptrs(std::vector<const T*>& ptrs,
                                   std::vector<shape_t*>& strides, int step,
                                   size_t dim) {
    for (size_t i = 0; i < ptrs.size(); ++i) {
        ptrs[i] += step * (int)(*strides[i])[dim];
    }
}

template <typename T>
void TensorIterator<T>::_move_ptrs(std::vector<T*>& ptrs,
                                   std::vector<shape_t*>& strides, int step,
                                   size_t dim) {
    for (size_t i = 0; i < ptrs.size(); ++i) {
        ptrs[i] += step * (int)(*strides[i])[dim];
    }
}

template <typename T>
void TensorIterator<T>::run(
    std::function<void(std::span<const T*>, std::span<T*>)> kernel) {
    std::vector<const T*> ptrs_input{};
    std::vector<shape_t*> strides_input{};

    std::vector<T*> ptrs_output{};
    std::vector<shape_t*> strides_output{};

    for (auto& operand : inputs_) {
        ptrs_input.push_back(operand.data_ptr_);
        strides_input.push_back(&operand.strides_);
    }

    for (auto& operand : outputs_) {
        ptrs_output.push_back(operand.data_ptr_);
        strides_output.push_back(&operand.strides_);
    }

    size_t index = 0;

    for (size_t i = 0; i < numel(); ++i) {
        index = i;
        for (size_t j = shape_.size(); j-- > 0;) {
            if (index == 0) [[unlikely]] {
                break;
            }

            if (index % shape_[j] == 0) {
                index /= shape_[j];
                _move_ptrs(ptrs_input, strides_input, -(int)shape_[j] + 1, j);
                _move_ptrs(ptrs_output, strides_output, -(int)shape_[j] + 1, j);
            } else {
                _move_ptrs(ptrs_input, strides_input, 1, j);
                _move_ptrs(ptrs_output, strides_output, 1, j);
                break;
            }
        }

        kernel(std::span<const T*>(ptrs_input), std::span<T*>(ptrs_output));
    }
}

template <typename T>
class TensorIteratorConfig {
  private:
    std::vector<const Tensor<T>*> inputs_;
    std::vector<Tensor<T>*> outputs_;

  public:
    TensorIteratorConfig& add_input(const Tensor<T>& tensor) {
        inputs_.push_back(&tensor);
        return *this;
    }

    TensorIteratorConfig& add_output(Tensor<T>& tensor) {
        outputs_.push_back(&tensor);
        return *this;
    }

    TensorIterator<T> build() { // требует улучшения, пока не знаю как для
                                // общего случая должны выглядить шейпы

        if (outputs_[0]->empty()) {
            assert((!inputs_.empty()));
            shape_t shape = TensorIterator<T>::broadcast_shape(inputs_);
            *outputs_[0] = Tensor<T>{shape};
        }

        return TensorIterator<T>{inputs_, outputs_};
    }
};

} // namespace torch::iterator
