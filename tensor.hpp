#pragma once
#include <algorithm>
#include <format>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "autodiff.hpp"
#include "utils.hpp"

namespace torch::iterator {
template <typename T>
class TensorIteratorConfig;
}

namespace torch::exception {
class shape_exception {
  public:
    shape_exception(std::string message) : message{message} {}
    std::string getMessage() const { return message; }

  private:
    std::string message;
};

} // namespace torch::exception

namespace torch {
using size_t = std::size_t;
using shape_t = std::vector<size_t>;

template <typename T>
class Storage {
  public:
    T* data_;
    size_t size_;

    Storage(size_t size) : size_(size) { data_ = new T[size]; }

    ~Storage() { delete[] data_; }
};

template <typename T>
class Tensor {
  private:
    std::shared_ptr<Storage<T>> storage_;

    shape_t shape_;
    shape_t strides_;

    size_t storage_offset_;

  public:
    Tensor()
        : storage_(nullptr),
          shape_({}),
          strides_({}),
          storage_offset_(0) {}
    Tensor(const shape_t& shape);
    Tensor(std::shared_ptr<Storage<T>> storage, const shape_t& shape,
           shape_t strides, size_t storage_offset)
        : storage_(storage),
          shape_(shape),
          strides_(strides),
          storage_offset_(storage_offset){};

    T& operator()(const shape_t& index);
    const T& operator()(const shape_t& index) const;

    T& operator[](const size_t index) {
        return storage_->data_[index + storage_offset_];
    }
    const T& operator[](const size_t index) const {
        return storage_->data_[index + storage_offset_];
    }

    Tensor<T> operator*(const Tensor<T>& other) const;
    Tensor<T> operator*(const T& other) const;

    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator+(const T& other) const;

    Tensor<T> operator/(const Tensor<T>& other) const;
    Tensor<T> operator/(const T& other) const;

    bool empty() const { return storage_ == nullptr; }

    size_t numel() const;
    size_t ndim() const { return shape_.size(); }
    size_t size(size_t axis) const { return shape_[axis]; }
    size_t storage_offset() const { return storage_offset_; }

    const shape_t& shape() const { return shape_; }
    const shape_t& strides() const { return strides_; }

    T* data_ptr() { return storage_->data_ + storage_offset_; }
    const T* data_ptr() const { return storage_->data_ + storage_offset_; }

    shape_t compute_strides(const shape_t& shape) const;
    size_t compute_index(const shape_t& index) const;

    bool is_contiguous() const;
    Tensor<T> contiguous() const;

    Tensor<T> transpose(size_t dim_i, size_t dim_j) const;
    Tensor<T> reshape(const shape_t& shape) const;

    // _ - на конце обозначает inplace операцию
    Tensor<T>& fill_(const T& other);

    std::shared_ptr<Tensor<T>> grad_;
    std::shared_ptr<autodiff::BaseNode<T>> grad_fn_;
    bool requires_grad_;

    void backward() {
        (*(grad_fn_->input_grad_)).fill_(1);
        grad_fn_->backward();
    }
};

template <typename T>
Tensor<T>::Tensor(const shape_t& shape) : shape_(shape) {
    if (!shape_.empty()) {
        size_t total_len = 1;
        for (size_t dim : shape_) {
            total_len *= dim;
        }

        storage_ = std::make_shared<Storage<T>>(total_len);

        strides_ = compute_strides(shape_);
    }

    storage_offset_ = 0;
}

template <typename T>
shape_t Tensor<T>::compute_strides(const shape_t& shape) const {
    size_t stride = 1;
    shape_t strides(shape.size());

    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

template <typename T>
size_t Tensor<T>::numel() const {
    return std::accumulate(shape_.begin(), shape_.end(), 1u,
                           std::multiplies<size_t>());
}

template <typename T>
size_t Tensor<T>::compute_index(const shape_t& index) const {
    size_t idx = storage_offset_;
    for (size_t i = 0; i < index.size(); ++i) {
        idx += index[i] * strides_[i];
    }

    return idx;
}

template <typename T>
bool Tensor<T>::is_contiguous() const {
    shape_t strides = compute_strides(shape_);

    for (size_t i = 0; i < strides_.size(); ++i) {
        if (strides[i] != strides_[i])
            return false;
    }

    return true;
}

template <typename T>
Tensor<T> Tensor<T>::contiguous() const {
    if (is_contiguous()) {
        return *this;
    }

    Tensor<T> new_contiguous_tensor{};

    iterator::TensorIteratorConfig<T>()
        .add_input(*this)
        .add_output(new_contiguous_tensor)
        .build()
        .run([](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            return (*ptrs_out[0]) = (*ptrs_in[0]);
        });

    return new_contiguous_tensor;
}

template <typename T>
Tensor<T> Tensor<T>::transpose(size_t dim_i, size_t dim_j) const {
    if (dim_i >= ndim() || dim_j >= ndim()) {
        throw std::out_of_range(
            std::format("transpose: dim {} is out of range for tensor with {} "
                        "dimensions",
                        dim_i >= ndim() ? dim_i : dim_j, ndim())

        );
    }

    shape_t new_shape = shape_;
    shape_t new_strides = strides_;

    std::swap(new_shape[dim_i], new_shape[dim_j]);
    std::swap(new_strides[dim_i], new_strides[dim_j]);

    return Tensor<T>{storage_, new_shape, new_strides, storage_offset_};
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const shape_t& shape) const {
    size_t input_numel = std::accumulate(shape.begin(), shape.end(), 1u,
                                         std::multiplies<size_t>());

    if (input_numel != numel()) {
        throw exception::shape_exception{
            std::format("reshape: cannot reshape tensor of shape {} ({} "
                        "elements) "
                        "into shape {} ({} elements)",
                        utils::vec_to_string(shape_), numel(),
                        utils::vec_to_string(shape), input_numel)};
    }

    Tensor<T> contiguous_tensor = contiguous();

    return Tensor<T>{contiguous_tensor.storage_, shape, compute_strides(shape),
                     storage_offset_};
}

template <typename T>
Tensor<T>& Tensor<T>::fill_(const T& other) {
    iterator::TensorIteratorConfig<T>()
    .add_output(*this)
    .build()
    .run(
        [other](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            ptrs_in; // чисто чтобы не ругался статический анализатор
            return (*ptrs_out[0]) = other;
        });

    return *this;
}

template <typename T>
T& Tensor<T>::operator()(const shape_t& index) {
    if (index.size() != shape_.size()) [[unlikely]] {
        throw std::out_of_range(
            std::format("Rank mismatch: tensor is {}-D but got {}-D index",
                        shape_.size(), index.size()));
    }

    for (size_t i = 0; i < index.size(); ++i) {
        if (index[i] >= shape_[i]) [[unlikely]] {
            throw std::out_of_range(
                std::format("Index {} out of range for axis {} with size {}",
                            index[i], i, shape_[i]));
        }
    }

    return storage_->data_[compute_index(index)];
}

template <typename T>
const T& Tensor<T>::operator()(const shape_t& index) const {
    if (index.size() != shape_.size()) [[unlikely]] {
        throw std::out_of_range(
            std::format("Rank mismatch: tensor is {}-D but got {}-D index",
                        shape_.size(), index.size()));
    }

    for (size_t i = 0; i < index.size(); ++i) {
        if (index[i] >= shape_[i]) [[unlikely]] {
            throw std::out_of_range(
                std::format("Index {} out of range for axis {} with size {}",
                            index[i], i, shape_[i]));
        }
    }

    return storage_->data_[compute_index(index)];
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
    Tensor<T> output{};

    auto& left = *this;
    auto& right = other;

    iterator::TensorIteratorConfig<T>()
        .add_input(left)
        .add_input(right)
        .add_output(output)
        .build()
        .run([](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            return (*ptrs_out[0]) = (*ptrs_in[0]) * (*ptrs_in[1]);
        });

    grad_fn_ = std::make_shared<const autodiff::BaseNode<T>>(autodiff::MultNode<T>(left, right, output.shape()));

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const T& other) const {
    Tensor<T> output{};

    iterator::TensorIteratorConfig<T>()
        .add_input(*this)
        .add_output(output)
        .build()
        .run([other](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            return (*ptrs_out[0]) = (*ptrs_in[0]) * other;
        });

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    Tensor<T> output{};

    iterator::TensorIteratorConfig<T>()
        .add_input(*this)
        .add_input(other)
        .add_output(output)
        .build()
        .run([](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            return (*ptrs_out[0]) = (*ptrs_in[1]) + (*ptrs_in[0]);
        });

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const T& other) const {
    Tensor<T> output{};

    iterator::TensorIteratorConfig<T>()
        .add_input(*this)
        .add_output(output)
        .build()
        .run([other](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            return (*ptrs_out[0]) = (*ptrs_in[0]) + other;
        });

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
    Tensor<T> output{};

    iterator::TensorIteratorConfig<T>()
        .add_input(*this)
        .add_input(other)
        .add_output(output)
        .build()
        .run([](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            return (*ptrs_out[0]) = (*ptrs_in[0]) / (*ptrs_in[1]);
        });

    return output;
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const T& other) const {
    Tensor<T> output{};

    iterator::TensorIteratorConfig<T>()
        .add_input(*this)
        .add_output(output)
        .build()
        .run([other](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            return (*ptrs_out[0]) = (*ptrs_in[0]) / other;
        });

    return output;
}

} // namespace torch
