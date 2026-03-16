#pragma once
#include <algorithm>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch::iterator {
template <typename T>
class TensorIteratorConfig;
}

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
           shape_t strides, size_t storage_offset);

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
    size_t n = 1;
    for (size_t s : shape_) {
        n *= s;
    }

    return n;
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

    iterator::TensorIteratorConfig<T>()
        .add_input(*this)
        .add_input(other)
        .add_output(output)
        .build()
        .run([](std::span<const T*> ptrs_in, std::span<T*> ptrs_out) {
            return (*ptrs_out[0]) = (*ptrs_in[1]) * (*ptrs_in[0]);
        });

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

} // namespace torch

namespace torch::exception {
class shape_exception {
  public:
    shape_exception(std::string message) : message{message} {}
    std::string getMessage() const { return message; }

  private:
    std::string message;
};

} // namespace torch::exception
