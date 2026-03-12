#pragma once
#include "tensor.hpp"


namespace torch::iterator {

    template<typename T>
    class Operand {
    public:
        T* data_ptr_;

        shape_t shape_;
        shape_t strides_;

        size_t offset_;

        bool is_output_; // broadcast?

        Operand(Tensor<T>& tensor, bool is_output=false);
    };

    template<typename T>
    Operand<T>::Operand(Tensor<T>& tensor, bool is_output) {
        shape_ = tensor.shape(); // Возвращает &, большой вопрос надо ли с этим что-то делать? Или оператор присваивания std::vector копирует
        strides_ = tensor.strides();
        offset_ = tensor.storage_offset();
        data_ptr_ = tensor.data_ptr();

        is_output_ = is_output;
    }


    template<typename T>
    class TensorIterator {
    public:
        using pointer_t = T*;

    private:

        shape_t shape_;

        struct IterOperand {
            pointer_t data;
            shape_t strides;
            size_t offset;
            bool broadcast;
        };

        std::vector<IterOperand> inputs_;
        std::vector<IterOperand> outputs_;

        size_t numel_;

    public:

        TensorIterator(
            const std::vector<Tensor<T>*>& inputs,
            const std::vector<Tensor<T>*>& outputs
        );

        size_t numel() const;

        void run(std::function<void(std::span<T*>)> kernel);
    };



    template<typename T>
    class TensorIteratorConfig {
    private:
        std::vector<Tensor<T>*> inputs_;
        std::vector<Tensor<T>*> outputs_;

    public:

        TensorIteratorConfig& add_input(Tensor<T>& tensor);

        TensorIteratorConfig& add_output(Tensor<T>& tensor);

        TensorIterator<T> build();
    };


}
