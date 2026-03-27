#pragma once
#include <cassert>
#include <memory>
#include <vector>

namespace torch {
template <typename T>
class Tensor;
}

namespace torch::autograd {
using size_t = std::size_t;
using shape_t = std::vector<size_t>;

struct GradMode {
    static bool is_enabled() { return enabled_; }
    static void set_enabled(bool enabled) { enabled_ = enabled; }

  private:
    inline static bool enabled_ = true;
};

// чесно спиздил у pytorch
struct AutoGradMode {

    AutoGradMode(bool enabled) : prev_mode(GradMode::is_enabled()) {
        GradMode::set_enabled(enabled);
    }
    AutoGradMode(const AutoGradMode&) = delete;
    AutoGradMode(AutoGradMode&&) = delete;
    AutoGradMode& operator=(const AutoGradMode&) = delete;
    AutoGradMode& operator=(AutoGradMode&&) = delete;
    ~AutoGradMode() { GradMode::set_enabled(prev_mode); }

    private:
        bool prev_mode;

};

struct NoGradGuard : public AutoGradMode {
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

template <typename T>
class BaseNode {
  public:
    std::shared_ptr<BaseNode<T>> prev_left_;
    std::shared_ptr<BaseNode<T>> prev_right_;

    std::shared_ptr<Tensor<T>> saved_values_left_;
    std::shared_ptr<Tensor<T>> saved_values_right_;

    mutable std::shared_ptr<Tensor<T>> grad_left_;
    mutable std::shared_ptr<Tensor<T>> grad_right_;

    mutable std::shared_ptr<Tensor<T>> input_grad_;

    mutable int counter_;

    BaseNode(const Tensor<T>& left, const Tensor<T>& right,
             const shape_t& shape)
        : counter_(0) {
        if (left.grad_fn_ == nullptr) { // лист
            grad_left_ = left.grad_;
        } else {
            prev_left_ = left.grad_fn_;
            prev_left_->counter_++;
        }

        if (right.grad_fn_ == nullptr) { // лист
            grad_right_ = right.grad_;
        } else {
            prev_right_ = right.grad_fn_;
            prev_right_->counter_++;
        }

        saved_values_left_ = std::make_shared<Tensor<T>>(left);
        saved_values_right_ = std::make_shared<Tensor<T>>(right);

        input_grad_ = std::make_shared<Tensor<T>>(Tensor<T>{shape});
        (*input_grad_).fill_(0);
    };

    virtual void backward() = 0;
};

template <typename T>
class MultNode : public BaseNode<T> {
    using BaseNode<T>::BaseNode;

    using BaseNode<T>::prev_left_;
    using BaseNode<T>::prev_right_;

    using BaseNode<T>::grad_left_;
    using BaseNode<T>::grad_right_;

    using BaseNode<T>::saved_values_left_;
    using BaseNode<T>::saved_values_right_;

    using BaseNode<T>::input_grad_;

    Tensor<T>& grad_left_calculation() const { 
        return *saved_values_right_; 
    }
    Tensor<T>& grad_right_calculation() const { 
        return *saved_values_left_; 
    }

    void backward() {
        assert((prev_left_ != nullptr ^ grad_left_ != nullptr)); // На самом деле это не верно,
        assert((prev_right_ != nullptr ^ grad_right_ != nullptr)); // если лист без градиента то это не выполнится

        if (prev_left_ != nullptr) {
            *(prev_left_->input_grad_) += grad_left_calculation() * (*input_grad_);
            prev_left_->counter_--;

            if (prev_left_->counter_ == 0) {
                prev_left_->backward();
            }
        }

        if (grad_left_ != nullptr) {
            *grad_left_ += grad_left_calculation() * (*input_grad_);
        }

        if (prev_right_ != nullptr) {
            *(prev_right_->input_grad_) += grad_right_calculation() * (*input_grad_);
            prev_right_->counter_--;

            if (prev_right_->counter_ == 0) {
                prev_right_->backward();
            }
        }

        if (grad_right_ != nullptr) {
            *grad_right_ += grad_right_calculation() * (*input_grad_);
        }
    }
};

} // namespace torch::autograd
