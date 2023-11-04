#include <torch/extension.h>

#include <tuple>

#include "imputer.h"

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> imputer_loss(
    const torch::Tensor &log_probs,
    const torch::Tensor &targets,
    const torch::Tensor &force_emits,
    const torch::Tensor &input_lengths,
    const torch::Tensor &target_lengths,
    int64_t BLANK,
    bool zero_infinity) {
  CHECK_INPUT(log_probs)
  CHECK_INPUT(targets)
  CHECK_INPUT(force_emits)
  CHECK_INPUT(input_lengths)
  CHECK_INPUT(target_lengths)
  torch::Tensor ilc =
      input_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  torch::Tensor tlc =
      target_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  at::IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  at::IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());

  auto res =
      imputer_loss_op(log_probs, targets.to(log_probs.device(), at::kLong),
                      force_emits.to(log_probs.device(), at::kLong), il, tl,
                      BLANK, zero_infinity);

  return res;
}

torch::Tensor imputer_loss_backward(
    const torch::Tensor &grad,
    const torch::Tensor &log_probs,
    const torch::Tensor &targets,
    const torch::Tensor &force_emits,
    const torch::Tensor &input_lengths,
    const torch::Tensor &target_lengths,
    const torch::Tensor &neg_log_likelihood,
    const torch::Tensor &log_alpha,
    int64_t BLANK,
    bool zero_infinity) {
  CHECK_INPUT(grad)
  CHECK_INPUT(log_probs)
  CHECK_INPUT(targets)
  CHECK_INPUT(force_emits)
  CHECK_INPUT(input_lengths)
  CHECK_INPUT(target_lengths)
  CHECK_INPUT(neg_log_likelihood)
  CHECK_INPUT(log_alpha)
  torch::Tensor ilc =
      input_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  torch::Tensor tlc =
      target_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  at::IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  at::IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());

  torch::Tensor res;

  res = imputer_loss_backward_op(
      grad, log_probs, targets.to(log_probs.device(), at::kLong),
      force_emits.to(log_probs.device(), at::kLong), il, tl, neg_log_likelihood,
      log_alpha, BLANK, zero_infinity);

  return res;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
best_alignment(const torch::Tensor &log_probs,
               const torch::Tensor &targets,
               const torch::Tensor &input_lengths,
               const torch::Tensor &target_lengths,
               int64_t BLANK,
               bool zero_infinity) {
  CHECK_INPUT(log_probs)
  CHECK_INPUT(targets)
  CHECK_INPUT(input_lengths)
  CHECK_INPUT(target_lengths)
  torch::Tensor ilc =
      input_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  torch::Tensor tlc =
      target_lengths.to(at::Device(at::kCPU), at::kLong).contiguous();
  at::IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  at::IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());

  auto res =
      best_alignment_op(log_probs, targets.to(log_probs.device(), at::kLong),
                        il, tl, BLANK, zero_infinity);

  return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("imputer_loss", &imputer_loss, "calculate imputer loss");
  m.def("imputer_loss_backward", &imputer_loss_backward,
        "calculate imputer loss gradient");
  m.def("best_alignment", &best_alignment, "get best alignments for ctc");
}
