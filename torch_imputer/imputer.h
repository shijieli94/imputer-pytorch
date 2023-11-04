// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Licensed under the BSD-3-Clause license
// This is the GPU implementation of the Connectionist Temporal Loss.
// We mostly follow Graves.
// 1. Graves et al: http://www.cs.toronto.edu/~graves/icml_2006.pdf
// We use the equations from above link, but note that [1] has 1-based indexing
// and we (of course) use 0-based. Graves et al call the probabilities y, we use
// log_probs (also calling them inputs) A few optimizations (simmilar to those
// here, but also some I didn't take) are described in
// 2. Minmin Sun:
// http://on-demand.gputechconf.com/gtc/2016/presentation/s6383-minmin-sun-speech-recognition.pdf

#pragma once

#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> imputer_loss_op(
    const torch::Tensor &log_probs,
    const torch::Tensor &targets,
    const torch::Tensor &force_emits,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    int64_t BLANK,
    bool zero_infinity);

torch::Tensor imputer_loss_backward_op(
    const torch::Tensor &grad,
    const torch::Tensor &log_probs,
    const torch::Tensor &targets,
    const torch::Tensor &force_emits,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    const torch::Tensor &neg_log_likelihood,
    const torch::Tensor &log_alpha,
    int64_t BLANK,
    bool zero_infinity);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
best_alignment_op(
    const torch::Tensor &log_probs,
    const torch::Tensor &targets,
    at::IntArrayRef input_lengths,
    at::IntArrayRef target_lengths,
    int64_t BLANK,
    bool zero_infinity);
