// sampling.h — Token sampling (temperature, top-K, top-P, nucleus)

#pragma once

#include "types.h"

// CPU-side token sampling: temperature -> top-K -> top-P -> random pick
int32_t sample_token(const float * logits, int n_vocab, const SamplingParams & sp,
                     std::mt19937 & rng);
