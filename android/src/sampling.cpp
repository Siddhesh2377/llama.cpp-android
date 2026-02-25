// sampling.cpp — Token sampling (temperature, top-K, top-P, nucleus)

#include "gguf-engine/sampling.h"

#include <cmath>
#include <vector>
#include <algorithm>

// CPU-side token sampling: temperature -> top-K -> top-P -> random pick
int32_t sample_token(const float * logits, int n_vocab, const SamplingParams & sp,
                     std::mt19937 & rng) {
    // Greedy (argmax)
    if (sp.temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    // Build (logit, index) pairs with temperature scaling
    std::vector<std::pair<float, int>> candidates(n_vocab);
    float inv_temp = 1.0f / sp.temp;
    for (int i = 0; i < n_vocab; i++) {
        candidates[i] = { logits[i] * inv_temp, i };
    }

    // Top-K: partial sort to keep only top_k candidates
    int k = (sp.top_k > 0 && sp.top_k < n_vocab) ? sp.top_k : n_vocab;
    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
        [](auto & a, auto & b) { return a.first > b.first; });
    candidates.resize(k);

    // Softmax over top-K
    float max_logit = candidates[0].first;
    float sum = 0.0f;
    for (auto & c : candidates) {
        c.first = expf(c.first - max_logit);
        sum += c.first;
    }
    for (auto & c : candidates) {
        c.first /= sum;
    }

    // Top-P (nucleus): keep smallest set with cumulative prob >= top_p
    if (sp.top_p < 1.0f && sp.top_p > 0.0f) {
        float cum = 0.0f;
        int cutoff = (int)candidates.size();
        for (int i = 0; i < (int)candidates.size(); i++) {
            cum += candidates[i].first;
            if (cum >= sp.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);

        // Re-normalize
        sum = 0.0f;
        for (auto & c : candidates) sum += c.first;
        for (auto & c : candidates) c.first /= sum;
    }

    // Random weighted pick
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cum = 0.0f;
    for (auto & c : candidates) {
        cum += c.first;
        if (r <= cum) return c.second;
    }
    return candidates.back().second;
}
