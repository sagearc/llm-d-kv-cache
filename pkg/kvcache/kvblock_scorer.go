/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kvcache

import (
	"context"
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// KVScoringStrategy defines the strategy used to score pods for KV cache block reuse.
type KVScoringStrategy string

const (
	// LongestPrefixMatch Score by longest consecutive match from start.
	LongestPrefixMatch KVScoringStrategy = "LongestPrefix"
)

// KVBlockScorerConfig holds the configuration for the KVBlockScorer.
type KVBlockScorerConfig struct {
	ScoringStrategy KVScoringStrategy
	BackendConfigs  []*KVCacheBackendConfig `json:"backendConfigs"`
}

// DefaultKVBlockScorerConfig returns the default configuration for the KVBlockScorer.
func DefaultKVBlockScorerConfig() *KVBlockScorerConfig {
	return &KVBlockScorerConfig{
		ScoringStrategy: LongestPrefixMatch,
		BackendConfigs:  DefaultKVCacheBackendConfig(),
	}
}

// KVBlockScorer defines the interface for implementing a KV block scoring
// strategy.
type KVBlockScorer interface {
	// Strategy returns the scoring strategy type.
	Strategy() KVScoringStrategy
	// Score scores the blocks based on the scoring strategy.
	// It returns a map of pod names to their scores.
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error)
}

// NewKVBlockScorer creates a new KVBlockScorer based on the provided strategy.
//
// HMA group-aware scoring needs no extra wiring: the pool stamps each PodEntry
// with its own group's attention kind and window, and the scorer reads them off
// the entry. Set CanonicalBlockSize to enable the sliding-window reduction.
func NewKVBlockScorer(config *KVBlockScorerConfig) (*LongestPrefixScorer, error) {
	switch config.ScoringStrategy {
	case LongestPrefixMatch:
		// Build weight map from list of BackendConfigs for efficient lookup
		weightMap := make(map[string]float64)
		for _, medium := range config.BackendConfigs {
			weightMap[medium.Name] = medium.Weight
		}

		return &LongestPrefixScorer{
			MediumWeights: weightMap,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported scoring strategy: %s", config.ScoringStrategy)
	}
}

// LongestPrefixScorer scores based on longest consecutive block matches count
// starting from block 0.
type LongestPrefixScorer struct {
	// MediumWeights maps medium/device tier names to their scoring weights.
	MediumWeights map[string]float64
	// CanonicalBlockSize is the token-processor block size at which request keys
	// are chunked. Sliding-window token counts convert to request-key counts with
	// it, so scoring never depends on any engine block size. Zero disables the
	// window-aware reduction (phase 2 and the SWA-only fallback become no-ops).
	CanonicalBlockSize int
}

// podAttention summarizes a pod's attention topology as observed from the
// entries present at the scored keys. Because only the entry's own group kind
// is recorded (no model-level summary), this is a best-effort view: a hybrid
// pod whose main-attention blocks are all absent from this query looks
// main-less here. That trades the SWA-evicted-while-FA-survives precision for a
// catalog-free scorer.
type podAttention struct {
	// hasMain is true if the pod has a main-attention (or non-HMA) entry present
	// at any scored key.
	hasMain bool
	// slidingWindowSize is the largest own-group sliding-window size (tokens)
	// among the pod's present entries (0 = no sliding-window entry present).
	slidingWindowSize int
}

// collectPodAttention builds the per-pod attention view from the entries at the
// scored keys.
func collectPodAttention(keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) map[string]podAttention {
	meta := make(map[string]podAttention)
	for _, entries := range keyToPods {
		for _, e := range entries {
			m := meta[e.PodIdentifier]
			// Non-HMA entries (no group) and main-attention groups both anchor
			// the main-prefix path.
			if !e.HasGroup || e.AttentionKind.IsMainAttention() {
				m.hasMain = true
			}
			if e.AttentionKind.IsSlidingWindow() && e.SlidingWindowSize > m.slidingWindowSize {
				m.slidingWindowSize = e.SlidingWindowSize
			}
			meta[e.PodIdentifier] = m
		}
	}
	return meta
}

// cdiv returns ceil(a/b) for non-negative a and positive b (mirrors vLLM's cdiv).
func cdiv(a, b int) int {
	return (a + b - 1) / b
}

// Strategy returns the strategy type: LongestPrefixMatch.
func (s *LongestPrefixScorer) Strategy() KVScoringStrategy {
	return LongestPrefixMatch
}

// fillMainWeights populates dst with the maximum weight per pod across the
// given block entries, counting only main-attention entries (full / MLA /
// sink-full). These define the candidate prefix and carry its per-block weight.
// The caller must clear dst before calling so it can be reused across blocks.
//
// vLLM emits one entry per KV cache group for HMA models. Sliding-window entries
// are skipped here — they carry no prefix weight — but they are not ignored by
// scoring overall: Score's phase 2 uses their presence to bound the hit length.
// Mamba, chunked-local, and other kinds are not modeled. Non-HMA entries (no
// group identity) always count, preserving legacy uniform-attention behavior.
// Classification reads the entry's own AttentionKind, stamped by the pool.
func (s *LongestPrefixScorer) fillMainWeights(dst map[string]float64, entries []kvblock.PodEntry) {
	for _, entry := range entries {
		if entry.HasGroup && !entry.AttentionKind.IsMainAttention() {
			continue
		}
		weight := 1.0
		if s.MediumWeights != nil {
			if w, exists := s.MediumWeights[entry.DeviceTier]; exists {
				weight = w
			}
		}
		if cur, exists := dst[entry.PodIdentifier]; !exists || weight > cur {
			dst[entry.PodIdentifier] = weight
		}
	}
}

// Score implements the longest prefix scoring logic with weighted sum based on BackendConfig.
//
// Scoring mirrors vLLM's hybrid cache-hit convergence in two phases:
//
//  1. Main-attention prefix: the longest run of consecutive blocks from block 0
//     for which the pod holds a main-attention entry (full / MLA / sink-full).
//     Full attention is the binding constraint — it requires the entire prefix.
//
//  2. Sliding-window reduction: a sliding-window hit needs only a trailing
//     window of contiguous cached blocks, so a right-to-left scan finds the
//     longest prefix whose trailing window is present. This can only shrink the
//     main-attention prefix, catching the case where a pod's SWA window for
//     this prefix was evicted while its full-attention blocks survived. Early
//     SWA blocks (outside the window) are not required, matching vLLM. With no
//     catalog (or no modeled sliding-window group) phase 2 is a no-op and the
//     score is the phase-1 prefix.
//
// Pods running SWA-only models (no main-attention group) never enter phase 1;
// they are scored separately, mirroring vLLM's unitary coordinator (see
// scorePodsWithoutMainGroup).
//
// The score is the tier-weighted sum over the converged hit length.
func (s *LongestPrefixScorer) Score(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	// Per-pod attention view, read from the kind/window stamped on each entry.
	podMeta := collectPodAttention(keyToPods)

	// Phase 1: per-pod main-attention prefix, recording cumulative per-block
	// weights so phase 2 can truncate to the converged hit without re-summing.
	cumWeights := make(map[string][]float64)

	// Scratch map reused across iterations to avoid per-key allocation.
	curWeights := make(map[string]float64)
	s.fillMainWeights(curWeights, keyToPods[keys[0]])

	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		cumWeights[pod] = []float64{w}
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		clear(curWeights)
		s.fillMainWeights(curWeights, keyToPods[keys[i]])

		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				cum := cumWeights[pod]
				cumWeights[pod] = append(cum, cum[len(cum)-1]+w)
			} else {
				delete(activePods, pod)
			}
		}
	}

	// Phase 2: reduce each pod's prefix by its sliding-window coverage, then
	// read the prefix-sum at the converged hit length.
	podScores := make(map[string]float64, len(cumWeights))
	for pod, cum := range cumWeights {
		hit := len(cum)
		if cb := s.contiguousBlocks(podMeta[pod].slidingWindowSize); cb > 0 {
			hit = slidingWindowHitLen(pod, cb, hit, keys, keyToPods)
		}
		if hit == 0 {
			podScores[pod] = 0
			continue
		}
		podScores[pod] = cum[hit-1]
	}

	s.scorePodsWithoutMainGroup(podScores, podMeta, keys, keyToPods)

	return podScores, nil
}

// contiguousBlocks converts a sliding-window size in tokens to the number of
// contiguous trailing canonical-block keys a hit requires: cdiv(window-1,
// canonicalBlockSize), mirroring vLLM's SlidingWindowManager in router units.
// Returns 0 when windowing is disabled (no window, or no canonical block size).
func (s *LongestPrefixScorer) contiguousBlocks(slidingWindowSize int) int {
	if slidingWindowSize <= 0 || s.CanonicalBlockSize <= 0 {
		return 0
	}
	need := cdiv(slidingWindowSize-1, s.CanonicalBlockSize)
	if need <= 0 {
		return 0
	}
	return need
}

// scorePodsWithoutMainGroup scores pods that look SWA-only at these keys (a
// sliding-window entry present, no main-attention entry present). vLLM serves
// SWA-only models through its unitary coordinator: the trailing-window scan
// runs over the whole request, not bounded by a full-attention prefix. Blocks
// inside the hit score at the pod's tier weight where it holds entries, and 1.0
// where it holds none: those are the null-prefix blocks vLLM fills in — outside
// every window, skipped by the engine entirely, so the saved compute is
// tier-independent.
//
// Detection is best-effort from entry presence: a hybrid pod whose
// main-attention blocks are absent from this query is indistinguishable from an
// SWA-only pod here, and may be scored by this path. Distinguishing them would
// require model-level topology, which the catalog-free scorer does not carry.
func (s *LongestPrefixScorer) scorePodsWithoutMainGroup(
	podScores map[string]float64,
	podMeta map[string]podAttention,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) {
	for pod, meta := range podMeta {
		if meta.hasMain {
			continue
		}
		cb := s.contiguousBlocks(meta.slidingWindowSize)
		if cb == 0 {
			// No sliding-window entry present (e.g. mamba-only): not scored.
			continue
		}
		hit := slidingWindowHitLen(pod, cb, len(keys), keys, keyToPods)
		if hit == 0 {
			continue
		}
		var score float64
		for i := 0; i < hit; i++ {
			score += s.maxPodWeight(keyToPods[keys[i]], pod)
		}
		podScores[pod] = score
	}
}

// maxPodWeight returns the maximum tier weight among pod's entries at this
// block, or 1.0 when the pod holds none (a null-prefix block inside a
// sliding-window hit; see scorePodsWithoutMainGroup).
func (s *LongestPrefixScorer) maxPodWeight(entries []kvblock.PodEntry, pod string) float64 {
	weight := 0.0
	found := false
	for _, e := range entries {
		if e.PodIdentifier != pod {
			continue
		}
		w := 1.0
		if s.MediumWeights != nil {
			if mw, exists := s.MediumWeights[e.DeviceTier]; exists {
				w = mw
			}
		}
		if !found || w > weight {
			weight = w
			found = true
		}
	}
	if !found {
		return 1.0
	}
	return weight
}

// slidingWindowHitLen returns the longest prefix length (in blocks, capped at
// maxBlocks) for which pod has the contiguous trailing window of
// contiguousBlocks sliding-window blocks. It mirrors vLLM's
// SlidingWindowManager.find_longest_cache_hit: scan right-to-left and stop once
// contiguousBlocks blocks are contiguously present; if the window never
// completes, the contiguous run anchored at block 0 still counts (a prefix
// shorter than the window).
//
// Presence at a block is a single sliding-window entry for the pod. One
// sliding-window spec per model is assumed; models with several distinct
// sliding-window specs (which vLLM looks up jointly) are not modeled here.
func slidingWindowHitLen(
	pod string,
	contiguousBlocks int,
	maxBlocks int,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) int {
	contiguous := 0
	for i := maxBlocks - 1; i >= 0; i-- {
		if slidingWindowPresent(keyToPods[keys[i]], pod) {
			contiguous++
			if contiguous >= contiguousBlocks {
				return i + contiguous
			}
		} else {
			contiguous = 0
		}
	}
	return contiguous
}

// slidingWindowPresent reports whether pod holds a sliding-window entry at this block.
func slidingWindowPresent(entries []kvblock.PodEntry, pod string) bool {
	for _, e := range entries {
		if e.PodIdentifier == pod && e.HasGroup && e.AttentionKind.IsSlidingWindow() {
			return true
		}
	}
	return false
}
