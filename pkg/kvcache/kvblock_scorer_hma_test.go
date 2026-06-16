/*
Copyright 2026 The llm-d Authors.

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

package kvcache_test

import (
	"context"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The HMA scoring tests run over 3 request keys at canonical block size 16. The
// sliding-window group's window of 32 tokens needs cdiv(31,16) = 2 contiguous
// trailing blocks for a hit. Each entry carries its own attention kind (the
// pool stamps it from the BlockStored event), so scoring reads topology off the
// entries with no catalog.
const (
	hmaTestBlockSize = 16
	hmaSWAWindow     = 32
)

// hmaEntry builds a gpu entry for a hybrid model: group 0 is main attention,
// group 1 is the sliding-window group.
func hmaEntry(pod string, groupIdx int) kvblock.PodEntry {
	return stampHMAKind(kvblock.PodEntry{
		PodIdentifier: pod, DeviceTier: "gpu", HasGroup: true, GroupIdx: kvblock.GroupID(groupIdx),
	})
}

// stampHMAKind sets the attention kind/window the pool would stamp for the
// hybrid model under test.
func stampHMAKind(e kvblock.PodEntry) kvblock.PodEntry {
	switch e.GroupIdx {
	case 0:
		e.AttentionKind = kvblock.AttentionMain
	case 1:
		e.AttentionKind = kvblock.AttentionSlidingWindow
		e.SlidingWindowSize = hmaSWAWindow
	}
	return e
}

// blocksFor maps keys 1..len(specs) to podA entries. Each spec lists the
// entries present at that block, space-separated: "1" = group 1 on gpu,
// "c1" = group 1 on cpu, "x" = an ungrouped legacy entry, "" = nothing cached.
func blocksFor(specs ...string) map[kvblock.BlockHash][]kvblock.PodEntry {
	keyToPods := make(map[kvblock.BlockHash][]kvblock.PodEntry, len(specs))
	for i, spec := range specs {
		var entries []kvblock.PodEntry
		for _, token := range strings.Fields(spec) {
			entry := kvblock.PodEntry{PodIdentifier: podA, DeviceTier: "gpu"}
			if token != "x" {
				if rest, isCPU := strings.CutPrefix(token, "c"); isCPU {
					entry.DeviceTier = "cpu"
					token = rest
				}
				entry.HasGroup = true
				entry.GroupIdx = kvblock.GroupID(token[0] - '0')
				entry = stampHMAKind(entry)
			}
			entries = append(entries, entry)
		}
		if entries != nil {
			keyToPods[kvblock.BlockHash(i+1)] = entries // #nosec G115 -- test data, i is small
		}
	}
	return keyToPods
}

func hmaScorer() *kvcache.LongestPrefixScorer {
	return &kvcache.LongestPrefixScorer{
		MediumWeights:      map[string]float64{"gpu": 1.0, "cpu": 0.5},
		CanonicalBlockSize: hmaTestBlockSize,
	}
}

func assertHMAScore(t *testing.T, keyToPods map[kvblock.BlockHash][]kvblock.PodEntry, want float64) {
	t.Helper()
	keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3})
	scores, err := hmaScorer().Score(context.Background(), keys, keyToPods)
	assert.NoError(t, err)
	assert.InDelta(t, want, scores[podA], 0.0001)
}

// TestHMAScoring verifies window-aware scoring: the main-attention group gates
// the contiguous prefix and a present sliding-window group reduces it to the
// longest prefix whose trailing window is present, mirroring vLLM's
// SlidingWindowManager.find_longest_cache_hit (only the trailing window is
// required; a block-0-anchored run shorter than the window still hits).
//
// Because the scorer reads only each entry's own group kind, the reduction can
// fire only when sliding-window entries are present at the scored keys: a
// prefix whose sliding-window blocks are entirely absent is not collapsed (see
// the "SWA entries absent" case). Distinguishing that from a non-hybrid model
// would need model-level topology, which the catalog-free scorer does not hold.
func TestHMAScoring(t *testing.T) {
	tests := []struct {
		name   string
		blocks map[kvblock.BlockHash][]kvblock.PodEntry
		want   float64
	}{
		{"full hit - main and SWA at every block", blocksFor("0 1", "0 1", "0 1"), 3.0},
		{"SWA trailing window evicted - hit shrinks to rightmost window", blocksFor("0 1", "0 1", "0"), 2.0},
		{"early SWA block evicted - only the trailing window is required", blocksFor("0", "0 1", "0 1"), 3.0},
		{"SWA window never completes - block-0-anchored run still hits", blocksFor("0 1", "0", "0"), 1.0},
		{"non-main entry at block 0 does not anchor the prefix", blocksFor("1", "0 1", "0 1"), 0.0},
		{"chain breaks where the main group goes missing", blocksFor("0 1", "0 1", "1"), 2.0},
		{"mixed tiers with truncation - weights sum over the hit only", blocksFor("0 1", "c0 1", "0"), 1.5},
		{"non-HMA entries - legacy behavior unchanged", blocksFor("x", "x", "x"), 3.0},
		// SWA entries absent from the prefix: with no sliding-window entry to
		// read, the reduction cannot fire, so the main prefix stands. (The
		// catalog-based scorer collapsed this to 0.0 from model topology.)
		{"SWA entries absent - main prefix is not collapsed", blocksFor("0", "0", "0"), 3.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) { assertHMAScore(t, tt.blocks, tt.want) })
	}
}

// TestHMAScoring_TwoPods verifies chain independence between pods.
func TestHMAScoring_TwoPods(t *testing.T) {
	keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3})
	keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{
		1: {hmaEntry(podA, 0), hmaEntry(podA, 1), hmaEntry(podB, 0), hmaEntry(podB, 1)},
		2: {hmaEntry(podA, 0), hmaEntry(podA, 1)}, // podB drops here
		3: {hmaEntry(podA, 0), hmaEntry(podA, 1)},
	}

	scores, err := hmaScorer().Score(context.Background(), keys, keyToPods)
	assert.NoError(t, err)
	assert.InDelta(t, 3.0, scores[podA], 0.0001)
	assert.InDelta(t, 1.0, scores[podB], 0.0001)
}

// TestHMAScoring_SWAOnlyModel verifies the unitary-coordinator mirror for pods
// with no main-attention entry present: the trailing-window scan runs over the
// whole key range, cached blocks score at their tier weight, and null-prefix
// blocks (outside the window, never cached, skipped by the engine entirely)
// count at the tier-independent weight 1.0.
func TestHMAScoring_SWAOnlyModel(t *testing.T) {
	assertHMAScore(t, blocksFor("", "1", "1"), 3.0)   // null prefix (1.0) + window
	assertHMAScore(t, blocksFor("", "c1", "c1"), 2.0) // cpu run 0.5+0.5, null prefix 1.0
	assertHMAScore(t, blocksFor("1", "", ""), 1.0)    // window never completes - anchored run
	assertHMAScore(t, blocksFor("", "", ""), 0.0)     // nothing cached
}

// TestHMAScoring_IndexerWiring exercises the real construction path:
// NewKVCacheIndexer must wire the token processor's block size into the scorer,
// and the scorer must read the attention kind off the stored entries. An
// unwired CanonicalBlockSize skips the reduction and reports the full prefix
// (3.0) where the present-but-incomplete window demands a collapse (1.0).
func TestHMAScoring_IndexerWiring(t *testing.T) {
	ctx := context.Background()

	config, err := kvcache.NewDefaultConfig()
	require.NoError(t, err)
	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSizeTokens: hmaTestBlockSize,
		HashSeed:        "test",
	})
	require.NoError(t, err)
	indexer, err := kvcache.NewKVCacheIndexer(ctx, config, tp)
	require.NoError(t, err)

	tokens := make([]uint32, 3*hmaTestBlockSize)
	for i := range tokens {
		tokens[i] = uint32(i + 1) // #nosec G115 -- test data, i is small
	}
	keys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, keys, 3)

	add := func(key kvblock.BlockHash, groupIdx int) {
		require.NoError(t, indexer.KVBlockIndex().Add(ctx,
			[]kvblock.BlockHash{key}, []kvblock.BlockHash{key},
			[]kvblock.PodEntry{hmaEntry(podA, groupIdx)}))
	}

	// Main attention at every block; the sliding-window group present only at
	// the first block, so its 2-block trailing window never completes and a
	// wired scorer collapses the hit to the block-0-anchored run (1.0).
	for _, key := range keys {
		add(key, 0)
	}
	add(keys[0], 1)

	scores, err := indexer.ScoreTokens(ctx, tokens, "test-model", nil, nil)
	require.NoError(t, err)
	assert.InDelta(t, 1.0, scores[podA], 0.0001)

	// Positive control: with the sliding-window group present at every block the
	// full hit returns.
	for _, key := range keys {
		add(key, 1)
	}
	scores, err = indexer.ScoreTokens(ctx, tokens, "test-model", nil, nil)
	require.NoError(t, err)
	assert.InDelta(t, 3.0, scores[podA], 0.0001)
}
