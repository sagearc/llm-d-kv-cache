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
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/assert"
)

func hmaCatalog(pods ...string) *kvblock.GroupCatalog {
	c := kvblock.NewGroupCatalog()
	for _, pod := range pods {
		c.Learn(pod, 0, kvblock.GroupMetadata{Kind: "full_attention", BlockSize: 16})
		c.Learn(pod, 1, kvblock.GroupMetadata{Kind: "sliding_window", BlockSize: 16})
	}
	return c
}

func hmaEntry(pod, tier string, groupIdx int) kvblock.PodEntry {
	return kvblock.PodEntry{PodIdentifier: pod, DeviceTier: tier, HasGroup: true, GroupIdx: kvblock.GroupID(groupIdx)}
}

func hmaScorer(catalog *kvblock.GroupCatalog) *kvcache.LongestPrefixScorer {
	return &kvcache.LongestPrefixScorer{
		MediumWeights: map[string]float64{"gpu": 1.0, "cpu": 0.5},
		Catalog:       catalog,
	}
}

// TestHMAScoring verifies HMA group-aware scoring across key scenarios.
func TestHMAScoring(t *testing.T) {
	keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3})

	tests := []struct {
		name      string
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry
		catalog   *kvblock.GroupCatalog
		want      map[string]float64
	}{
		{
			name:    "full hit — both groups present",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)},
				2: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)},
				3: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)},
			},
			want: map[string]float64{podA: 3.0},
		},
		{
			name:    "chain break — any group missing at first block",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, "gpu", 1)},
				2: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)},
				3: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)},
			},
			want: map[string]float64{podA: 0.0},
		},
		{
			name:    "chain breaks when any group goes missing mid-prefix",
			catalog: hmaCatalog(podA),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)},
				2: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)},
				3: {hmaEntry(podA, "gpu", 0)},
			},
			want: map[string]float64{podA: 2.0},
		},
		{
			name:    "non-HMA entries — legacy behavior unchanged",
			catalog: kvblock.NewGroupCatalog(),
			keyToPods: map[kvblock.BlockHash][]kvblock.PodEntry{
				1: {{PodIdentifier: podA, DeviceTier: "gpu"}},
				2: {{PodIdentifier: podA, DeviceTier: "gpu"}},
				3: {{PodIdentifier: podA, DeviceTier: "gpu"}},
			},
			want: map[string]float64{podA: 3.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scores, err := hmaScorer(tt.catalog).Score(context.Background(), keys, tt.keyToPods)
			assert.NoError(t, err)
			for pod, want := range tt.want {
				assert.InDelta(t, want, scores[pod], 0.0001, "pod %s", pod)
			}
		})
	}
}

// TestHMAScoring_TwoPods verifies chain independence between pods.
func TestHMAScoring_TwoPods(t *testing.T) {
	catalog := hmaCatalog(podA, podB)

	keys := int64KeysToKVBlockKeys([]uint64{1, 2, 3})
	keyToPods := map[kvblock.BlockHash][]kvblock.PodEntry{
		1: {
			hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1),
			hmaEntry(podB, "gpu", 0), hmaEntry(podB, "gpu", 1),
		},
		2: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)}, // podB drops here
		3: {hmaEntry(podA, "gpu", 0), hmaEntry(podA, "gpu", 1)},
	}

	scores, err := hmaScorer(catalog).Score(context.Background(), keys, keyToPods)
	assert.NoError(t, err)
	assert.InDelta(t, 3.0, scores[podA], 0.0001)
	assert.InDelta(t, 1.0, scores[podB], 0.0001)
}
