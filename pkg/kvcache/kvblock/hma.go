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

package kvblock

import "sync"

// GroupID identifies a vLLM KV cache group.
type GroupID int

// KVCacheSpecKind identifies vLLM KV cache group semantics. The string values
// mirror vLLM's KVCacheSpec.kind wire values. This is the single source of
// truth for the vocabulary; pkg/kvevents uses it directly (kvevents imports
// this package, so the type must live here to avoid an import cycle).
//
// Only the kinds the scorer classifies on are enumerated. vLLM's other kinds
// (mamba, chunked_local_attention, cross_attention, encoder_only_attention)
// flow through unchanged and are treated as not-modeled — neither main
// attention nor sliding window — so no named constants are needed for them.
type KVCacheSpecKind string

const (
	KVCacheSpecKindFullAttention     KVCacheSpecKind = "full_attention"
	KVCacheSpecKindMLAAttention      KVCacheSpecKind = "mla_attention"
	KVCacheSpecKindSlidingWindow     KVCacheSpecKind = "sliding_window"
	KVCacheSpecKindSlidingWindowMLA  KVCacheSpecKind = "sliding_window_mla"
	KVCacheSpecKindSinkFullAttention KVCacheSpecKind = "sink_full_attention"
)

// IsMainAttention reports whether the kind is a "main attention" group
// (full / MLA / sink-full attention).
//
// In vLLM's hybrid KV cache, the realized prefix-cache hit length converges to
// the minimum across all groups, and the main-attention group is usually the
// binding constraint: it requires the entire prefix to be cached, whereas
// sliding window, Mamba, and other groups are looser (they need only a trailing
// window or no prefix at all). Scoring therefore uses main-attention groups to
// define the candidate prefix; sliding-window groups can only shrink it (see
// GroupCatalog.SlidingWindowGroups), and other kinds are not modeled.
func (k KVCacheSpecKind) IsMainAttention() bool {
	switch k {
	case KVCacheSpecKindFullAttention, KVCacheSpecKindMLAAttention, KVCacheSpecKindSinkFullAttention:
		return true
	default:
		return false
	}
}

// IsSlidingWindow reports whether the kind is a sliding-window group, which
// needs only a trailing window of contiguous cached blocks for a hit (see
// SlidingWindowGroups).
func (k KVCacheSpecKind) IsSlidingWindow() bool {
	return k == KVCacheSpecKindSlidingWindow || k == KVCacheSpecKindSlidingWindowMLA
}

// GroupMetadata holds per-group KV cache spec info learned from BlockStored events.
type GroupMetadata struct {
	Kind              KVCacheSpecKind
	BlockSize         int
	SlidingWindowSize *int
}

// GroupCatalog is a thread-safe catalog of per-pod KV cache group metadata.
type GroupCatalog struct {
	mu      sync.RWMutex
	entries map[string]map[GroupID]GroupMetadata
}

// NewGroupCatalog creates a new, empty GroupCatalog.
func NewGroupCatalog() *GroupCatalog {
	return &GroupCatalog{
		entries: make(map[string]map[GroupID]GroupMetadata),
	}
}

// Learn records group metadata for a pod.
func (c *GroupCatalog) Learn(podID string, g GroupID, meta GroupMetadata) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.entries[podID] == nil {
		c.entries[podID] = make(map[GroupID]GroupMetadata)
	}
	c.entries[podID][g] = meta
}

// IsMainGroup reports whether group g on podID is a main-attention group whose
// blocks gate prefix-cache routing (see KVCacheSpecKind.IsMainAttention).
//
// When the catalog has not yet learned the group's kind — an event race before
// the first BlockStored, or an older vLLM that does not emit kv_cache_spec_kind
// — it falls back to treating group_idx 0 as the main group. The method is
// nil-safe: a nil catalog always uses the group_idx 0 fallback, so a scorer
// wired without a catalog still routes correctly for the common case where full
// attention is group 0.
func (c *GroupCatalog) IsMainGroup(podID string, g GroupID) bool {
	if c != nil {
		if meta, ok := c.Get(podID, g); ok && meta.Kind != "" {
			return meta.Kind.IsMainAttention()
		}
	}
	return g == 0
}

// SlidingWindowGroup describes a sliding-window KV cache group that the scorer
// can model precisely against the request-key block sequence.
type SlidingWindowGroup struct {
	// GroupID is the vLLM KV cache group identity.
	GroupID GroupID
	// ContiguousBlocks is the number of contiguous cached blocks a prefix-cache
	// hit requires at its trailing edge, cdiv(window-1, blockSize) — mirroring
	// vLLM's SlidingWindowManager._contiguous_blocks_for_hit.
	ContiguousBlocks int
}

// SlidingWindowGroups returns the sliding-window groups for podID that can be
// modeled at the given hashBlockSize (the request-key block size, mirroring
// vLLM's hash_block_size).
//
// A group qualifies only when (1) its kind is sliding window, (2) its block
// size equals hashBlockSize so its blocks align 1:1 with request keys — the
// router indexes a single hash block size and cannot reconstruct a
// differently-sized group's blocks — and (3) its window is known and large
// enough to require at least one trailing block. Groups that cannot be modeled
// precisely are omitted, so the caller simply does not use them to shrink the
// hit — never under-counting.
func (c *GroupCatalog) SlidingWindowGroups(podID string, hashBlockSize int) []SlidingWindowGroup {
	if c == nil {
		return nil
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	var groups []SlidingWindowGroup
	for g, meta := range c.entries[podID] {
		if !meta.Kind.IsSlidingWindow() {
			continue
		}
		if meta.BlockSize != hashBlockSize || meta.SlidingWindowSize == nil {
			continue
		}
		need := cdiv(*meta.SlidingWindowSize-1, hashBlockSize)
		if need <= 0 {
			continue
		}
		groups = append(groups, SlidingWindowGroup{GroupID: g, ContiguousBlocks: need})
	}
	return groups
}

// cdiv returns ceil(a/b) for non-negative a and positive b (mirrors vLLM's cdiv).
func cdiv(a, b int) int {
	return (a + b - 1) / b
}

// Get returns the metadata for a pod group.
func (c *GroupCatalog) Get(podID string, g GroupID) (GroupMetadata, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	groups, ok := c.entries[podID]
	if !ok {
		return GroupMetadata{}, false
	}
	meta, ok := groups[g]
	return meta, ok
}
