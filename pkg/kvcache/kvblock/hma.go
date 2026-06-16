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

// GroupMetadata is the per-group attention info the pool stamps onto each
// PodEntry. The pool keeps it in a catalog so a BlockRemoved event (which
// carries only group_idx) can be rebuilt into the same entry that was stored.
type GroupMetadata struct {
	// Kind is the group's attention kind.
	Kind AttentionKind
	// SlidingWindowSize is the window in tokens for a sliding-window group, else 0.
	SlidingWindowSize int
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

// Get returns the metadata for a pod group. It is nil-safe.
func (c *GroupCatalog) Get(podID string, g GroupID) (GroupMetadata, bool) {
	if c == nil {
		return GroupMetadata{}, false
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	groups, ok := c.entries[podID]
	if !ok {
		return GroupMetadata{}, false
	}
	meta, ok := groups[g]
	return meta, ok
}
