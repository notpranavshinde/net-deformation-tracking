"""Explicit, immutable topology for a marker net."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class NetTopology:
    """A marker-node list and undirected edge list.

    Node IDs are external identities (and become ``obj_id`` values).  Numeric
    solver arrays use the row ordering in :attr:`node_ids`.
    """

    def __init__(self, node_ids: Iterable[int], edges: Iterable[tuple[int, int]]) -> None:
        self.node_ids = tuple(int(value) for value in node_ids)
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("node_ids must be unique")
        self._index = {node_id: index for index, node_id in enumerate(self.node_ids)}
        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for first, second in edges:
            a, b = int(first), int(second)
            if a == b:
                raise ValueError("Self edges are not allowed")
            if a not in self._index or b not in self._index:
                raise ValueError(f"Edge ({a}, {b}) references an absent node")
            edge = (a, b) if self._index[a] < self._index[b] else (b, a)
            if edge not in seen:
                seen.add(edge)
                normalized.append(edge)
        self.edges = tuple(normalized)
        self._neighbors = {node_id: [] for node_id in self.node_ids}
        for a, b in self.edges:
            self._neighbors[a].append(b)
            self._neighbors[b].append(a)

    def __len__(self) -> int:
        return len(self.node_ids)

    def index(self, node_id: int) -> int:
        """Return the dense solver index for an external node ID."""
        return self._index[int(node_id)]

    def neighbors(self, node_id: int) -> tuple[int, ...]:
        """Return neighboring external node IDs in deterministic order."""
        return tuple(self._neighbors[int(node_id)])

    @property
    def edge_indices(self) -> np.ndarray:
        """Return edges as an ``(E, 2)`` dense-index integer array."""
        return np.asarray(
            [(self._index[a], self._index[b]) for a, b in self.edges], dtype=np.int32
        ).reshape(-1, 2)

    @classmethod
    def from_grid(
        cls,
        cols: int,
        rows: int,
        present: Iterable[int] | np.ndarray | None = None,
    ) -> "NetTopology":
        """Build row-major four-neighbor topology, optionally omitting cells.

        ``present`` may be a flat/2-D boolean mask or an iterable of row-major
        grid IDs.  Original grid IDs are retained, so a hole never renumbers
        the nodes around it.
        """
        cols, rows = int(cols), int(rows)
        if cols < 1 or rows < 1:
            raise ValueError("Grid dimensions must be positive")
        count = cols * rows
        if present is None:
            keep = set(range(count))
        else:
            values = np.asarray(present)
            if values.dtype == np.bool_:
                if values.size != count:
                    raise ValueError("Boolean present mask must have cols*rows entries")
                keep = set(np.flatnonzero(values.reshape(-1)).tolist())
            else:
                keep = {int(value) for value in values.reshape(-1)}
                if any(value < 0 or value >= count for value in keep):
                    raise ValueError("present contains an ID outside the grid")
        node_ids = [node_id for node_id in range(count) if node_id in keep]
        edges = []
        for node_id in node_ids:
            row, col = divmod(node_id, cols)
            if col + 1 < cols and node_id + 1 in keep:
                edges.append((node_id, node_id + 1))
            if row + 1 < rows and node_id + cols in keep:
                edges.append((node_id, node_id + cols))
        return cls(node_ids, edges)

    @classmethod
    def from_sections(cls, section_layout) -> "NetTopology":
        """Build a sectioned net (reserved for the Phase 1c layout schema)."""
        raise NotImplementedError("Section-layout topology is not implemented yet")
