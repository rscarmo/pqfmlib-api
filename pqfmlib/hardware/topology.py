"""Portable connectivity snapshots, independent of simulator and credentials."""

from __future__ import annotations

import math

from pqfmlib.hardware.coupling import get_coupling_edges_and_costs
from pqfmlib.utils.io import load_json, save_json


def load_topology_snapshot(path):
    """Load ordered physical edges and optional assignment costs from JSON.

    Costs influence subgraph/feature selection only; they do not add noise.
    Missing costs use the existing backend reader's default of 1.0.
    """
    data = load_json(path)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Topology snapshot requires schema_version=1.")
    entries = data.get("coupling_edges")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Topology snapshot requires a nonempty coupling_edges list.")
    edges = []
    for edge in entries:
        if (not isinstance(edge, list) or len(edge) != 2
                or any(type(q) is not int or q < 0 for q in edge) or edge[0] == edge[1]):
            raise ValueError("coupling_edges must contain pairs of distinct nonnegative integer qubits.")
        edges.append(tuple(edge))
    if len(set(edges)) != len(edges):
        raise ValueError("coupling_edges contains duplicate directed edges.")
    raw_costs = data.get("edge_costs", [])
    if not isinstance(raw_costs, list):
        raise ValueError("edge_costs must be a list of [source, target, cost] entries.")
    supplied = {}
    available = set(edges)
    for entry in raw_costs:
        if (not isinstance(entry, list) or len(entry) != 3
                or any(type(q) is not int or q < 0 for q in entry[:2])):
            raise ValueError("edge_costs must contain [source, target, cost] entries.")
        u, v, cost = entry
        if (u, v) not in available and (v, u) not in available:
            raise ValueError("edge_costs references an edge outside coupling_edges.")
        if (isinstance(cost, bool) or not isinstance(cost, (int, float))
                or not 0 <= cost <= 1 or not math.isfinite(cost)):
            raise ValueError("Edge costs must be finite numbers between 0 and 1.")
        if (u, v) in supplied:
            raise ValueError("edge_costs contains duplicate directed edges.")
        supplied[u, v] = float(cost)
    costs = {edge: supplied.get(edge, supplied.get(edge[::-1], 1.0)) for edge in edges}
    return edges, costs


def save_topology_snapshot(backend, path) -> None:
    """Save backend connectivity/selection costs without submitting any job."""
    edges, costs = get_coupling_edges_and_costs(backend)
    data = {
        "schema_version": 1,
        "coupling_edges": [[int(u), int(v)] for u, v in edges],
        "edge_costs": [[int(u), int(v), float(costs[u, v])] for u, v in edges],
    }
    save_json(data, path)
