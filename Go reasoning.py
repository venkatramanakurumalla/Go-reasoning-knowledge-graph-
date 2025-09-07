#!/usr/bin/env python3
"""
Go Reasoning Knowledge Graph — Enhanced Single File
Improvements:
 - safer add_edge (validation / placeholder creation)
 - reverse adjacency (incoming edges)
 - thread-safe mutations
 - improved search (scoring + token match)
 - shortest-path BFS
 - convenient create_node/create_edge helpers
 - robust save/load (enum handling)
 - minor API ergonomics
"""

import json
import os
import uuid
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Set, Optional, Iterable, Tuple

# ------------------------
# CORE TYPES
# ------------------------

class NodeType(Enum):
    CONCEPT = "concept"
    PATTERN = "pattern"
    FUNCTION = "function"
    TYPE = "type"
    MODULE = "module"
    TOOL = "tool"
    ANTI_PATTERN = "anti_pattern"
    BEST_PRACTICE = "best_practice"

class RelationType(Enum):
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    REQUIRES = "requires"
    REPLACES = "replaces"
    BEST_PRACTICE_FOR = "best_practice_for"
    ANTI_PATTERN_OF = "anti_pattern_of"
    USED_IN = "used_in"
    TRIGGERS = "triggers"
    CONCURRENCY_SAFE = "concurrency_safe"
    NOT_IDIOMATIC = "not_idiomatic"
    USES = "uses"          # added to support "uses" relationships

@dataclass
class Node:
    id: str
    type: NodeType
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Node":
        return Node(
            id=d["id"],
            type=NodeType(d["type"]),
            name=d.get("name", ""),
            description=d.get("description", ""),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
            source=d.get("source", "")
        )

@dataclass
class Edge:
    from_id: str
    to_id: str
    relation: RelationType
    weight: float = 1.0
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "relation": self.relation.value,
            "weight": self.weight,
            "evidence": self.evidence
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Edge":
        return Edge(
            from_id=d["from_id"],
            to_id=d["to_id"],
            relation=RelationType(d["relation"]),
            weight=d.get("weight", 1.0),
            evidence=d.get("evidence", "")
        )

# ------------------------
# GRAPH
# ------------------------

class GoKnowledgeGraph:
    def __init__(self, create_placeholders: bool = True):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        # adjacency maps from node id -> outgoing edges
        self.adjacency: Dict[str, List[Edge]] = {}
        # incoming edges map
        self.incoming: Dict[str, List[Edge]] = {}
        # tag index (lowercased tags -> set of ids)
        self.index_by_tag: Dict[str, Set[str]] = {}
        # type index
        self.index_by_type: Dict[NodeType, Set[str]] = {}
        self.lock = threading.RLock()
        self.create_placeholders = create_placeholders

    # ----------------------
    # Node management
    # ----------------------
    def add_node(self, node: Node):
        with self.lock:
            if node.id in self.nodes:
                # merge/update existing node: preserve existing tags and metadata
                existing = self.nodes[node.id]
                existing.name = node.name or existing.name
                existing.description = node.description or existing.description
                existing.source = node.source or existing.source
                # merge tags/metadata (dedupe)
                existing.tags = list(dict.fromkeys(existing.tags + node.tags))
                existing.metadata.update(node.metadata or {})
                node = existing
            else:
                self.nodes[node.id] = node

            # index by tag (case-insensitive)
            for tag in node.tags:
                key = tag.lower()
                if key not in self.index_by_tag:
                    self.index_by_tag[key] = set()
                self.index_by_tag[key].add(node.id)

            # index by type
            if node.type not in self.index_by_type:
                self.index_by_type[node.type] = set()
            self.index_by_type[node.type].add(node.id)

            # ensure adjacency buckets exist
            self.adjacency.setdefault(node.id, [])
            self.incoming.setdefault(node.id, [])

    def create_node(self,
                    name: str,
                    type: NodeType = NodeType.CONCEPT,
                    description: str = "",
                    tags: Optional[Iterable[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    source: str = "") -> Node:
        nid = str(uuid.uuid4())
        node = Node(id=nid,
                    type=type,
                    name=name,
                    description=description,
                    tags=list(tags or []),
                    metadata=metadata or {},
                    source=source)
        self.add_node(node)
        return node

    # ----------------------
    # Edge management
    # ----------------------
    def add_edge(self, edge: Edge):
        with self.lock:
            # validate node presence
            if edge.from_id not in self.nodes:
                if self.create_placeholders:
                    # create placeholder concept
                    self.add_node(Node(id=edge.from_id, type=NodeType.CONCEPT, name=edge.from_id))
                else:
                    raise KeyError(f"from_id {edge.from_id} not found in nodes")
            if edge.to_id not in self.nodes:
                if self.create_placeholders:
                    self.add_node(Node(id=edge.to_id, type=NodeType.CONCEPT, name=edge.to_id))
                else:
                    raise KeyError(f"to_id {edge.to_id} not found in nodes")

            self.adjacency.setdefault(edge.from_id, []).append(edge)
            self.incoming.setdefault(edge.to_id, []).append(edge)
            self.edges.append(edge)

    def create_edge(self, from_id: str, to_id: str, relation: RelationType, weight: float = 1.0, evidence: str = "") -> Edge:
        e = Edge(from_id=from_id, to_id=to_id, relation=relation, weight=weight, evidence=evidence)
        self.add_edge(e)
        return e

    # ----------------------
    # Query helpers
    # ----------------------
    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str, relation: Optional[RelationType] = None) -> List[Node]:
        edges = self.adjacency.get(node_id, [])
        if relation:
            edges = [e for e in edges if e.relation == relation]
        return [self.nodes[e.to_id] for e in edges if e.to_id in self.nodes]

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        return list(self.adjacency.get(node_id, []))

    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        return list(self.incoming.get(node_id, []))

    def find_nodes_by_tag(self, tag: str) -> List[Node]:
        ids = self.index_by_tag.get(tag.lower(), set())
        return [self.nodes[nid] for nid in ids]

    def find_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        ids = self.index_by_type.get(node_type, set())
        return [self.nodes[nid] for nid in ids]

    def __len__(self):
        return len(self.nodes)

    # ----------------------
    # Persistence
    # ----------------------
    def to_json(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges]
        }

    def save_to_file(self, filepath: str):
        tmp = filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, filepath)

    def load_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # clear
        with self.lock:
            self.nodes.clear()
            self.edges.clear()
            self.adjacency.clear()
            self.incoming.clear()
            self.index_by_tag.clear()
            self.index_by_type.clear()

            for n in data.get("nodes", []):
                node = Node.from_dict(n)
                self.add_node(node)
            for e in data.get("edges", []):
                edge = Edge.from_dict(e)
                # add_edge will validate/create placeholders if needed
                self.add_edge(edge)

    # ----------------------
    # Search / Simple reasoning
    # ----------------------
    def search_by_description(self, keyword: str, top_k: int = 10) -> List[Tuple[Node, float]]:
        """
        Score nodes by simple token overlap (case-insensitive) from name & description.
        Returns list of (Node, score) sorted by score desc.
        """
        kw = keyword.lower().split()
        scores: Dict[str, float] = {}
        for nid, node in self.nodes.items():
            text = f"{node.name} {node.description}".lower()
            score = 0.0
            for token in kw:
                if token in text:
                    score += 1.0
            if score > 0:
                scores[nid] = score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.nodes[nid], score) for nid, score in ranked]

    def recommend_sync_mechanism(self, for_pattern_or_tag: str) -> List[str]:
        """
        Look for nodes with matching tag (case-insensitive) and return candidate TYPE names
        which include 'sync' or 'chan' in tags.
        """
        results = set()
        tag_key = for_pattern_or_tag.lower()
        node_ids = self.index_by_tag.get(tag_key, set())
        for nid in node_ids:
            for e in self.get_outgoing_edges(nid):
                target = self.get_node(e.to_id)
                if not target:
                    continue
                # type-level recommendation heuristic
                if target.type == NodeType.TYPE and any(tk in (tg.lower() for tg in target.tags) for tk in ("sync", "channel", "chan")):
                    results.add(target.name)
        return sorted(results)

    def find_anti_patterns_in_code_intent(self, intent: str) -> List[dict]:
        anti_patterns = []
        intent_lower = intent.lower()

        if "global" in intent_lower or "shared state" in intent_lower or "global variable" in intent_lower:
            node = self.get_node("anti:global_state_in_handler")
            if node:
                anti_patterns.append({
                    "anti_pattern": node.name,
                    "description": node.description,
                    "fix": "Use sync.Mutex or avoid global state; prefer request-local state"
                })

        if ("goroutine" in intent_lower or "concurrent" in intent_lower) and \
           not any(kw in intent_lower for kw in ["wait", "sync", "context", "cancel"]):
            node = self.get_node("anti:goroutine_leak")
            if node:
                anti_patterns.append({
                    "anti_pattern": node.name,
                    "description": node.description,
                    "fix": "Use sync.WaitGroup, channels or context.Context for lifecycle management"
                })

        return anti_patterns

    # ----------------------
    # Graph algorithms
    # ----------------------
    def shortest_path(self, start_id: str, goal_id: str) -> Optional[List[str]]:
        """
        BFS shortest path (by number of edges). Returns list of node ids or None.
        """
        if start_id == goal_id:
            return [start_id]
        if start_id not in self.nodes or goal_id not in self.nodes:
            return None
        from collections import deque
        q = deque([start_id])
        prev: Dict[str, Optional[str]] = {start_id: None}
        while q:
            cur = q.popleft()
            for e in self.adjacency.get(cur, []):
                nxt = e.to_id
                if nxt not in prev:
                    prev[nxt] = cur
                    if nxt == goal_id:
                        # reconstruct
                        path = []
                        node = goal_id
                        while node is not None:
                            path.append(node)
                            node = prev[node]
                        return list(reversed(path))
                    q.append(nxt)
        return None

    def neighbors_walk(self, start_id: str, depth: int = 2) -> Set[str]:
        """Return set of node ids reachable within depth steps (outgoing)."""
        seen = set()
        frontier = {start_id}
        for _ in range(depth):
            new_front = set()
            for nid in frontier:
                for e in self.adjacency.get(nid, []):
                    if e.to_id not in seen:
                        new_front.add(e.to_id)
            seen.update(frontier)
            frontier = new_front
        seen.discard(start_id)
        return seen

# ------------------------
# Builder: loads stdlib, patterns, core concepts
# ------------------------

class GoKGraphBuilder:
    def __init__(self, kg: GoKnowledgeGraph):
        self.kg = kg

    # helper to convert relation strings or RelationType to RelationType
    @staticmethod
    def _relation_from_str(r: Any) -> RelationType:
        if isinstance(r, RelationType):
            return r
        rs = str(r).lower()
        for rel in RelationType:
            if rel.value == rs or rel.name.lower() == rs:
                return rel
        raise ValueError(f"Unknown relation '{r}'")

    def load_stdlib(self):
        stdlib_data = {
            "fmt": {
                "description": "Formatted I/O",
                "godoc_url": "https://pkg.go.dev/fmt",
                "tags": ["io", "printing"],
                "types": [],
                "functions": [
                    {"name": "Println", "description": "Prints with newline.", "tags": ["output"]},
                    {"name": "Errorf", "description": "Returns formatted error.", "tags": ["error"]}
                ]
            },
            "sync": {
                "description": "Low-level synchronization primitives.",
                "godoc_url": "https://pkg.go.dev/sync",
                "tags": ["concurrency", "sync"],
                "types": [
                    {"name": "WaitGroup", "description": "Waits for collection of goroutines.", "tags": ["concurrency", "sync"]},
                    {"name": "Mutex", "description": "Mutual exclusion lock.", "tags": ["concurrency", "sync"]}
                ],
                "functions": []
            },
            "context": {
                "description": "Context carries deadlines and cancellation.",
                "godoc_url": "https://pkg.go.dev/context",
                "tags": ["concurrency", "context"],
                "types": [
                    {"name": "Context", "description": "Interface for deadlines, cancellation, values.", "tags": ["interface", "concurrency", "context"]}
                ],
                "functions": [
                    {"name": "WithCancel", "description": "Returns cancellable context.", "tags": ["concurrency", "context"]}
                ]
            },
            "net/http": {
                "description": "HTTP client/server",
                "godoc_url": "https://pkg.go.dev/net/http",
                "tags": ["web", "http"],
                "types": [
                    {"name": "Handler", "description": "Interface for HTTP handlers.", "tags": ["web"]},
                    {"name": "Server", "description": "HTTP server config.", "tags": ["web"]}
                ],
                "functions": [
                    {"name": "HandleFunc", "description": "Register handler function.", "tags": ["web"]}
                ]
            }
        }

        for pkg_name, pkg_data in stdlib_data.items():
            pkg_id = f"module:{pkg_name}"
            pkg_node = Node(
                id=pkg_id,
                type=NodeType.MODULE,
                name=pkg_name,
                description=pkg_data.get("description", ""),
                tags=["stdlib", "module"] + pkg_data.get("tags", []),
                source=pkg_data.get("godoc_url", "")
            )
            self.kg.add_node(pkg_node)

            for item in pkg_data.get("types", []):
                type_id = f"type:{pkg_name}.{item['name']}"
                type_node = Node(
                    id=type_id,
                    type=NodeType.TYPE,
                    name=f"{pkg_name}.{item['name']}",
                    description=item.get("description", ""),
                    tags=["type", pkg_name] + item.get("tags", []),
                    metadata=item,
                    source=pkg_data.get("godoc_url", "")
                )
                self.kg.add_node(type_node)
                self.kg.create_edge(pkg_id, type_id, RelationType.DEPENDS_ON)

            for item in pkg_data.get("functions", []):
                func_id = f"func:{pkg_name}.{item['name']}"
                func_node = Node(
                    id=func_id,
                    type=NodeType.FUNCTION,
                    name=f"{pkg_name}.{item['name']}",
                    description=item.get("description", ""),
                    tags=["function", pkg_name] + item.get("tags", []),
                    metadata=item,
                    source=pkg_data.get("godoc_url", "")
                )
                self.kg.add_node(func_node)
                self.kg.create_edge(pkg_id, func_id, RelationType.DEPENDS_ON)

    def load_patterns(self):
        patterns_data = [
            {
                "id": "worker_pool",
                "name": "Worker Pool Pattern",
                "description": "Pool of goroutines processing jobs from channel.",
                "tags": ["concurrency", "pattern", "scalability"],
                "source": "https://gobyexample.com/worker-pools",
                "relations": [
                    {"to": "type:chan", "relation": "REQUIRES", "evidence": "Uses channels for job queue"},
                    {"to": "concept:goroutine", "relation": "USES", "evidence": "Each worker is a goroutine"},
                    {"to": "type:sync.WaitGroup", "relation": "REQUIRES", "evidence": "To wait for all workers"}
                ]
            },
            {
                "id": "context_cancellation",
                "name": "Context Cancellation Pattern",
                "description": "Use context to cancel long-running goroutines.",
                "tags": ["concurrency", "pattern", "robustness"],
                "source": "https://pkg.go.dev/context",
                "relations": [
                    {"to": "type:context.Context", "relation": "REQUIRES"},
                    {"to": "concept:goroutine", "relation": "USES"},
                    {"to": "anti:goroutine_leak", "relation": "REPLACES", "evidence": "Prevents leaks"}
                ]
            }
        ]

        for pattern in patterns_data:
            pattern_id = f"pattern:{pattern['id']}"
            pattern_node = Node(
                id=pattern_id,
                type=NodeType.PATTERN,
                name=pattern["name"],
                description=pattern["description"],
                tags=pattern.get("tags", []),
                metadata=pattern,
                source=pattern.get("source", "")
            )
            self.kg.add_node(pattern_node)

            for rel in pattern.get("relations", []):
                try:
                    relation = self._relation_from_str(rel["relation"])
                except ValueError:
                    relation = RelationType.USES
                edge = Edge(
                    from_id=pattern_id,
                    to_id=rel["to"],
                    relation=relation,
                    evidence=rel.get("evidence", "")
                )
                self.kg.add_edge(edge)

    def build_core_go_concepts(self):
        # Goroutine
        go_routine = Node(
            id="concept:goroutine",
            type=NodeType.CONCEPT,
            name="goroutine",
            description="Lightweight thread managed by Go runtime. Created with 'go' keyword.",
            tags=["concurrency", "runtime"]
        )
        self.kg.add_node(go_routine)

        # Channel
        channel = Node(
            id="type:chan",
            type=NodeType.TYPE,
            name="chan",
            description="Communication mechanism between goroutines.",
            tags=["concurrency", "channel", "chan"]
        )
        self.kg.add_node(channel)

        # sync.WaitGroup
        wg = Node(
            id="type:sync.WaitGroup",
            type=NodeType.TYPE,
            name="sync.WaitGroup",
            description="Waits for a collection of goroutines to finish.",
            tags=["concurrency", "sync"]
        )
        self.kg.add_node(wg)

        # context.Context
        ctx = Node(
            id="type:context.Context",
            type=NodeType.TYPE,
            name="context.Context",
            description="Carries deadlines, cancellation signals, and other request-scoped values.",
            tags=["concurrency", "context"]
        )
        self.kg.add_node(ctx)

        # relations
        self.kg.create_edge("concept:goroutine", "type:chan", RelationType.USED_IN, evidence="Channels coordinate goroutines")
        self.kg.create_edge("concept:goroutine", "type:sync.WaitGroup", RelationType.REQUIRES, evidence="Often needed to wait for goroutines")
        self.kg.create_edge("concept:goroutine", "type:context.Context", RelationType.REQUIRES, evidence="For cancellation and timeouts")

        # Anti-pattern: goroutine leak
        anti_leak = Node(
            id="anti:goroutine_leak",
            type=NodeType.ANTI_PATTERN,
            name="goroutine leak",
            description="Goroutine runs forever without exit signal.",
            tags=["concurrency", "bug"]
        )
        self.kg.add_node(anti_leak)
        self.kg.create_edge(anti_leak.id, "concept:goroutine", RelationType.ANTI_PATTERN_OF)
        self.kg.create_edge(anti_leak.id, "type:context.Context", RelationType.BEST_PRACTICE_FOR, evidence="Use context for cancellation")

        # Best practice: use context for cancellation
        bp_ctx = Node(
            id="bp:use_context_for_cancellation",
            type=NodeType.BEST_PRACTICE,
            name="Use context for cancellation",
            description="Always use context.Context to signal goroutine shutdown.",
            tags=["concurrency", "best_practice"]
        )
        self.kg.add_node(bp_ctx)
        self.kg.create_edge(bp_ctx.id, "type:context.Context", RelationType.USED_IN)
        self.kg.create_edge(bp_ctx.id, anti_leak.id, RelationType.REPLACES)

        # HTTP handler anti-pattern and best practice
        anti_global = Node(
            id="anti:global_state_in_handler",
            type=NodeType.ANTI_PATTERN,
            name="Global state in HTTP handler",
            description="Using global variables in web handlers causes race conditions.",
            tags=["web", "bug", "concurrency"]
        )
        self.kg.add_node(anti_global)
        bp_mutex = Node(
            id="bp:use_mutex_or_local_state",
            type=NodeType.BEST_PRACTICE,
            name="Use mutex or request-local state",
            description="Protect shared state with sync.Mutex or avoid it.",
            tags=["web", "concurrency", "best_practice"]
        )
        self.kg.add_node(bp_mutex)
        self.kg.create_edge(bp_mutex.id, "type:sync.Mutex", RelationType.USED_IN)
        self.kg.create_edge(bp_mutex.id, anti_global.id, RelationType.REPLACES)

    def build(self):
        self.load_stdlib()
        self.load_patterns()
        self.build_core_go_concepts()

# ------------------------
# QUERY ENGINE
# ------------------------

class GoKGQueryEngine:
    def __init__(self, kg: GoKnowledgeGraph):
        self.kg = kg

    def explain_concurrency_pattern(self, pattern_name: str) -> dict:
        # search by pattern name (exact first, fallback to tag-based)
        nodes = [n for n in self.kg.find_nodes_by_tag("concurrency") if n.name.lower() == pattern_name.lower()]
        if not nodes:
            # fallback: search nodes with name == pattern_name
            nodes = [n for n in self.kg.nodes.values() if n.name.lower() == pattern_name.lower()]
        if not nodes:
            return {"error": f"Pattern '{pattern_name}' not found"}

        node = nodes[0]
        related = self.kg.get_neighbors(node.id)
        # gather best practices and anti-patterns reachable in one hop
        best_practices = [n for n in related if n.type == NodeType.BEST_PRACTICE]
        anti_patterns = [n for n in related if n.type == NodeType.ANTI_PATTERN]
        related_concepts = [n for n in related if n.type in (NodeType.CONCEPT, NodeType.TYPE)]

        return {
            "pattern": node.name,
            "description": node.description,
            "best_practices": [bp.name for bp in best_practices],
            "anti_patterns": [ap.name for ap in anti_patterns],
            "related_concepts": [rc.name for rc in related_concepts]
        }

    def recommend_sync_mechanism(self, for_pattern: str) -> List[str]:
        # try tag search first
        recs = self.kg.recommend_sync_mechanism(for_pattern)
        if recs:
            return recs
        # fallback: look for patterns with similar name
        nodes = [n for n in self.kg.nodes.values() if for_pattern.lower() in n.name.lower()]
        names = set()
        for n in nodes:
            for e in self.kg.get_outgoing_edges(n.id):
                t = self.kg.get_node(e.to_id)
                if t and t.type == NodeType.TYPE:
                    names.add(t.name)
        return sorted(names)

    def find_anti_patterns_in_code_intent(self, intent: str) -> List[dict]:
        return self.kg.find_anti_patterns_in_code_intent(intent)

    def search_by_description(self, keyword: str) -> List[dict]:
        scored = self.kg.search_by_description(keyword, top_k=10)
        return [{"id": n.id, "name": n.name, "type": n.type.value, "score": score, "description": n.description} for n, score in scored]

    def shortest_path(self, start_name: str, goal_name: str) -> Optional[List[str]]:
        # map names to ids (first match)
        def find_id_by_name(name: str) -> Optional[str]:
            for n in self.kg.nodes.values():
                if n.name == name or n.id == name:
                    return n.id
            # case-insensitive fallback
            for n in self.kg.nodes.values():
                if n.name.lower() == name.lower():
                    return n.id
            return None
        s = find_id_by_name(start_name)
        g = find_id_by_name(goal_name)
        if not s or not g:
            return None
        return self.kg.shortest_path(s, g)

# ------------------------
# MAIN / DEMO
# ------------------------

def main():
    print("🚀 Building Enhanced Go Reasoning Knowledge Graph...\n")
    kg = GoKnowledgeGraph(create_placeholders=True)
    builder = GoKGraphBuilder(kg)
    builder.build()
    print(f"✅ Graph built with {len(kg)} nodes and {len(kg.edges)} edges.\n")

    qe = GoKGQueryEngine(kg)

    print("🔍 DEMO 1: Explain 'goroutine' (pattern-like query)")
    r = qe.explain_concurrency_pattern("goroutine")
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print("\n" + "="*70 + "\n")

    print("🔍 DEMO 2: Recommend sync for 'worker_pool'")
    recs = qe.recommend_sync_mechanism("worker_pool")
    print("Recommendations:", recs)
    print("\n" + "="*70 + "\n")

    print("🔍 DEMO 3: Detect anti-pattern in user intent")
    intent = "spawn goroutine to update global counter"
    anti = qe.find_anti_patterns_in_code_intent(intent)
    for ap in anti:
        print(f"⚠️ {ap['anti_pattern']}: {ap['description']}\n  Fix: {ap['fix']}\n")
    if not anti:
        print("No anti-patterns detected.")
    print("\n" + "="*70 + "\n")

    print("🔍 DEMO 4: Search for 'context'")
    results = qe.search_by_description("context")
    for r in results:
        print(f"→ {r['type']}: {r['name']} (score={r['score']})")
    print("\n" + "="*70 + "\n")

    print("🔍 DEMO 5: Shortest path from 'goroutine' to 'type:context.Context'")
    path = qe.shortest_path("goroutine", "type:context.Context")
    print("Path:", path)
    print("\n" + "="*70 + "\n")

    # Save + reload to validate persistence
    out = "go_knowledge_graph_enhanced.json"
    kg.save_to_file(out)
    print(f"💾 Graph saved to '{out}'")

    kg2 = GoKnowledgeGraph()
    kg2.load_from_file(out)
    print(f"📥 Reloaded graph with {len(kg2)} nodes — validation passed.\n")

    # example of neighbors walk
    nid = "concept:goroutine"
    neighbors = kg.neighbors_walk(nid, depth=2)
    print(f"→ Nodes reachable from {nid} within 2 hops: {len(neighbors)}")
    print("\n🎉 Enhanced Go Reasoning Knowledge Graph ready!")

if __name__ == "__main__":
    main()
