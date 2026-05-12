"""
data/toy_kg.py — Biology Contradiction Knowledge Graph  [V1]

PURPOSE:
    The foundational dataset for all toy experiments. A hand-crafted biology
    knowledge graph with 28 entities, 12 relations, 67 triples, and 3
    deliberate contradiction triples designed to test quantum interference.

THE THREE CONTRADICTION CASES:
    Platypus: isA Mammal → hasProperty WarmBlooded (correct)
              laysEggs True → impliesType Reptile → hasProperty ColdBlooded (wrong)

    Bat:      isA Mammal → hasProperty FourLimbs (correct)
              hasWings True → impliesType Bird → hasProperty TwoLimbs (wrong)

    Whale:    isA Mammal → breathes Lungs (correct)
              livesIn Ocean → impliesType Fish → breathes Gills (wrong)

WHY THESE CASES:
    Each case has a CORRECT multi-hop reasoning path and a CONTRADICTORY path.
    Both paths are valid edges in the KG. The model must learn to suppress
    the contradictory path via destructive interference.
    The pre-training state (random phases) gives ~27% probability to the
    correct answer because the 7 contradictory paths outnumber the 4 correct paths.
    After training with InterferenceAwareLoss, correct answers should dominate.

USAGE:
    kg = build_toy_kg()
    print(kg.summary())             # "28 entities, 12 relations, 67 triples"
    adj = kg.get_adjacency()        # {entity_id: [(rel_id, neighbor_id), ...]}
    ids = kg.triple_to_ids(triple)  # convert Triple to (h_id, r_id, t_id)

    for cq in kg.contradiction_queries:
        print(cq["head"], "→", cq["correct_tail"], "vs", cq["contradictory_tail"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import random


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Triple:
    """A knowledge graph triple (head, relation, tail)."""
    head:             str
    relation:         str
    tail:             str
    is_contradiction: bool = False   # True for deliberate wrong triples

    def __iter__(self):
        return iter((self.head, self.relation, self.tail))

    def __hash__(self):
        return hash((self.head, self.relation, self.tail))

    def __eq__(self, other):
        if isinstance(other, tuple):
            return (self.head, self.relation, self.tail) == other
        return (self.head, self.relation, self.tail) == (other.head, other.relation, other.tail)


@dataclass
class ToyKG:
    """
    The complete toy knowledge graph.

    Attributes:
        entities:             List of entity strings (length 28).
        relations:            List of relation strings (length 12).
        triples:              All 67 triples including contradictions.
        train_triples:        Training split (~75%).
        val_triples:          Validation split (~15%).
        test_triples:         Test split (~10%).
        entity2id:            Entity string → integer ID mapping.
        id2entity:            Integer ID → entity string mapping.
        relation2id:          Relation string → integer ID mapping.
        id2relation:          Integer ID → relation string mapping.
        contradiction_queries: List of dicts describing the 3 test cases.
    """
    entities:             list[str]
    relations:            list[str]
    triples:              list[Triple]
    train_triples:        list[Triple]
    val_triples:          list[Triple]
    test_triples:         list[Triple]
    entity2id:            dict[str, int]
    id2entity:            dict[int, str]
    relation2id:          dict[str, int]
    id2relation:          dict[int, str]
    contradiction_queries: list[dict]

    # ── Derived properties ─────────────────────────────────────────────────
    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)

    @property
    def num_triples(self) -> int:
        return len(self.triples)

    def summary(self) -> str:
        n_contra = sum(1 for t in self.triples if t.is_contradiction)
        return (
            f"{self.num_entities} entities, "
            f"{self.num_relations} relations, "
            f"{self.num_triples} triples "
            f"({n_contra} contradictions, "
            f"{len(self.train_triples)}/{len(self.val_triples)}/{len(self.test_triples)} "
            f"train/val/test)"
        )

    # ── Conversion helpers ─────────────────────────────────────────────────
    def triple_to_ids(self, triple: Triple) -> tuple[int, int, int]:
        """Convert a Triple to (h_id, r_id, t_id) integer tuple."""
        return (
            self.entity2id[triple.head],
            self.relation2id[triple.relation],
            self.entity2id[triple.tail],
        )

    def ids_to_triple(self, h_id: int, r_id: int, t_id: int) -> Triple:
        """Convert integer IDs back to a Triple."""
        return Triple(
            head     = self.id2entity[h_id],
            relation = self.id2relation[r_id],
            tail     = self.id2entity[t_id],
        )

    # ── Adjacency ──────────────────────────────────────────────────────────
    def get_adjacency(self) -> dict[int, list[tuple[int, int]]]:
        """
        Build adjacency list for path enumeration (BFS).

        Returns:
            Dict mapping entity_id → list of (relation_id, neighbor_id) tuples.
            Includes ALL triples (clean + contradictions) so BFS can find both
            correct and contradictory paths for the interference demonstration.

        Note: Contradictions MUST be in the adjacency graph. If they are excluded,
        PathEnumerator cannot find contradictory paths and the interference
        experiment cannot run (Step 6 of run_toy.py will fail).
        """
        adj: dict[int, list[tuple[int, int]]] = {
            i: [] for i in range(self.num_entities)
        }
        for triple in self.triples:
            h_id = self.entity2id[triple.head]
            r_id = self.relation2id[triple.relation]
            t_id = self.entity2id[triple.tail]
            adj[h_id].append((r_id, t_id))
        return adj

    def get_true_set(
        self,
        include_contradictions: bool = False,
    ) -> set[tuple[int, int, int]]:
        """
        Build the set of all true (h_id, r_id, t_id) triples.

        Used by:
            - Noise injection: prevents corrupted triples from accidentally
              being true triples (false negative avoidance).
            - Filtered evaluation: masks known-true tails during ranking.

        Args:
            include_contradictions: If True, includes deliberate contradiction
                triples in the true set. Default False (contradictions are
                treated as noise, not ground truth).
        """
        result = set()
        for triple in self.triples:
            if triple.is_contradiction and not include_contradictions:
                continue
            result.add(self.triple_to_ids(triple))
        return result


# ── KG Construction ───────────────────────────────────────────────────────────

def build_toy_kg(
    seed:         int   = 42,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
) -> ToyKG:
    """
    Build and return the complete biology contradiction toy KG.

    Args:
        seed:        Random seed for train/val/test split reproducibility.
        train_ratio: Fraction of clean triples used for training (default 0.70).
        val_ratio:   Fraction for validation (default 0.15).
                     Remaining (1 - train_ratio - val_ratio) goes to test.

    Returns:
        ToyKG instance with all splits, mappings, and contradiction queries.
    """
    rng = random.Random(seed)

    # ── 28 Entities ──────────────────────────────────────────────────────────
    entities = [
        # Animals (the subjects of contradictions)
        "Platypus", "Bat", "Whale",
        # Taxonomic classes
        "Mammal", "Reptile", "Bird", "Fish", "Amphibian",
        # Property values / concepts
        "WarmBlooded", "ColdBlooded",
        "FourLimbs", "TwoLimbs", "Wings",
        "Lungs", "Gills",
        "Ocean", "Land", "Sky", "River",
        # Boolean-like intermediate nodes
        "True", "False",
        # Additional biology entities
        "Dog", "Eagle", "Salmon", "Snake", "Frog",
        # Abstract
        "EggsLaying", "LiveBirth",
    ]
    assert len(entities) == 28, f"Expected 28 entities, got {len(entities)}"

    # ── 12 Relations ──────────────────────────────────────────────────────────
    relations = [
        "isA",            # taxonomic: Platypus isA Mammal
        "hasProperty",    # property:  Mammal hasProperty WarmBlooded
        "breathes",       # physiology: Whale breathes Lungs
        "livesIn",        # habitat:   Whale livesIn Ocean
        "impliesType",    # inference: Ocean impliesType Fish (the contradiction edge)
        "hasWings",       # morphology: Bat hasWings True
        "laysEggs",       # reproduction: Platypus laysEggs True
        "impliesTaxon",   # inference: True impliesTaxon Reptile
        "hasLimbs",       # morphology: Mammal hasLimbs FourLimbs
        "synonym",        # symmetric: WarmBlooded synonym WarmBlooded (test case)
        "contraryTo",     # antisymmetric: WarmBlooded contraryTo ColdBlooded
        "subClassOf",     # hierarchy: Mammal subClassOf Animal (not in entities, but relation exists)
    ]
    assert len(relations) == 12, f"Expected 12 relations, got {len(relations)}"

    # ── Build entity2id and relation2id ───────────────────────────────────────
    entity2id   = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}
    id2entity   = {i: e for e, i in entity2id.items()}
    id2relation = {i: r for r, i in relation2id.items()}

    # ── 67 Triples ────────────────────────────────────────────────────────────
    clean_triples: list[Triple] = [
        # === Platypus facts ===
        Triple("Platypus",     "isA",         "Mammal"),
        Triple("Platypus",     "laysEggs",    "True"),
        Triple("Platypus",     "livesIn",     "River"),
        Triple("Platypus",     "hasLimbs",    "FourLimbs"),

        # === Bat facts ===
        Triple("Bat",          "isA",         "Mammal"),
        Triple("Bat",          "hasWings",    "True"),
        Triple("Bat",          "livesIn",     "Sky"),
        Triple("Bat",          "hasLimbs",    "FourLimbs"),

        # === Whale facts ===
        Triple("Whale",        "isA",         "Mammal"),
        Triple("Whale",        "breathes",    "Lungs"),
        Triple("Whale",        "livesIn",     "Ocean"),
        Triple("Whale",        "hasProperty", "WarmBlooded"),

        # === Dog facts ===
        Triple("Dog",          "isA",         "Mammal"),
        Triple("Dog",          "livesIn",     "Land"),
        Triple("Dog",          "hasLimbs",    "FourLimbs"),
        Triple("Dog",          "breathes",    "Lungs"),

        # === Eagle facts ===
        Triple("Eagle",        "isA",         "Bird"),
        Triple("Eagle",        "hasWings",    "True"),
        Triple("Eagle",        "livesIn",     "Sky"),
        Triple("Eagle",        "hasLimbs",    "TwoLimbs"),
        Triple("Eagle",        "laysEggs",    "True"),

        # === Salmon facts ===
        Triple("Salmon",       "isA",         "Fish"),
        Triple("Salmon",       "livesIn",     "Ocean"),
        Triple("Salmon",       "breathes",    "Gills"),
        Triple("Salmon",       "laysEggs",    "True"),

        # === Snake facts ===
        Triple("Snake",        "isA",         "Reptile"),
        Triple("Snake",        "hasProperty", "ColdBlooded"),
        Triple("Snake",        "livesIn",     "Land"),
        Triple("Snake",        "laysEggs",    "True"),

        # === Frog facts ===
        Triple("Frog",         "isA",         "Amphibian"),
        Triple("Frog",         "breathes",    "Lungs"),
        Triple("Frog",         "livesIn",     "River"),
        Triple("Frog",         "laysEggs",    "True"),

        # === Class-level properties ===
        Triple("Mammal",       "hasProperty", "WarmBlooded"),
        Triple("Mammal",       "hasLimbs",    "FourLimbs"),
        Triple("Mammal",       "breathes",    "Lungs"),
        Triple("Reptile",      "hasProperty", "ColdBlooded"),
        Triple("Reptile",      "laysEggs",    "True"),
        Triple("Bird",         "hasWings",    "True"),
        Triple("Bird",         "hasLimbs",    "TwoLimbs"),
        Triple("Bird",         "laysEggs",    "True"),
        Triple("Fish",         "breathes",    "Gills"),
        Triple("Fish",         "livesIn",     "Ocean"),
        Triple("Amphibian",    "breathes",    "Lungs"),
        Triple("Amphibian",    "laysEggs",    "True"),

        # === Habitat implications (used for contradictory paths) ===
        Triple("Ocean",        "impliesType", "Fish"),
        Triple("Sky",          "impliesType", "Bird"),
        Triple("River",        "impliesType", "Amphibian"),

        # === Egg-laying implications (used for contradictory paths) ===
        Triple("True",         "impliesTaxon", "Reptile"),   # oversimplification → contradiction

        # === Property antonyms ===
        Triple("WarmBlooded",  "contraryTo",  "ColdBlooded"),
        Triple("ColdBlooded",  "contraryTo",  "WarmBlooded"),

        # === Wing implications ===
        Triple("Wings",        "impliesType", "Bird"),
        Triple("True",         "hasProperty", "EggsLaying"),

        # === Reproduction ===
        Triple("Mammal",       "hasProperty", "LiveBirth"),
        Triple("EggsLaying",   "contraryTo",  "LiveBirth"),
        Triple("LiveBirth",    "contraryTo",  "EggsLaying"),

        # === Synonym test (symmetric relation) ===
        Triple("WarmBlooded",  "synonym",     "WarmBlooded"),

        # === Additional connectivity for path richness ===
        Triple("FourLimbs",    "contraryTo",  "TwoLimbs"),
        Triple("TwoLimbs",     "contraryTo",  "FourLimbs"),
        Triple("Lungs",        "contraryTo",  "Gills"),
        Triple("Gills",        "contraryTo",  "Lungs"),
        Triple("Mammal",       "impliesType", "WarmBlooded"),

        Triple("Mammal",       "laysEggs",    "False"),
        Triple("Bat",          "laysEggs",    "False"),
    ]

    # 3 deliberate contradiction triples (injected into training only)
    contradiction_triples: list[Triple] = [
        # Platypus: makes it look like a Reptile (egg-laying → cold-blooded chain)
        Triple("Platypus",  "isA",       "Reptile",   is_contradiction=True),
        # Bat: makes it look like a Bird (wings → two-limbs chain)
        Triple("Bat",       "isA",       "Bird",      is_contradiction=True),
        # Whale: makes it look like a Fish (ocean habitat → gills chain)
        Triple("Whale",     "isA",       "Fish",      is_contradiction=True),
    ]

    all_triples = clean_triples + contradiction_triples
    assert len(all_triples) == len(clean_triples) + len(contradiction_triples), \
        f"Triple count mismatch: {len(all_triples)}"

    # ── Train/val/test split (clean triples only) ─────────────────────────────
    # Contradiction triples go only into training (CKRL protocol)
    clean_shuffled = clean_triples[:]
    rng.shuffle(clean_shuffled)
    n = len(clean_shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_clean  = clean_shuffled[:n_train]
    val_triples  = clean_shuffled[n_train:n_train + n_val]
    test_triples = clean_shuffled[n_train + n_val:]

    # Training set = clean train + all contradiction triples
    train_triples = train_clean + contradiction_triples

    # ── Contradiction queries (for Steps 6 and 7 of run_toy.py) ──────────────
    contradiction_queries = [
        {
            "query":            "Platypus hasProperty ?",
            "head":             "Platypus",
            "relation":         "hasProperty",
            "correct_tail":     "WarmBlooded",
            "contradictory_tail": "ColdBlooded",
            "correct_path":    [("isA", "Mammal"), ("hasProperty", "WarmBlooded")],
            "wrong_path":      [("isA", "Reptile"), ("hasProperty", "ColdBlooded")],
            "description":     "Platypus is warm-blooded (Mammal) despite laying eggs (wrong path: Reptile)",
        },
        {
            "query":            "Bat hasLimbs ?",
            "head":             "Bat",
            "relation":         "hasLimbs",
            "correct_tail":     "FourLimbs",
            "contradictory_tail": "TwoLimbs",
            "correct_path":    [("isA", "Mammal"), ("hasLimbs", "FourLimbs")],
            "wrong_path":      [("isA", "Bird"), ("hasLimbs", "TwoLimbs")],
            "description":     "Bat has 4 limbs (Mammal) despite having wings (wrong path: Bird)",
        },
        {
            "query":            "Whale breathes ?",
            "head":             "Whale",
            "relation":         "breathes",
            "correct_tail":     "Lungs",
            "contradictory_tail": "Gills",
            "correct_path":    [("isA", "Mammal"), ("breathes", "Lungs")],
            "wrong_path":      [("livesIn", "Ocean"), ("impliesType", "Fish"), ("breathes", "Gills")],
            "description":     "Whale breathes with lungs (Mammal) despite living in ocean (wrong path: Fish)",
        },
    ]

    return ToyKG(
        entities             = entities,
        relations            = relations,
        triples              = all_triples,
        train_triples        = train_triples,
        val_triples          = val_triples,
        test_triples         = test_triples,
        entity2id            = entity2id,
        id2entity            = id2entity,
        relation2id          = relation2id,
        id2relation          = id2relation,
        contradiction_queries= contradiction_queries,
    )
