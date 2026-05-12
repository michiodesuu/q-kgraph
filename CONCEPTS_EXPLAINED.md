# Quantum KG Concepts Explained — Plain Language Guide

> **Who this is for:** You built this system. This document explains the math and ideas behind
> every key concept in plain English, with analogies, so you can explain them in a presentation
> to someone who is not a quantum physicist or math expert.
>
> Written: 2026-05-07

---

## Table of Contents

1. [What are Cross-Terms vs Self-Terms (Interference)?](#1-cross-terms-vs-self-terms)
2. [Non-Commutative 4D Space, quantum_logic_lattice.py, multi_view_aggregator.py](#2-non-commutative-4d-and-logic-lattice)
3. [Unconstrained Bell State Matrices and Weyl Corrections](#3-bell-state-and-weyl-corrections)
4. [quantum_reasoner.py vs quaternion_reasoner.py](#4-quantum-vs-quaternion-reasoner)
5. [Why Phase Separation in interference_loss.py Matters](#5-phase-separation-loss)
6. [V6's ListNet Ranking Loss](#6-listnet-ranking-loss)
7. [MRR and Hits@K Metrics](#7-mrr-and-hitsk)
8. [Differences Between Every Loss in the Codebase](#8-all-losses-compared)
9. [What is a Quaternion and How Does it Differ From Quantum?](#9-quaternion-vs-quantum)
10. [Kolmogorov-Smirnov Test (KS-Test)](#10-ks-test)
11. [noise_guarantee.py and interference_guarantee.py Explained](#11-guarantee-files)
12. [V5 Polarity Gradients and Unitary Spanning Proofs](#12-v5-polarity-and-spanning)
13. [Real Units and Imaginary Units of Complex Vectors](#13-real-and-imaginary-units)
14. [V1 Diagonal Matrices, V4 Quaternion Logic, V5 Matrix Exponential and U(d)](#14-unitary-evolution-per-version)
15. [1-Hop Born Rule Proxy, Multi-Hop Decoherence Branch, Teleportation Branch](#15-three-scoring-branches)
16. [What is Quantum Teleportation (in this context)?](#16-quantum-teleportation)
17. [Decoherence: Pure State, Per-Relation Channel, Quantum Trace Operation](#17-decoherence-explained)
18. [Contextuality Loss](#18-contextuality-loss)

---

## 1. Cross-Terms vs Self-Terms

### The Short Answer

When you add two numbers and then square them, you get extra "bonus" terms that wouldn't exist
if you squared them separately. Those bonus terms are called **cross-terms**. In this project,
cross-terms are the mechanism that makes quantum reasoning different from classical reasoning.

### The Analogy

Imagine two radio signals being broadcast at the same time in the same room:

- **Signal A** has strength 3
- **Signal B** has strength 4

If you add them before measuring the power:
```
Power = (3 + 4)² = 49
```

If you measure each separately and add:
```
Power = 3² + 4² = 9 + 16 = 25
```

The difference (49 - 25 = 24) is the **cross-term**. It represents the signals **interfering**
with each other — they interact and create more (or less) total energy than they would alone.

### In the Quantum KG Model

The scoring formula is:
```
Score(head → tail) = |sum of all path amplitudes|²
                   = |A₁ + A₂ + A₃ + ...|²
```

When you expand this squared sum:
```
|A₁ + A₂|² = |A₁|² + |A₂|² + 2 · Re(A₁ · Ā₂)
               ↑         ↑          ↑
           self-term  self-term   CROSS-TERM
```

- **Self-terms** (`|A₁|²`): The contribution of each path by itself. These are always positive.
  This is what classical models compute — just sum up individual path strengths.

- **Cross-terms** (`2 · Re(A₁ · Ā₂)`): The interaction between pairs of paths. These can be
  **positive** (constructive interference → correct answer gets boosted) or
  **negative** (destructive interference → wrong answer gets suppressed).

### Why This Matters

Classical models only see self-terms. They can only say "path A supports answer X" or
"path B also supports answer X." They cannot say "path A and path B CANCEL EACH OTHER OUT
for wrong answer Y."

The quantum model can **actively suppress wrong answers** via destructive interference —
making the cross-terms negative for incorrect predictions. This is the core claim of the paper.

---

## 2. Non-Commutative 4D Space, quantum_logic_lattice.py, multi_view_aggregator.py

### Non-Commutative 4D Space

**"Non-commutative"** means: the **order of operations matters**. In regular arithmetic,
`3 × 5 = 5 × 3` (commutative). In matrix math and quaternions, `A × B ≠ B × A` in general.

**Why is order important for knowledge graphs?**

Consider reasoning: "Person → born_in → City → capital_of → Country"

That's different from: "Person → capital_of → ??? → born_in → Country"

The **path order** matters. Non-commutativity means the model can respect that
"born_in then capital_of" is a different operation than "capital_of then born_in."

**"4D space"** refers to quaternions (see Q9), which use 4 numbers (a, b, c, d) to represent
a point, instead of the usual 2 (real + imaginary) in complex numbers.

---

### quantum_logic_lattice.py (V4)

**What problem does it solve?**

The earlier versions (V1-V3) only looked at **local paths** — "does this path lead to the answer?"
They ignored **global logic** — "is this answer even logically possible given everything else
in the graph?"

**The core idea — a "logic firewall":**

In logic, if statement P is true, then NOT-P must be false. They can't both be true. The logic
lattice enforces this **orthogonality** rule on the knowledge graph.

Think of it like a **rule book**:
- If "Einstein was born in Germany" is true (✓)
- Then "Einstein was born in France" should be impossible (⊥)
- These two should be **completely separate / orthogonal** — one scores high, the other scores 0

**How it works in code (3 loss components):**

```
L_ELoss (Entity Loss):
  Each entity embedding should act like a valid logical statement.
  It should be "binary-like" — either clearly true or clearly false,
  not ambiguously in between.

L_LLoss (Logical Relationship Loss):
  If (Einstein, born_in, Germany) is TRUE
  and (Einstein, born_in, France) is FALSE,
  then their embeddings should be ORTHOGONAL (perpendicular).
  Orthogonal = opposite = contradictory.

L_MLoss (Membership Loss):
  Entities should "belong" to their correct domain.
  E.g., "Einstein" should be in the "Person" category for the "born_in" relation,
  not in the "City" category.
```

**V4 extension:** This lattice is applied to entire **paths**, not just individual triples.
A path that violates global logic constraints gets multiplied by a lower weight (αᵢ), making
it contribute less to the final score.

---

### multi_view_aggregator.py (V4)

**What problem does it solve?**

By V4, the model has multiple paths from head to tail. But how should they be combined?
The MultiViewInterferenceAggregator separates this into **two explicit parts**:

```
Total Score = Self-Terms + Cross-Terms (interference)
            = Σ|Aᵢ|² + 2Σ Re(Aᵢ · Āⱼ) × λᵢⱼ
```

The key addition: `λᵢⱼ` — the **logic weights** from the QuantumLogicLattice.

If path i and path j are logically contradictory (orthogonal in logic space), their
cross-term is **suppressed** (`λᵢⱼ ≈ 0`). If they are logically consistent, their
cross-term is **amplified** (`λᵢⱼ ≈ 1`).

This means the model learns to let **logically compatible paths interfere constructively**
and **logically contradictory paths cancel each other out** — just like real quantum mechanics.

---

## 3. Unconstrained Bell State Matrices and Weyl Corrections

### Background: What is a Bell State?

In quantum physics, a **Bell state** is a special configuration of two particles that are
"entangled" — measuring one instantly tells you something about the other, regardless of
distance. It's the purest form of quantum correlation.

In the knowledge graph context, an **entangled pair** is a (head entity, tail entity) pair
connected by a relation. The Bell state encodes the **full relationship** between them,
richer than just a single vector direction.

---

### Unconstrained Bell State Matrices (V6)

In earlier versions (V1-V3), the relation operators were forced to be **unitary matrices** —
mathematically perfect "rotation" matrices that preserve length and can be undone exactly.

**V6 removes this constraint.** The `BellStateRelation` is just a **free complex matrix**:

```python
M_r ∈ ℂ^(d×d)   # any d×d complex matrix, no restrictions
```

**Why remove the constraint?**

Unitary matrices can only rotate and reflect — they cannot scale or mix amplitudes in arbitrary
ways. A free (unconstrained) matrix can represent **richer relationships**:
- Asymmetric relations (A causes B, but B does not cause A)
- Many-to-one mappings (many heads map to one tail via a relation)
- Partial information transfer (not all amplitude is preserved)

**Why it doesn't blow up:** The matrix is **Frobenius-normalized** — divided by its total
"size" — so it stays bounded:

```
M̃_r = M_r / ‖M_r‖_F   (divide by the total magnitude of all entries)
```

Think of it like: "whatever shape this matrix has, normalize it so its overall volume = 1."

---

### Weyl Corrections (Generalized Pauli Corrections, V6)

**Analogy:** In quantum teleportation, after you "send" a quantum state, the receiver gets
a slightly scrambled version. To fix it, they need to apply a **correction operation**. There's
a finite set of possible corrections — in 2D quantum computing, these are the 4 Pauli matrices
(X, Y, Z, I). For d-dimensional systems, they're called **Weyl operators**.

**What are Weyl operators?**

For a d-dimensional system, there are exactly d² Weyl operators `C_{mn}`. Together, they
form a complete "toolbox" — any possible error or transformation can be expressed as a
combination of them.

```
C_{mn}|j⟩ = ω^(nj) |(j+m) mod d⟩
```

In plain English: each Weyl operator does two things:
1. **Shift** the state by m steps (like shifting an array by m positions)
2. **Rotate** by a phase that depends on n and the current position

**Why n_corrections = 16 in V6?**

With d=16 (or d²=256 for d=16), using 16 corrections covers a meaningful subset of the
d² possible correction operators. More corrections = richer expressiveness but higher
computation cost. 16 was chosen as the practical sweet spot.

**In practice (V6 TeleportationScorer):**

```python
score = |Σₖ √qₖ · ⟨t|Cₖ† · M_r|h⟩|²
         ↑                ↑      ↑
    learned     Weyl     relation  head
    weights   correction  matrix   state
```

The model learns which Weyl correction `Cₖ` is most useful for each query, via a small
neural network (`DifferentiableBellMeasurement`) that outputs weights `qₖ`.

---

## 4. quantum_reasoner.py vs quaternion_reasoner.py

These are two different model families — they approach the problem from completely different
mathematical frameworks.

---

### quantum_reasoner.py (V1 → V5 → V6)

**Core idea:** Entities are **quantum states** (complex vectors in Hilbert space).
Relations are **unitary operators** (quantum gates that rotate states).
Scoring is **Born rule** (squared inner product = probability of measurement).

```
Entity:   h ∈ ℂᵈ  (complex vector of dimension d)
Relation: U_r ∈ U(d)  (unitary rotation matrix)
Score:    |⟨t|U_r|h⟩|²  (how much does rotated head overlap with tail?)
```

**Trajectory:**
- V1: DiagonalUnitary (simple phase rotations per dimension)
- V2: Added PhaseSeparationLoss to prevent Phase Collapse
- V3: MatrixExpUnitary (full Hermitian generator, much richer rotations)
- V5: InterferencePolarityLoss, formal theorems
- V6: Added teleportation scoring, decoherence channels, ranking loss

**Main file structure:**
- `QuantumStateEncoder`: Stores entity embeddings (real + imaginary separately)
- `score_triple()`: Fast 1-hop scoring for training
- `score_with_paths()`: Full multi-hop interference scoring for evaluation
- `analyze_interference()`: Outputs phase statistics, cross-terms for paper figures

---

### quaternion_reasoner.py (V4)

**Core idea:** Entities live in **4D hypercomplex space** (quaternions, not 2D complex).
Relations are **Hamilton product rotations** (3D rotations in 4D space).
Scoring is **quaternion inner product**.

```
Entity:   h = (a, b, c, d) ∈ ℍ  (4 real numbers forming a quaternion)
Relation: r ∈ ℍ¹  (unit quaternion = 3D rotation)
Score:    ⟨h ⊗ r, t⟩  (dot product after rotating head by relation)
```

**Additional innovations in V4:**
1. `QuantumLogicLattice` — global logical consistency (see Q2)
2. `MultiViewInterferenceAggregator` — explicit cross-terms with logic weights (see Q2)
3. `DynamicQuantumRouter` — routes each query to either fast (classical), medium (quantum), or
   full path depending on how "confident" the model is about the query complexity

**Key difference from quantum_reasoner.py:**
| Feature | quantum_reasoner | quaternion_reasoner |
|---------|-----------------|---------------------|
| State space | Complex ℂᵈ | 4D quaternion ℍ |
| Relation | Unitary matrix U(d) | Unit quaternion rotation |
| Interference | Born rule |²  | Lattice-weighted cross-terms |
| Global logic | No | Yes (LogicLattice) |
| Routing | No | Yes (DynamicRouter) |
| V6 features | Yes | No |

**Why does quantum_reasoner win on benchmarks?**

The complex Hilbert space gives **d² degrees of freedom per relation** (V5 MatrixExp),
while quaternions only give **3 degrees of freedom** (a unit quaternion on S³).
The richer relation space of complex quantum models outweighs the logical consistency
gains from the quaternion approach on standard KG benchmarks.

---

## 5. Why Phase Separation in interference_loss.py Matters

### What is Phase Collapse?

Every complex number has two parts: a **magnitude** (how large it is) and a **phase**
(which direction it points on the complex circle).

```
Complex number z = |z| · e^(iφ)
                    ↑         ↑
                magnitude   phase angle
```

**Phase Collapse** is when all the phase angles `φ` drift to 0 (or all become equal).
If all phases are 0, then all complex numbers become **real numbers** — and real numbers
cannot produce the destructive interference (negative cross-terms) that the model relies on.

Think of it like: if you have a wave machine but all waves are perfectly synchronized
(same phase), you can only get constructive interference (bigger waves), never destructive
interference (cancel out). The model needs **varied phases** to work.

---

### PhaseSeparationLoss

The loss in `interference_loss.py` solves this by actively **pushing positive and negative
answer phases apart**:

```python
L_phase = mean( cos(phase_of_positive_path - phase_of_negative_path) )
```

- When phases are the same → `cos(0) = 1` → loss is HIGH → gradient pushes them apart
- When phases differ by π (180°) → `cos(π) = -1` → loss is LOW → this is the goal

**Why π (180°) specifically?**

If the positive answer's amplitude points in direction +φ and the negative answer points in
direction -φ (i.e., they are 180° apart), then when you take the Born rule sum:

```
For positive: |A_pos + A_other|² → cross-term is POSITIVE (constructive) → high score ✓
For negative: |A_neg + A_other|² → cross-term is NEGATIVE (destructive) → low score ✓
```

A 180° phase difference is **the maximum possible destructive interference**.

---

### InterferenceRegularization

Companion regularization that **penalizes small imaginary components**:

```python
L_reg = -mean( ||Im(entity_states)||² + ||phases(unitary)||² )
```

This directly punishes Phase Collapse by making small imaginary parts costly.
If entity states become purely real (all `Im = 0`), this term becomes very large,
forcing the optimizer to keep imaginary parts alive.

---

### Why Not Just Use a Standard Loss?

Standard losses like BCE only care about "is the score for the positive triple higher than
for the negative triple?" They don't care **why** the score is higher.

PhaseSeparationLoss cares about the **geometric reason** — the score should be higher because
of interference structure, not just because of larger raw magnitudes. This preserves the
quantum mechanical interpretation of the model.

---

## 6. V6's ListNet Ranking Loss

### What Problem Does it Solve?

The standard training loss (BCE = Binary Cross Entropy) treats each triple independently:
"Is (Einstein, born_in, Germany) true? Yes/No." It doesn't compare triples against each other.

But the evaluation metric MRR asks: "When I rank ALL possible tail entities, does Germany
appear near the top?" This is a **ranking** question, not just a yes/no question.

**The mismatch:** Training optimizes binary classification, but evaluation measures ranking.
This is why V5 had MRR ≈ 0.271 despite training seeming to converge — the model learned
"Germany is probably true" but not "Germany should rank above France and China."

---

### What is ListNet?

ListNet is a ranking loss that works like a **weighted vote**:

1. Compute scores for the positive answer AND K random wrong answers
2. Apply softmax to get probabilities (like: "how confident is the model in each one?")
3. Penalize the model whenever the positive answer doesn't get the highest probability

```python
L = -log( softmax([score_positive, score_neg_1, ..., score_neg_K])[0] )
```

In plain English: "How surprised should we be that the positive answer didn't rank #1
among these K+1 candidates?" The more surprised = the higher the loss.

---

### Why Is the Gradient Always Helpful?

```
∂L/∂score_positive = -(1 - probability_of_positive)
```

This is always **negative** (between -1 and 0). So gradient descent **always increases** the
positive score, no matter what. There's no "dead zone" like in hinge/margin losses where
the gradient is 0 once the margin is satisfied.

Similarly, gradients for negative scores are always positive — gradient descent always
decreases negative scores.

**Result:** The positive score and negative scores always drift further apart during training,
directly optimizing rank position → directly optimizing MRR.

---

### V6 Configuration

```yaml
use_ranking_loss: true
ranking_loss_weight: 0.3    # weight in combined loss
```

K = 64 negative samples per positive triple.

---

## 7. MRR and Hits@K

These are the two main evaluation metrics for knowledge graph completion.

---

### MRR (Mean Reciprocal Rank)

**The task:** Given (Einstein, born_in, ???), rank all ~14,000 entities.
"Germany" should be rank 1.

**Reciprocal Rank:** If Germany is at rank 3, the score is 1/3.
If Germany is at rank 1, the score is 1/1 = 1.0 (perfect).
If Germany is at rank 100, the score is 1/100 = 0.01 (terrible).

**Mean Reciprocal Rank:** Average this across all test queries.

```
MRR = (1/N) × Σ (1/rank_of_correct_answer)
```

**What good MRR looks like:**
- 1.0 = always rank #1 (perfect, impossible in practice)
- 0.3 = correct answer is at rank ~3 on average (good for this task)
- 0.1 = correct answer is at rank ~10 on average (mediocre)
- 0.05 = correct answer at rank ~20 (bad)

**V6 achieves MRR ≈ 0.312** on FB15k-237 (better than V5's 0.271).

---

### Hits@K

**The task:** Same ranking. Is the correct answer in the top K?

```
Hits@1 = fraction of queries where correct answer is rank 1
Hits@3 = fraction of queries where correct answer is in top 3
Hits@10 = fraction of queries where correct answer is in top 10
```

**Example:** If Hits@10 = 0.50, it means in 50% of queries, the correct answer appeared
in the model's top 10 suggestions.

**V6 results:**
```
Hits@1  ≈ 0.221  (correct answer is #1 in 22% of queries)
Hits@3  ≈ 0.345  (in top 3 in 34% of queries)
Hits@10 ≈ 0.507  (in top 10 in 51% of queries)
```

---

### Filtered vs Unfiltered

**Problem:** The graph has many true facts. If (Einstein, born_in, Germany) is the test query,
maybe (Berlin, born_in, Germany) is ALSO in the training set. If the model ranks Berlin above
Germany, that's actually correct — it just found a different true fact.

**Unfiltered MRR** penalizes this. **Filtered MRR** removes all other known-true triples
from the ranking before scoring. Filtered is the standard reported metric because it
fairly evaluates the model's knowledge.

**Filtered MRR is always ≥ Unfiltered MRR.** The gap (about +0.045 for V6) tells you how
many other true facts are being retrieved in the top-10.

---

## 8. Differences Between Every Loss in the Codebase

### Training Losses (grouped by purpose)

#### Group 1: Base Correctness Losses

**`training/losses.py` — BCE Loss (all versions)**
```
Purpose: Basic "is this triple true or false?" signal
Formula: -[y·log(σ(s)) + (1-y)·log(1-σ(s))]
When active: Always, every batch
Analogy: "Did you answer true/false correctly?"
```

**`training/novel_loss.py` — RankingAwareLoss (V6 ListNet)**
```
Purpose: Make the correct answer rank ABOVE wrong answers
Formula: -log(softmax([s_pos, s_neg1,...,s_negK])[0])
When active: V6 only, weight=0.3
Analogy: "Did you rank it highest among all contestants?"
```

---

#### Group 2: Interference Quality Losses

**`training/interference_loss.py` — PhaseSeparationLoss (V2)**
```
Purpose: Keep positive and negative answer phases 180° apart
Formula: mean(cos(φ_pos - φ_neg))
When active: V2+, weight=0.5
Analogy: "Are your 'true' waves and 'false' waves pointing opposite directions?"
```

**`training/interference_loss.py` — ContrastiveInterferenceLoss (V2)**
```
Purpose: Directly penalize wrong answers that have constructive interference
Formula: max(0, P_wrong - P_correct + margin)
When active: V2+
Analogy: "The wrong answer should not be interfering constructively."
```

**`training/interference_loss.py` — InterferenceRegularization (V2)**
```
Purpose: Prevent Phase Collapse (imaginary parts → 0)
Formula: -mean(||Im(states)||² + ||phases||²)
When active: V2+
Analogy: "Keep your phases alive and spread out — don't collapse to real numbers."
```

**`training/v5_loss.py` — InterferencePolarityLoss (V5)**
```
Purpose: Enforce DESTRUCTIVE interference on wrong paths (not just phase separation)
Formula: max(0, Int(h,t_wrong) + margin)
When active: V5+
Analogy: "The interference cross-terms for wrong answers must be NEGATIVE."
Note: Theorem V5.2 proves Phase Collapse is not a valid solution to this loss.
```

---

#### Group 3: Logical Consistency Losses (V4 only)

**`training/v4_loss.py` — LogicEntityLoss (L_ELoss)**
```
Purpose: Entity embeddings should be valid logical propositions (binary-like)
Formula: ||E ⊙ (1-E)||² (penalize middle values, push to 0 or 1)
When active: V4 QuaternionReasoner only
```

**`training/v4_loss.py` — LogicRelationLoss (L_LLoss)**
```
Purpose: True and false triples should have orthogonal logical embeddings
Formula: (E_true · E_false)² (penalize non-zero dot products)
When active: V4 only
```

**`training/v4_loss.py` — LogicMembershipLoss (L_MLoss)**
```
Purpose: Entities should fit their relation's domain/range types
Formula: ||E_h - domain(r)||² + ||E_t - range(r)||²
When active: V4 only
```

---

#### Group 4: V6 Novel Losses

**`training/novel_loss.py` — TeleportationContrastiveLoss (V6)**
```
Purpose: Teleportation scores for correct triples should exceed wrong ones
Formula: max(0, tel_score_wrong - tel_score_correct + margin) + entropy_penalty
When active: V6 use_teleportation=True, weight=0.5
Analogy: "The teleportation channel should deliver correct answers more reliably."
```

**`training/novel_loss.py` — EntanglementEntropyRegularizer (V6)**
```
Purpose: Keep entanglement entropy at a physically meaningful level
Formula: ||S(ρ) - target_entropy||²
When active: V6, weight=0.01
Analogy: "The quantum channel should be genuinely entangled — not too much, not too little."
```

**`training/novel_loss.py` — ContextualityLoss (V6)**
```
Purpose: Force the model to use genuinely quantum (non-classical) correlations
Formula: ReLU(margin - |score_joint - score_factorized|)
When active: V6 use_contextuality_loss=True, weight=0.1
Details: See Q18
```

**`training/novel_loss.py` — DecoherenceLoss (V6)**
```
Purpose: Shape decoherence rates based on path diversity and coherence
Formula: correlation between path diversity and decoherence rate
When active: V6 use_decoherence=True, annealed from ε=0.3 → 0.01
```

---

## 9. What is a Quaternion, and How Does it Differ From Quantum?

### What is a Complex Number? (Foundation)

Before quaternions, understand complex numbers:

A complex number has TWO parts: `z = a + bi`
- `a` is the "real" part (ordinary number)
- `b` is the "imaginary" part (multiplied by `i`, where `i² = -1`)

Think of it as a point on a 2D grid, or a 2D vector with a direction AND a magnitude.
Complex numbers are excellent for representing 2D rotations.

---

### What is a Quaternion?

A quaternion extends this to **4 parts**:
```
q = a + bi + cj + dk
```
- `a` = real part
- `b, c, d` = three imaginary parts (each with different imaginary unit: i, j, k)
- These follow special rules: `i² = j² = k² = -1`, `ij = k`, `ji = -k`

The last rule (`ij ≠ ji`) is the **non-commutative** property. Multiplication order matters.

**Quaternions represent 3D rotations** — you can encode any rotation in 3D space as a unit
quaternion (where `a² + b² + c² + d² = 1`). This is why game engines and computer graphics
use quaternions to rotate objects in 3D space.

---

### Quaternion Reasoning in V4

In the V4 model:
- Each entity is stored as a unit quaternion `q = a + bi + cj + dk`
- Each relation is also a unit quaternion
- Reasoning is done via **Hamilton product** (quaternion multiplication): `q_head ⊗ q_relation`
- Scoring: does the rotated head match the tail? `score = q_head ⊗ q_relation · q_tail`

**Benefits:**
- Captures 3D rotation structure in relationship space
- Non-commutativity naturally handles asymmetric relations
- Can model hierarchy, symmetry, anti-symmetry patterns

---

### How is Quaternion DIFFERENT from Quantum?

This is the most important distinction:

| Feature | Quaternion (V4) | Quantum (V1-V6) |
|---------|----------------|-----------------|
| **Math space** | 4D quaternion ℍ | Complex Hilbert space ℂᵈ |
| **Dimension** | Fixed 4D | Configurable d (e.g., 128) |
| **Interference** | No Born rule | Yes — |sum of amplitudes|² |
| **Scoring** | Dot product | Squared inner product |
| **Cross-terms** | No | Yes — quantum superposition |
| **Inspiration** | Hypercomplex algebra | Quantum mechanics |
| **Hardware** | No quantum computer needed | Can run on quantum hardware |

**The fundamental difference:**
- Quaternion model: `score = ⟨h⊗r, t⟩` — simple dot product after rotation
- Quantum model: `score = |⟨t|U_r|h⟩|²` — SQUARED magnitude of inner product

The **squaring** in quantum produces cross-terms. The dot product in quaternions does not.
This is why quaternions are a "classical" math approach dressed in 4D clothing, while
the quantum model genuinely exploits interference.

---

### Summary

- **Quaternions** = a 4D math tool from 1843 (before quantum mechanics!) used for 3D rotations
- **Quantum reasoning** = inspired by actual quantum mechanics, uses complex Hilbert space,
  produces genuine interference between paths via the Born rule
- **They are completely different** — quaternions just happen to also use imaginary numbers

---

## 10. Kolmogorov-Smirnov Test (KS-Test)

### What is it?

The KS-test is a **statistical test** that answers the question:
"Does my data come from a particular distribution?"

In this project, it answers: **"Are the phase angles in the entity embeddings
uniformly spread out, or are they all clustered near zero (Phase Collapse)?"**

---

### The Analogy

Imagine you spin a wheel 1000 times and record where it stops (0° to 360°).

- If the wheel is **fair**: every angle should appear roughly equally → uniform distribution
- If the wheel is **broken** and always stops near 0°: all readings cluster → non-uniform

The KS-test compares your actual data's distribution against the expected distribution
(uniform) and measures: **how far apart are they?**

---

### How It Works

The test computes:
```
D = max gap between your empirical CDF and the theoretical CDF
```

**CDF** (Cumulative Distribution Function) = "What fraction of values are below x?"

For uniform distribution on (-π, π):
- CDF at 0 should be 0.5 (half the values below 0)
- CDF at π should be 1.0 (all values below π)

If Phase Collapse happens:
- All phases ≈ 0 → CDF jumps from 0 to 1 at x=0 (step function)
- Maximum gap D ≈ 0.5 (huge deviation from uniform)

**Rule:** If D > threshold (typically 1.36/√n where n = number of samples), the phases
are NOT uniform → Phase Collapse is happening → the model needs intervention.

---

### In the Code (`evaluation/phase_statistics.py`)

After every epoch, the code:
1. Extracts all entity phase angles: `φ = arctan(Im(h) / Re(h))`
2. Runs KS-test against Uniform(-π, π)
3. Reports `D` value and whether it's significant
4. If D > threshold → logs a warning that Phase Collapse may be occurring

---

## 11. noise_guarantee.py and interference_guarantee.py Explained

These files contain the **formal mathematical theorems** that back up the paper's claims.
They're the "proof" section of the paper, translated into runnable code.

---

### noise_guarantee.py — Theorem 8.3

**The claim it proves:**
"Under random noise, the quantum model degrades more gracefully than classical models."

**What is "quantum gap"?**

The quantum model scores the correct answer higher than wrong answers by some margin.
That margin is the "gap." Theorem 8.3 says:

```
Quantum gap under noise ∝ (1-p)² × sin²(φ/2)
Classical gap under noise → 0 when the classical model can't distinguish paths
```

Where:
- `p` = noise rate (fraction of random errors injected)
- `φ` = phase difference between correct and incorrect paths
- `(1-p)²` = surviving signal after noise destroys some paths
- `sin²(φ/2)` = how much the phase difference helps (maximum at φ=π)

**In plain English:**
Even with significant noise, as long as the phase separation is maintained (φ > 0), the
quantum model can still distinguish right from wrong. The classical model's gap goes to zero
faster because it has no phase-based mechanism to survive noise.

**What the code does:**
- `verify_theorem_83()`: Takes a trained model, injects various noise levels p ∈ {0, 0.1, 0.2, ...}
- Measures actual gap at each noise level
- Compares to the formula's prediction
- Reports `R²` score (how well the formula matches actual behavior)

---

### inteference_guarantee.py — Three Theorems

**Lemma V5.1 (Interference Polarity)**

"Whether interference is constructive or destructive depends only on the phase difference."

```
Cross-term Int_ij is DESTRUCTIVE (negative) ⟺ phase difference φ_ij ∈ (π/2, 3π/2)
```

In plain English: If two paths have phases more than 90° apart (but less than 270°),
they will destructively interfere. The code verifies this by:
1. Computing all pairwise phase differences
2. Checking the sign of each cross-term
3. Confirming the sign matches the formula prediction

---

**Theorem V5.2 (Gradient Lemma)**

"Phase Collapse cannot satisfy InterferencePolarityLoss."

This is a **proof that the training is sound**. It proves that:
- When Phase Collapse occurs (all phases = 0), the gradient of InterferencePolarityLoss
  is always **non-zero and pointing away from zero**
- This means the optimizer will always push phases away from 0
- Phase Collapse is NOT a stable fixed point of the training process with this loss

The code verifies this by:
1. Setting all phases to 0 (simulating Phase Collapse)
2. Computing gradients of the loss
3. Confirming the gradient magnitude is above a threshold (i.e., non-zero)

---

**Theorem V5.3 (RotatE Separation)**

"MatrixExpUnitary (V5) is strictly more expressive than RotatE."

RotatE is a popular prior model that also uses complex numbers. This theorem proves
the V5 model is **not the same thing** as RotatE and is strictly more capable.

The proof:
- RotatE uses diagonal unitary operators (only d parameters per relation)
- MatrixExpUnitary uses full Hermitian generators (d² parameters per relation)
- d² > d for d > 1 (always strictly more parameters)
- Off-diagonal elements in the Hermitian generator enable **dimension-mixing** that
  diagonal matrices cannot do

The code demonstrates this by:
1. Constructing a matrix that MatrixExpUnitary can represent but RotatE cannot
2. Verifying the matrix has off-diagonal entries (non-diagonal = non-RotatE)
3. Confirming the parameter count difference

---

## 12. V5 Polarity Gradients and Unitary Spanning Proofs

### V5 Polarity Gradients

**The problem with Phase Separation Loss (V2):**

PhaseSeparationLoss just said "push positive and negative phases apart." But it didn't
specify which direction — positive phases could drift to +π, negative to 0, or vice versa.
Both directions could be valid.

**V5 InterferencePolarityLoss goes further:**

It directly computes the **cross-term** (the interference contribution) for wrong answers
and says: "The cross-term for wrong answers must be NEGATIVE."

```python
Int(h, t_wrong) = 2 × Re(A_path1 × conj(A_path2))  # cross-term value
L_polarity = max(0, Int(h, t_wrong) + margin)        # penalty if positive
```

**The gradient analysis:**
- When `Int > -margin` (wrong answer has constructive or weak destructive interference):
  Gradient is non-zero → pushes phases toward destructive configuration
- When `Int < -margin` (good destructive interference):
  Gradient is zero → no unnecessary perturbation

**Why is this better?**

PhaseSeparationLoss could be "solved" by Phase Collapse (if imaginary parts → 0, the phase
difference is undefined, and the loss might accidentally go to 0). Theorem V5.2 proves that
InterferencePolarityLoss **cannot** be satisfied by Phase Collapse — the gradient at collapse
is always pointing away.

---

### Unitary Spanning Proofs (Theorem V5.3)

**"Spanning" refers to parameter space coverage:**

- **DiagonalUnitary** spans `T^d` — the set of all diagonal unitary matrices
  - Dimension: d (one phase per dimension)
  - Geometrically: a torus (d independent circles)

- **MatrixExpUnitary** spans `U(d)` — the set of all d×d unitary matrices
  - Dimension: d² (full Hermitian matrix)
  - Geometrically: all possible "quantum rotations" in d-dimensional space

The proof `T^d ⊊ U(d)` means: every diagonal unitary IS a matrix exp unitary (by setting
off-diagonal elements to 0), but NOT every matrix exp unitary is diagonal. The gap is
`d² - d` extra dimensions of expressiveness.

**Why does this matter for the paper?**

It formally shows that V5's model cannot be reduced to RotatE or DiagonalUnitary V1.
The improvements from V1 to V5 are **provably** more than just engineering tweaks —
they expand the fundamental expressiveness of the model.

---

## 13. Real Units and Imaginary Units of Complex Unit Vectors

### What is a Unit Vector?

A **unit vector** has length exactly 1. In regular 2D space: `(x, y)` is unit if `x² + y² = 1`.

A **complex unit vector** of dimension d is a vector of d complex numbers where the
total length (norm) is 1:
```
h = (h₁, h₂, ..., h_d) where each hⱼ = a_j + i·b_j (complex)
‖h‖ = √(|h₁|² + |h₂|² + ... + |h_d|²) = 1
```

---

### Real Parts

For a complex number `hⱼ = a + bi`:
- The **real part** `a = Re(hⱼ)` represents the component along the "horizontal" axis of the
  complex plane for dimension j
- In the entity embedding: `Re(h)` = the classical, straightforward "what this entity means"
  component
- Real parts alone would give you a standard vector embedding (like Word2Vec or TransE)

---

### Imaginary Parts

- The **imaginary part** `b = Im(hⱼ)` represents the component along the "vertical" axis
  (the imaginary axis) for dimension j
- In the entity embedding: `Im(h)` = the "phase" component that enables interference
- Imaginary parts are what make quantum reasoning different from classical

**Why imaginary parts matter for interference:**

Consider the inner product of two states:
```
⟨t|h⟩ = Σⱼ t̄ⱼ · hⱼ = Σⱼ (a_j^t - i·b_j^t)(a_j^h + i·b_j^h)
       = Σⱼ [(a_j^t·a_j^h + b_j^t·b_j^h) + i(a_j^t·b_j^h - b_j^t·a_j^h)]
              ↑ real part of inner product     ↑ imaginary part
```

The **imaginary part of the inner product** is what creates the complex phase `φ` when you take
`|⟨t|h⟩|² = Re(⟨t|h⟩)² + Im(⟨t|h⟩)²`. If `Im(h) = 0` everywhere, then `Im(⟨t|h⟩) = 0`
for all t, h — and you lose the ability to have different phases for different paths →
Phase Collapse.

---

### How They're Stored in Code

The entity encoder stores them **separately** (not as a single complex tensor):
```python
self.entity_real = nn.Embedding(num_entities, complex_dim)  # Re(h)
self.entity_imag = nn.Embedding(num_entities, complex_dim)  # Im(h)
```

This is deliberate — it allows different learning rates for real vs imaginary parts:
```
lr_real = base_lr          (e.g., 0.001)
lr_imag = 3 × base_lr     (e.g., 0.003)
```

The 3× multiplier compensates for the fact that imaginary parts contribute to interference
terms which are "second-order" effects and have smaller gradients early in training.

---

## 14. V1 Diagonal Matrices, V4 Quaternion Logic, V5 Matrix Exponential, U(d)

### V1: DiagonalUnitary (Simple Phase Rotations)

**What it does:**

Each relation `r` gets d independent phase values `θ = (θ₁, θ₂, ..., θ_d)`.
The unitary operator is a diagonal matrix:

```
U_r = diag(e^(iθ₁), e^(iθ₂), ..., e^(iθ_d))
```

When applied to state h:
```
U_r · h = (e^(iθ₁)·h₁, e^(iθ₂)·h₂, ..., e^(iθ_d)·h_d)
```

It **independently rotates each dimension** by its own angle. Dimensions never interact with
each other.

**Analogy:** Each light beam gets its own color filter that shifts its hue. All beams are
treated independently — there's no mixing.

**Parameters:** d per relation (one angle per dimension)
**Hardware:** Maps directly to RZ quantum gates — one RZ gate per dimension

---

### V4: Quaternion Logic

Rather than phase rotation, V4 uses the **Hamilton product** of quaternions.
Each entity is a 4D quaternion; each relation is also a 4D unit quaternion.

The "logic" part refers to the `QuantumLogicLattice` (see Q2) that adds orthogonality
constraints. V4 combines:
- 4D quaternion rotation for relation application
- Lattice-based logical consistency checking

**Analogy:** Instead of independent color filters (V1), you use a gyroscope (3D rotation).
A gyroscope rotation of an object depends on the full 3D orientation, not just independent
axes.

**Parameters:** 4 per entity (quaternion), 3 effective DOF per relation (unit quaternion)

---

### V5: MatrixExpUnitary — Full U(d)

**The key upgrade:** Instead of diagonal phase rotations, V5 uses a **full Hermitian matrix**
`H_r` as the generator, and computes `U_r = exp(i · H_r)`.

**What is a Hermitian matrix?**
A matrix where `H = H†` (equal to its own conjugate transpose). Hermitian matrices have
real eigenvalues and are the "generators" of unitary matrices.

**What does exp(iH) mean?**
The **matrix exponential** of iH. Like `exp(x)` for numbers, but for matrices:
```
exp(iH) = I + iH + (iH)²/2! + (iH)³/3! + ...
```

This is computed numerically using the Cayley approximation or eigendecomposition.

**Why is this much more expressive?**

- Diagonal V1: only d parameters per relation → can only independently rotate dimensions
- MatrixExp V5: d² parameters per relation → can mix any dimension with any other dimension

**Example with d=2:**

DiagonalUnitary (V1):
```
U = [[e^(iθ₁),    0    ],
     [   0,    e^(iθ₂) ]]
```

MatrixExpUnitary (V5) — example with off-diagonal coupling:
```
H = [[0.5, 0.3+0.2i],
     [0.3-0.2i, 0.5]]

U = exp(iH) = [[... , ...],   (off-diagonal entries appear!)
               [...,  ...]]
```

The off-diagonal entries mean dimension 1 can "talk to" dimension 2 — they're coupled.
This is what enables **non-commutative composition**:
```
U_{r2} @ U_{r1}  ≠  U_{r1} @ U_{r2}   (path ordering matters!)
```

**What is U(d)?**

`U(d)` is the group of all d×d unitary matrices. "Group" means: any two U(d) matrices
multiplied together give another U(d) matrix; every U(d) matrix has an inverse in U(d).

MatrixExpUnitary spans **all of U(d)** — any unitary relation transformation is achievable.
DiagonalUnitary only spans a tiny corner of U(d): the diagonal ones (called the maximal torus T^d).

---

## 15. 1-Hop Born Rule Proxy, Multi-Hop Decoherence Branch, Teleportation Branch

These are the three **scoring branches** in V6's `QuantumReasoner`:

---

### Branch 1: Fast 1-Hop Born Rule Proxy

**What it does:** Directly computes `|⟨t|U_r|h⟩|²` without using any paths.

```
score_triple(h, r, t) = |⟨t | U_r | h⟩|²
```

**Why it's "fast":**
- No path search needed
- Single matrix-vector multiplication
- O(d²) computation

**When it's used:** During **training** — computing scores for billions of triples needs
to be fast. The 1-hop score is a good proxy for training signal.

**The limitation:** It doesn't capture multi-hop reasoning. "Einstein → born_in → Germany
→ located_in → Europe" cannot be captured by a single U_r.

---

### Branch 2: Multi-Hop Decoherence Branch

**What it does:** Finds paths of length 2, 3, ... between head and tail through the graph,
applies each relation's unitary in sequence, and combines them with noise (decoherence).

```
State after path P = (r₁, r₂, ..., rₖ):
  |ψ_P⟩ = U_rₖ · ... · U_r₂ · U_r₁ · |h⟩

With decoherence (noise) after each step:
  ρ_P = (1-ε)^k · |ψ_P⟩⟨ψ_P| + (1-(1-ε)^k) · I/d
  ↑ pure quantum state                  ↑ random noise

Score contribution from path P:
  ⟨t|ρ_P|t⟩ = (1-ε)^k · |⟨t|ψ_P⟩|² + (1-(1-ε)^k)/d
```

**Why decoherence?**
Real quantum systems lose coherence (become "noisy") over time. Long paths (many hops)
have more accumulated noise. The decoherence parameter `ε` models this — each hop
introduces `ε` probability of the quantum state being replaced by random noise.

**What it enables:**
- 2-hop, 3-hop reasoning
- Modeling confidence degradation over long inference chains
- Training with `interference_train_fraction=0.3` (30% of batches use this branch)

---

### Branch 3: Teleportation Branch

**What it does:** Computes scores using the Bell State relation matrices and Weyl corrections
(see Q3).

```
tel_score(h, r, t) = |Σₖ √qₖ · ⟨t | Cₖ† · M_r | h⟩|²
                           ↑            ↑      ↑
                     learned       Weyl     unconstrained
                     weights     correction  Bell matrix
```

**Why "teleportation"?**

In quantum teleportation, Alice wants to send a quantum state to Bob without physically
transmitting the quantum particle. She:
1. Creates an entangled Bell pair (shared between her and Bob)
2. Measures her particle with the state she wants to send
3. Classically tells Bob which of 4 outcomes she got (Bell measurement result)
4. Bob applies the corresponding correction (Pauli/Weyl operator) to get the original state

In the KG context:
- Alice = head entity h
- Bob = tail entity t  
- Bell pair = relation M_r (the "channel" between them)
- Bell measurement outcome = which Weyl correction k to apply
- DifferentiableBellMeasurement = the "measurement" that picks k (learned, not physical)

**What it enables:**
- Unconstrained relation matrices (richer than unitary-only V1-V5)
- Content-dependent correction selection (different correction for different queries)
- Entanglement-based reasoning (captures quantum correlations between head and tail)

---

### How the Three Branches Combine

During training (V6):
```
Total score = λ₁ × score_triple     (1-hop, fast)
            + λ₂ × score_decoherence (multi-hop, slower)
            + λ₃ × score_teleportation (teleportation branch)
```

During evaluation:
```
Final score = score_with_paths()  ← uses decoherence + teleportation
```

---

## 16. What is Quantum Teleportation (in this context)?

### Real Quantum Teleportation (Background)

In physics, quantum teleportation lets you **transfer a quantum state** from one place to
another using:
1. A shared entangled pair (one particle each for sender and receiver)
2. A classical message (2 bits saying which Bell state was measured)
3. A local correction (Pauli rotation based on the 2 bits)

**Key insight:** The original quantum state is "destroyed" at the sender and "reconstructed"
at the receiver. The state was never physically moved — only classical information traveled.

The **4 possible correction operations** in 2D quantum computing are the Pauli matrices (I, X, Y, Z).
In d-dimensional systems, there are d² Weyl operators.

---

### Quantum Teleportation as Knowledge Graph Scoring

The V6 model borrows this idea as a **mathematical framework** for scoring:

1. **"Alice"** = head entity `h` — the entity we're reasoning from
2. **"Shared entangled pair"** = `BellStateRelation M_r` — encodes the relationship channel
3. **"Bell measurement"** = `DifferentiableBellMeasurement` — a learned MLP that takes
   the head embedding and relation embedding and outputs probabilities `qₖ` over which
   Weyl correction to use
4. **"Correction"** = `GeneralizedPauliCorrections Cₖ†` — the Weyl operator applied to the
   "transmitted" state
5. **"Bob"** = tail entity `t` — checks if the corrected state matches

**The scoring formula:**
```
score = |Σₖ √qₖ · ⟨t | Cₖ† · M_r | h⟩|²
```

- Multiple corrections k are tried, each with learned probability `qₖ`
- The amplitudes for each correction are summed before squaring → interference between branches!
- `qₖ` depends on the content of h and r → the model learns "which correction channel works
  best for which type of relation/entity"

**Why is this better than direct scoring?**

1. **Unconstrained M_r**: The Bell state matrix can be any complex matrix (not just unitary)
   → richer relation representations
2. **Content-aware corrections**: Different queries use different Weyl corrections
   → adaptive reasoning
3. **Interference between corrections**: The sum before squaring creates cross-terms between
   different correction branches → extra expressive power

---

## 17. Decoherence: Pure State, Per-Relation Channel, Quantum Trace

### What is Quantum Decoherence?

In quantum mechanics, a "pure state" is a perfectly isolated quantum particle — like a single
photon in a vacuum with no disturbances. In reality, every quantum system interacts with its
environment (air molecules, electromagnetic fields, vibrations), and these interactions
gradually **destroy the quantum properties**.

This destruction of quantum properties is called **decoherence**.

**Analogy:** Imagine a perfectly clear glass of water (pure state). If you add a drop of ink,
it spreads and makes the water murky. The water can never be perfectly clear again without
active purification. Decoherence is like that ink drop — once quantum information mixes with
the environment, it's hard to get back.

---

### Pure State (ρ = |ψ⟩⟨ψ|)

A **pure state** is the most "quantum" state possible — maximum coherence, no noise.

In the code:
```python
|ψ⟩ = U_r @ h_state   # apply relation to head entity
ρ_pure = outer_product(|ψ⟩, conj(|ψ⟩))  # = |ψ⟩⟨ψ|
```

This is a d×d matrix (called a **density matrix**). For a pure state, this matrix has
special properties:
- Trace = 1 (Tr(ρ) = 1)
- Idempotent: ρ² = ρ
- Only one non-zero eigenvalue (= 1)

---

### Per-Relation Decoherence Channel

Each relation gets its own **noise rate** `ε_r`:

```python
self.log_decoherence_rate = nn.Parameter(torch.zeros(num_relations))  # learnable!
ε_r = sigmoid(log_decoherence_rate[r])  # ε_r ∈ (0, 1)
```

After k hops through relations with noise rate ε:
```
ρ_noisy = (1-ε)^k × ρ_pure + (1-(1-ε)^k) × I/d
           ↑ quantum part        ↑ noise part (uniform random)
```

- `(1-ε)^k` = probability of surviving k noisy hops intact
- `(1-(1-ε)^k)` = probability of being completely randomized
- `I/d` = completely random state (identity matrix divided by d = uniform distribution)

**Why make ε learnable?**

Some relations are "reliable" (the inference is always clear — e.g., "is_a" in a taxonomy).
Others are "noisy" (uncertain, ambiguous — e.g., "related_to" in a broad sense).
The model learns which relations are reliable (ε_r → 0) vs noisy (ε_r → 0.5).

---

### Decoherence Rate Annealing

The noise rate is also **annealed** during training:

```
ε(epoch) = ε_final + (ε_init - ε_final) × exp(-epoch/τ)
         = 0.01 + 0.29 × exp(-epoch/33)
```

- **Epoch 0:** ε = 0.30 (lots of noise — acts like regularization, prevents overfitting)
- **Epoch 33:** ε ≈ 0.12 (moderate noise)
- **Epoch 100:** ε ≈ 0.02 (almost no noise — model trains in clean quantum regime)

**Why start with high noise?** Like simulated annealing in optimization, high noise early
prevents getting stuck in bad local minima. Then gradually "cool" to let the model settle
into a precise, low-noise solution.

---

### Quantum Trace Operation

The **trace** of a matrix is the sum of its diagonal elements:
```
Tr(M) = M[0,0] + M[1,1] + M[2,2] + ... + M[d,d]
```

For scoring with a density matrix:
```
score = Tr(ρ × |t⟩⟨t|) = ⟨t|ρ|t⟩ = ρ[t, t] after projecting to the t-th dimension
```

**In plain English:** Take the density matrix ρ (which represents the uncertain quantum state
of the system after path traversal), then look at the diagonal entry corresponding to entity t.
That entry tells you: "What is the probability that a quantum measurement would give result t?"

This is the **Born rule extended to mixed states** (noisy quantum states). For a pure state,
this reduces to the simple `|⟨t|ψ⟩|²` we use in V1-V5. For a mixed state (decoherent),
it's `(1-ε)^k × |⟨t|ψ⟩|² + (1-(1-ε)^k)/d`.

**The physical meaning:** A score of 1.0 means "if we 'measured' the quantum state after
traversing the relation path, we would always observe entity t." A score of 1/d means "it's
completely random — all entities equally likely."

---

## 18. Contextuality Loss

### What is "Contextuality" in Quantum Mechanics?

In classical physics, a measurement's outcome depends only on the thing being measured,
not on what else you happen to measure at the same time. This is called **non-contextuality**.

Quantum mechanics violates this. Whether a spin measures "up" or "down" can depend on what
other measurements you make simultaneously. This is called **quantum contextuality** and is
one of the deepest non-classical features of quantum mechanics.

**Simplified analogy:** Imagine asking someone "are you happy?" The answer might depend on
what other questions you ask alongside it:
- If paired with "are you healthy?": "Yes, very happy"
- If paired with "are you tired?": "Well, not really..."

A classical, context-independent person would give the same answer no matter what else
you ask. A "contextual" person's answer changes based on the measurement context.

---

### What is a "Factorizable" Score?

A scoring function is **factorizable** (classical, non-contextual) if it can be written as:
```
score(head, relation, tail) = f(head) × g(relation) × h(tail)
```

The head, relation, and tail contribute independently — no cross-dimensional interaction.

Models like TransE are almost factorizable: `score = -‖h + r - t‖` — the contributions
add independently. The real and imaginary parts of entities in a factorized model would
contribute independently:
```
score_factorized = |⟨t_re|U_r^re|h_re⟩|² × |⟨t_im|U_r^im|h_im⟩|²
                   (real part score)              (imaginary part score, independent)
```

---

### What is ContextualityLoss?

The loss **penalizes the model whenever its scores are factorizable**:

```python
score_joint      = |⟨t|U_r|h⟩|²                    # full quantum score (real+imag interact)
score_factorized = |⟨t_re|U_r^re|h_re⟩|² × |⟨t_im|U_r^im|h_im⟩|²  # separated scores

gap = |score_joint - score_factorized|

L_contextuality = ReLU(margin - gap)
                = max(0, margin - gap)
```

- If `gap < margin` (scores are too similar = nearly factorizable): loss is positive → penalty
- If `gap ≥ margin` (scores are genuinely different = contextual): loss is zero → no penalty

**The gradient:**
- Pushes the joint score away from the factorized score
- Forces the model to develop **cross-dimensional correlations** between real and imaginary parts
- These correlations are exactly what makes quantum models non-classical

---

### Why Does This Matter?

If the model's score were always factorizable, then:
1. The real and imaginary parts contribute independently → no genuine quantum behavior
2. The model degenerates to two separate real-valued models running in parallel
3. The interference cross-terms between real and imaginary parts would be zero
4. This is essentially Phase Collapse in a different form

By enforcing contextuality (gap ≥ margin), the model is forced to:
1. Use the **interplay** between real and imaginary parts
2. Maintain non-trivial correlations between head, relation, and tail simultaneously
3. Exploit quantum-like behavior that no classical factorized model can reproduce

**The connection to Bell inequalities:**

In quantum physics, Bell's theorem proved that quantum mechanics violates certain inequalities
that any classical (factorizable) system must obey. Our ContextualityLoss is the knowledge
graph analogue: we enforce that the scoring function violates the "classical factorizability
inequality" by at least `margin`.

When the loss is near zero at convergence (which it is for a well-trained V6 model),
it means: **the model has learned to use genuinely quantum-like (non-factorizable) reasoning.**

---

## Quick Reference Summary

| Concept | One-Line Explanation |
|---------|---------------------|
| Cross-terms | Bonus/penalty terms from squaring a sum — enable interference |
| Self-terms | Per-path squared amplitude — this is all classical models compute |
| Non-commutative | Order of operations matters (AB ≠ BA) |
| Logic lattice | Enforces logical contradictions to have orthogonal (zero-overlap) embeddings |
| Multi-view aggregator | Combines self-terms + cross-terms with logic weights |
| Bell state matrix | Unconstrained complex matrix per relation — richer than unitary |
| Weyl correction | One of d² complete "error correction" operators in d-dimensional space |
| quantum_reasoner | Complex Hilbert space model using Born rule |ψ|² scoring |
| quaternion_reasoner | 4D hypercomplex model using Hamilton product, NO interference |
| Phase separation | Force positive/negative phase angles 180° apart for max interference |
| Phase collapse | Disaster where imaginary parts → 0, killing all interference |
| ListNet | Ranking loss that always improves positive answer's rank (no dead zones) |
| MRR | Average reciprocal rank of the correct answer; 0.3 ≈ rank 3 on average |
| Hits@K | What fraction of queries had correct answer in top K? |
| Filtered MRR | Remove other true triples from ranking before scoring |
| Quaternion | 4D hypercomplex number for 3D rotations (NOT quantum mechanics) |
| KS-test | Statistical test: are phase angles uniformly spread? (collapse check) |
| Theorem 8.3 | Quantum gap degrades as (1-p)²sin²(φ/2); classical gap hits zero faster |
| T^d ⊊ U(d) | Diagonal unitaries are a tiny subset of all unitary matrices |
| Polarity gradient | V5 loss ensures cross-terms for wrong answers are negative (not just smaller) |
| Real part | Amplitude component of complex embedding — classical information |
| Imaginary part | Phase component — enables interference (3× LR to prevent collapse) |
| DiagonalUnitary | V1: d independent phase rotations, one per dimension |
| MatrixExpUnitary | V5: full d² Hermitian generator — any rotation in U(d), dimension-mixing |
| U(d) | All d×d unitary matrices — V5 spans ALL of this space |
| 1-hop proxy | Fast training score: single relation application, O(d²) cost |
| Decoherence branch | Multi-hop with per-hop noise; long paths become less certain |
| Teleportation branch | Bell matrix + Weyl corrections + learned measurement weights |
| Quantum teleportation | Transfer quantum state via classical message + shared entangled pair |
| Pure state |ψ⟩⟨ψ| | Maximum coherence density matrix; no noise, fully quantum |
| Per-relation ε | Each relation has its own learnable noise rate |
| Quantum trace Tr(ρ|t⟩⟨t|) | Born rule for noisy states: probability of measuring entity t |
| Decoherence annealing | Start with ε=0.3 (noisy, exploratory) → end at ε=0.01 (clean) |
| Contextuality | Score depends on real/imag interaction — cannot be factored apart |
| Factorizable score | Real and imaginary contribute independently — classical, kills interference |
| ContextualityLoss | Penalize factorizability; force model to use quantum-like correlations |

---

*Document written 2026-05-07. Author: research assistant for quantum KG project.*
*Questions? Reference the corresponding code files and README(Architecture).MD PART 9.*
