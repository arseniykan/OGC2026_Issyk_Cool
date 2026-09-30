# Optimization Grand Challenge 2026 (OGC 2026)
## The Grand Shipyard Puzzle: Pack the Block, Beat the Clock


> A restatement of the official problem. The authoritative specification is the
> Organizing Committee's problem statement PDF.

This document contains a comprehensive, cleaned, and structured version of the **Optimization Grand Challenge 2026** problem statement. It covers terminology, constraints, mathematical formulations, objective functions, input/output specifications, and submission/evaluation details.

---

## 1. Problem Description

### 1.1 Overview
The **Optimization Grand Challenge (OGC) 2026** addresses a spatial block scheduling problem drawn from shipbuilding. In a shipyard, each structural block of a ship is built inside a fixed workspace called a **bay**. 
Blocks are three-dimensional objects with irregular, layered shapes, so placing them in a bay without overlap is a hard geometric problem. Furthermore, every block carries a release date, a processing time, and a due date. Because bay space is shared, a block may have to wait or stay longer than needed depending on the other blocks around it.

An algorithm must simultaneously decide for each block:
1. **Which bay** it goes to.
2. **Where and how** it is positioned/oriented.
3. **When** it enters ($\text{ENTRY}$) and leaves ($\text{EXIT}$) the bay.

---

### 1.2 Terminology & Notation

*   **Bay ($j \in M$):** A physically partitioned workspace in a shipyard where operations are performed. Each bay $j$ has a fixed size $W_j \times H_j$, where $M := \{1, 2, \dots, m\}$ is the set of bays.
*   **Block ($i \in N$):** A production unit representing a large structural component of a ship. Each block $i \in N$ consists of $K_i$ polygonal layers forming a 3D structure, where $N := \{1, 2, \dots, n\}$ is the set of blocks.
*   **Orientation ($o$):** A block can be rotated. The number of pre-calculated orientation options for block $i$ is denoted by $O_i$.
*   **Release Date ($R_i$):** The earliest time at which block $i$ can start processing (be placed in a bay).
*   **Due Date ($D_i$):** The desired completion time of block $i$.
*   **Processing Time ($P_i$):** The time required to process block $i$.
*   **Entry Time ($\text{ENTRY}_i$):** The time at which block $i$ is placed into a bay.
*   **Exit Time ($\text{EXIT}_i$):** The time at which block $i$ leaves the bay.
*   **Tardiness ($T_i$):** The delay beyond the due date, defined as:
    $$T_i = \max(0, \text{EXIT}_i - D_i)$$
*   **Workload ($L_i$):** The total workload required during the processing of block $i$.
*   **Preference Score ($S_{ij}$):** The preference score of block $i$ to be assigned to bay $j$. The scores for a single block always sum to $100$ across all bays.

---

### 1.3 Bay Scheduling & Operations
Each bay operates independently with its own crane. There are two primary crane operations:
*   **$\text{ENTRY}$:** Placing a block into a bay.
*   **$\text{EXIT}$:** Removing a block from a bay.

#### Key Execution Rules:
1.  **Operation Slack:** Typically, $D_i - R_i > P_i$, which implies there is some "slack" time to optimize the $\text{ENTRY}$ and $\text{EXIT}$ times.
2.  **Daily Crane Operations:** Crane operations are performed before the day's working time. Any number of operations can be performed on a single day.
3.  **Daily Operation Order:** To maintain physical consistency, **all $\text{EXIT}$ operations must be performed before any $\text{ENTRY}$ operations** on the same day. Within the same type of operation (e.g., multiple $\text{ENTRY}$s), the algorithm can determine their relative order.
4.  **Temporal Consistency:** For a block $i$ with processing time $P_i$, if $\text{ENTRY}_i = t$, then the earliest it can exit is $\text{EXIT}_i = t + P_i$. The block is physically present in the bay during the interval $[\text{ENTRY}_i, \text{EXIT}_i)$.

---

### 1.4 Block Geometry
Each block consists of $K_i$ layers.
*   **No Floating Layers:** Layers are ordered from lowest (layer $0$) to highest (layer $K_i - 1$). Any two adjacent layers are physically attached; there are no empty or floating layers.
*   **Non-Convexity:** The polygon defining each layer does not need to be convex.
*   **Reference Point:** The first vertex of the first layer (layer $0$, orientation $0$) is always defined as $[0.0, 0.0]$. This is the **reference point** of the block.
*   **Translation:** When a block is placed at $(x, y)$, its coordinates are translated relative to the reference point. Note that depending on the orientation, the reference point may not be at the bottom-left of the block's bounding box.
*   **Fractional Vertices:** Vertices can have fractional coordinates with up to 4 decimal places. Due to potential numerical instability during polygon intersection checks, it is **strongly recommended** to use the collision checking functions in `utils.py`.

---

## 2. Constraints

Decisions must respect three groups of constraints:

### 2.1 Assignment & Operation Constraints
*   **Assignment Constraint:** Every block must be assigned to exactly one bay. No block can remain unassigned.
*   **Operation Constraint:** Every block must have exactly one $\text{ENTRY}$ operation and one $\text{EXIT}$ operation. Once placed, a block cannot be moved or rotated.

### 2.2 Temporal Constraints
*   **Release Date Constraint:** A block cannot enter the bay before its release date:
    $$\text{ENTRY}_i \geq R_i$$
*   **Processing Time Constraint:** A block must remain in the bay for at least its required processing time:
    $$\text{EXIT}_i - \text{ENTRY}_i \geq P_i$$

### 2.3 Spatial Constraints
Let $P^{o, x, y}_{i, l}$ represent the closed polygon of layer $l$ of block $i$ at position $(x, y)$ with orientation $o$. Let $\text{int}(\cdot)$ represent the interior of the polygon. We define the collision function:
$$C(i_1, l_1, o_1, x_1, y_1, i_2, l_2, o_2, x_2, y_2) = \begin{cases} 
0 & \text{if } \text{int}(P^{o_1, x_1, y_1}_{i_1, l_1}) \cap \text{int}(P^{o_2, x_2, y_2}_{i_2, l_2}) = \emptyset \\ 
1 & \text{otherwise} 
\end{cases}$$
Let $N(t, j)$ be the set of blocks assigned to bay $j$ that are present on date $t$:
$$N(t, j) := \{i \in N : \text{ENTRY}_i \leq t < \text{EXIT}_i \text{ and } i \text{ is assigned to bay } j\}$$

1.  **Bay Containment Constraint:** Each block must be fully contained within the boundary of the assigned bay $j$ (i.e., within $[0, W_j] \times [0, H_j]$).
2.  **Layer Collision-Free Constraint:** For any date $t$, bay $j$, and any two distinct blocks $i_1, i_2 \in N(t, j)$, their layers must not overlap:
    $$C(i_1, l, o_1, x_1, y_1, i_2, l, o_2, x_2, y_2) = 0 \quad \forall l = 0, 1, \dots, \min(K_{i_1}, K_{i_2}) - 1$$
    *(Note: Blocks are allowed to touch/share boundaries, i.e., boundary intersections are permitted).*
3.  **Crane Operation Constraint (Vertical Clearance):** To place or remove blocks, the crane must be able to move them vertically without physical interference. An $\text{ENTRY}$ or $\text{EXIT}$ operation for block $i_1$ is feasible only if:
    $$C(i_1, l_1, o_1, x_1, y_1, i_2, l_2, o_2, x_2, y_2) = 0 \quad \forall l_1 \leq l_2$$
    where $i_2 \in N(t, j) \setminus \{i_1\}$, $l_1 \in \{0, \dots, K_{i_1}-1\}$, and $l_2 \in \{0, \dots, K_{i_2}-1\}$.
    *Plain English:* You cannot slide a block under a wider layer of an existing block, nor can you block the path of a block underneath. A block can only be placed/removed if its layers do not overlap with any equal or higher layer levels of other active blocks.

---

## 3. Objective Function

The goal is to minimize a weighted combination of three metrics:
$$\text{Minimize } Z = w_1 Z_1 + w_2 Z_2 + w_3 Z_3$$
where $w_1, w_2, w_3 \geq 0$ are the weight parameters provided in the problem instance.

### 3.1 Total Tardiness ($Z_1$)
The sum of tardiness across all blocks:
$$Z_1 := \sum_{i=1}^{n} T_i = \sum_{i=1}^{n} \max(0, \text{EXIT}_i - D_i)$$

### 3.2 Workload Imbalance ($Z_2$)
The maximum weighted workload difference between any pair of bays:
$$Z_2 := \max_{j_1, j_2 \in M : j_1 \neq j_2} \left| u_{j_1} \sum_{i \in N(j_1)} L_i - u_{j_2} \sum_{i \in N(j_2)} L_i \right|$$
where $N(j)$ is the set of blocks assigned to bay $j$, and $u_j$ is a size-based normalization weight:
$$u_j = \frac{\sum_{k \in M} W_k \times H_k}{m \cdot (W_j \times H_j)}$$
*(Note: Larger bays have smaller weight coefficients $u_j$ because they reduce congestion for the same workload).*

### 3.3 Total Preference Score Penalty ($Z_3$)
The penalty incurred by not placing blocks in their most preferred bays:
$$Z_3 := \sum_{j \in M} \sum_{i \in N(j)} (S^{\max}_i - S_{ij})$$
where $S^{\max}_i = \max_{j \in M} \{S_{ij}\}$. If every block is assigned to its highest-preference bay, $Z_3 = 0$.

---

## 4. Input & Output Formats

All indices (bays, blocks, orientations) are **0-based** in the JSON files and code, whereas the mathematical description in Section 1 uses 1-based indices.

### 4.1 Input JSON Schema
A problem instance JSON contains four main root keys:
*   `"name"`: String name of the problem.
*   `"bays"`: List of dicts, each with `"width"` and `"height"`.
*   `"blocks"`: List of dicts, each containing:
    *   `"release_time"`: Integer
    *   `"due_date"`: Integer
    *   `"processing_time"`: Integer
    *   `"workload"`: Integer
    *   `"bay_preferences"`: Array of integers summing to 100, where index `j` represents preference score $S_{ij}$ for bay `j`.
    *   `"shape"`: List of orientations, containing `"orientation"` (int ID) and `"layers"` (list of lists of vertex coordinates `[x,y]`).
*   `"weights"`: Dict with keys `"w1"`, `"w2"`, `"w3"` (integers).

```json
{
  "name": "training_problem_01",
  "bays": [
    { "width": 125, "height": 15 },
    { "width": 71, "height": 18 }
  ],
  "blocks": [
    {
      "release_time": 37,
      "due_date": 44,
      "processing_time": 7,
      "workload": 20,
      "bay_preferences": [30, 70],
      "shape": [
        {
          "orientation": 0,
          "layers": [
            [[0.0, 0.0], [0.8093, 5.4723], [1.2955, 11.0993]],
            [[-3.0413, 0.3435], [0.5671, -0.064], [4.8185, 3.4301]]
          ]
        }
      ]
    }
  ],
  "weights": {
    "w1": 26667,
    "w2": 10,
    "w3": 300
  }
}
```

### 4.2 Output Solution Schema
The solution must be returned as a Python dictionary containing a single key `"operations"`. 
The value is a dictionary where the keys are string representations of dates (e.g., `"0"`, `"28"`).
*   **Only include dates with operations.**
*   For each date, specify a list of operation objects.
*   **EXIT operations must be listed before ENTRY operations on the same day.**
*   Positions `"x"` and `"y"` must be integers.

```json
{
  "operations": {
    "0": [
      {
        "type": "ENTRY",
        "block_id": 48,
        "bay_id": 1,
        "x": 0,
        "y": 0,
        "orient_idx": 0
      }
    ],
    "28": [
      {
        "type": "EXIT",
        "block_id": 48,
        "bay_id": 1
      },
      {
        "type": "ENTRY",
        "block_id": 15,
        "bay_id": 0,
        "x": 106,
        "y": 0,
        "orient_idx": 0
      }
    ]
  }
}
```

---

## 5. Development & Baseline Algorithm

### 5.1 Python Environment (`ogc2026`)
The evaluation server executes submissions under Python 3.12. Key pre-installed scientific and optimization libraries include:
*   `shapely` (2.1.2) — essential for spatial checks.
*   `ortools` (9.15.6755), `gurobipy` (13.0.2), `xpress` (9.8.1) — optimization solvers.
*   `pandas` (2.2.3), `scipy` (1.15.2), `scikit-learn` (1.6.1), `numba` (0.61.0), `networkx` (3.4.2).
*   `torch` (2.11.0), `tensorflow` (2.21.0) — deep learning (CPU execution only; no GPU available on evaluation server).

### 5.2 Baseline Strategy
The provided baseline (`baseline_greedy.py`) implements a greedy approach:
1.  Sorts blocks using the **Earliest Due Date (EDD)** rule.
2.  Iteratively places blocks at candidate positions (specifically the bottom-right corners of the Axis-Aligned Bounding Box, AABB).
3.  Because it ignores crane interference during initial placement, it resolves spatial conflicts and crane violations by delaying the `ENTRY` time of conflicting blocks to a later date when the bay clears.

### 5.3 Testing & Verification
You can verify solution feasibility locally using `utils.py`:
```python
from utils import check_feasibility
result = check_feasibility(prob_info, solution)
print(result)  # Output: {"stage": "PASS", "objective": <value>} on success
```

---

## 6. Submission Guidelines & Evaluation Server

*   **Submission Channel:** Email ZIP file to `submission@optichallenge.com` from the registered email address.
*   **ZIP Constraints:** Size must be under 15 MB. `myalgorithm.py` must be at the root of the ZIP file. Do not modify `utils.py` (it will be overwritten during evaluation).
*   **Cooldown:** A 12-hour cooldown period is enforced between successful submissions.
*   **Evaluation Environment:**
    *   **Hardware:** AMD Ryzen Threadripper PRO 9955WX.
    *   **Limits:** 4 CPU cores (400% CPU usage max), 16 GB RAM, no external internet access.
    *   **Time Limit:** Varies by instance (from a few minutes to 30 minutes wall-clock time).
*   **Leaderboard Ranking:**
    *   Invalid, crashed, or timed-out solutions score $-1$.
    *   Feasible solutions score $R - n_b$, where $R$ is the total number of teams, and $n_b$ is the number of teams with a strictly better objective value.
