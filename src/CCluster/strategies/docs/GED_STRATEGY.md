# Graph Edit Distance (GED) in CMiner

## Objective

The GED strategy computes a distance matrix between patterns/graphs using NetworkX Graph Edit Distance.

In this project, the strategy is implemented in:
- `src/CCluster/strategies/ged_distance_strategy.py`

The resulting matrix is then used by the existing medoid-based clustering flow in CCluster.

## Intuitive GED definition

GED measures the minimum cost required to transform a graph $G_1$ into a graph $G_2$ through edit operations:
- node insertion/deletion
- edge insertion/deletion
- attribute substitution (internally handled by NetworkX according to match/cost functions)

Formally:

$$
GED(G_1, G_2) = \min_{\pi \in \Pi(G_1, G_2)} \sum_{o \in \pi} c(o)
$$

Where:
- $\Pi(G_1, G_2)$ is the set of valid edit-operation sequences
- $c(o)$ is the cost of the elementary operation $o$

## How it is implemented in this project

For each graph pair $(i, j)$, with $i < j$:
1. `networkx.graph_edit_distance(...)` is called
2. the resulting distance is stored symmetrically in the matrix
3. if the estimation does not finish in time (`None`), a fallback upper bound is used
4. the distance is optionally normalized

### Node and edge matching

- Nodes: the `labels` attribute is compared as sorted tuples of strings.
  - two nodes match if the tuples are exactly equal
- Edges: the `type` attribute is compared as a string.
  - two edges match if the type is exactly equal

This makes GED sensitive to both structure and semantic labels.

## GED CLI parameters

When using `--strategy ged`, you can configure:

- `--ged_timeout`:
  - timeout in seconds per pair
  - if `<= 0`, timeout is disabled
- `--ged_normalize`:
  - `1` enables distance normalization
  - `0` keeps the raw distance
- `--ged_node_del_cost`: node deletion cost
- `--ged_node_ins_cost`: node insertion cost
- `--ged_edge_del_cost`: edge deletion cost
- `--ged_edge_ins_cost`: edge insertion cost

All costs must be non-negative.

## Fallback when GED times out

If `graph_edit_distance` returns `None`, the following upper bound is used:

$$
UB = |V_1| \cdot c_{nd} + |V_2| \cdot c_{ni} + |E_1| \cdot c_{ed} + |E_2| \cdot c_{ei}
$$

Where:
- $|V_1|, |V_2|$ are node counts
- $|E_1|, |E_2|$ are edge counts
- $c_{nd}, c_{ni}, c_{ed}, c_{ei}$ are node/edge delete/insert costs

This guarantees a numeric value in the matrix.

## Normalization

If `--ged_normalize 1`, distance is divided by the same upper bound:

$$
d_{norm}(G_1, G_2) = \frac{d(G_1, G_2)}{UB}
$$

With this choice:
- values typically lie in $[0, 1]$
- identical graphs produce distance $0$

If $UB = 0$ (edge case with empty graphs and zero costs), distance is set to $0$.

## Complexity and practical impact

GED is computationally expensive.
With $n$ graphs, the number of pairwise comparisons is:

$$
\frac{n(n-1)}{2}
$$

Each comparison can be costly, especially with:
- large graphs
- many distinct labels
- high timeout values

Practical suggestions:
- start with small timeouts (for example, 1-3 seconds)
- use GED on small/medium datasets
- keep `--verbose 1` enabled to monitor progress

## Usage examples

Clustering with normalized GED:

CMiner /path/to/db.data -c 4 --strategy ged --ged_timeout 3 --ged_normalize 1

Clustering with raw GED and custom costs:

CMiner /path/to/db.data -c 4 --strategy ged --ged_normalize 0 --ged_node_del_cost 1.0 --ged_node_ins_cost 1.0 --ged_edge_del_cost 0.5 --ged_edge_ins_cost 0.5

## Operational notes

- GED strategy is available in the CCluster strategy registry.
- The final matrix is always symmetric with a zero diagonal.
- The result is fully compatible with the current medoid-based clustering pipeline.
