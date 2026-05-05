# CCluster

```bash
CMiner <db_file> -c <num_clusters | auto> [options]
```

| Flag | Default | Description |
| --- | --- | --- |
| `-c`, `--num_clusters` | — | Number of clusters or `auto` |
| `-d`, `--is_directed` | `0` | `1` for directed graphs |
| `-o`, `--output_path` | — | Save one file per cluster to this folder |
| `--auto_k_max` | `ceil(sqrt(n_graphs))` | Upper bound on K when using `auto` |
| `--verbose` | `0` | Progress logs (`1` = enabled) |

```bash
CMiner db.data -c 4
CMiner db.data -c auto
CMiner db.data -c auto --auto_k_max 12
CMiner db.data -c 4 -o ./output
```

---

# Strategies

```
--<type>_strategy <name> [key=value ...]
```

## `--enrichment_strategy`

Enriches graphs with semantic content before embedding.

### `noop` *(default)*
Passes graphs through unchanged. No parameters.

---

### `semantic_label_clustering`
Adds synthetic super-nodes grouping semantically similar node/edge labels via SentenceTransformer + K-Means (automatic K via silhouette).

```bash
--enrichment_strategy semantic_label_clustering [key=value ...]
```

| Parameter | Default | Description |
| --- | --- | --- |
| `model_name` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `random_state` | `42` | Seed for K-Means |
| `max_k` | `ceil(sqrt(n_labels))` | Upper bound for the K grid search |
| `verbose` | inherits `--verbose` | Progress logs |

---

### `label_cluster_replacement`
Same semantic clustering pipeline as `semantic_label_clustering`, but instead of adding super-nodes/edges it **replaces** each node label and edge type with the name of its cluster (`semantic_cluster_<id>`). The graph structure stays identical to the original; only the labels change.

```bash
--enrichment_strategy label_cluster_replacement [key=value ...]
```

| Parameter | Default | Description |
| --- | --- | --- |
| `model_name` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `random_state` | `42` | Seed for K-Means |
| `max_k` | `ceil(sqrt(n_labels))` | Upper bound for the K grid search |
| `verbose` | inherits `--verbose` | Progress logs |

---

## `--embedding_strategy`

Produces a pairwise distance matrix from the (enriched) graphs.

### `simple_structural` *(default)*
5-feature vector per graph (n_nodes, n_edges, n_node_labels, n_edge_labels, avg_degree) → normalized → Euclidean distances. No parameters.

---

### `flexible_subgraph`
Enumerates connected subgraphs within a size range → TF-IDF matrix → cosine distances.

```bash
--embedding_strategy flexible_subgraph [key=value ...]
```

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `subgraph_method` | `nodes`, `edges` | `nodes` | Node-induced or edge-induced |
| `min_size` | `int >= 1` | `2` / `1` | Minimum subgraph size (inclusive) |
| `max_size` | `int >= min_size` | no cap | Maximum subgraph size — cap this to control runtime |

---

## `--clustering_strategy`

Clusters graphs from the distance matrix.

### `kmedoids` *(default)*
Partitioning Around Medoids on the precomputed distance matrix.

```bash
--clustering_strategy kmedoids [key=value ...]
```

| Parameter | Values | Default | Description |
| --- | --- | --- | --- |
| `init_method` | `random`, `kmeans++` | `random` | Medoid initialization |
| `max_iter` | `int >= 1` | `100` | Maximum PAM iterations |
| `tolerance` | `float` | `1e-4` | Convergence threshold |
| `verbose` | inherits `--verbose` | Per-iteration logs |

---

# Examples

```bash
# flexible_subgraph with graphlet-kernel-style cap
CMiner db.data -c 4 \
    --embedding_strategy flexible_subgraph subgraph_method=nodes min_size=3 max_size=5

# Full pipeline with super-node enrichment
CMiner db.data -c 4 \
    --enrichment_strategy semantic_label_clustering model_name=all-MiniLM-L6-v2 \
    --embedding_strategy flexible_subgraph subgraph_method=edges min_size=2 max_size=5 \
    --clustering_strategy kmedoids init_method=kmeans++ max_iter=200 \
    --verbose 1 -o ./out

# Full pipeline with label replacement enrichment
CMiner db.data -c 4 \
    --enrichment_strategy label_cluster_replacement model_name=all-MiniLM-L6-v2 \
    --embedding_strategy flexible_subgraph subgraph_method=edges min_size=2 max_size=5 \
    --clustering_strategy kmedoids init_method=kmeans++ max_iter=200 \
    --verbose 1 -o ./out
```



