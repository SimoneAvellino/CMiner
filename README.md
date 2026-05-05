<!-- @format -->

# CMiner

CMiner is a command-line tool for **mining frequent patterns** and
**clustering graphs** in a graph database. The CLI exposes two mutually
exclusive modes:

```bash
CMiner <db_file> -s <support>      [mining_options]      # mining mode
CMiner <db_file> -c <num_clusters> [clustering_options]  # clustering mode
```

For details on each mode, see the dedicated documentation:

-   **Mining mode (`-s`)** → [`src/CMiner/README.md`](src/CMiner/README.md)
-   **Clustering mode (`-c`)** → [`src/CCluster/README.md`](src/CCluster/README.md)

## Installation

#### Prerequisites

-   **Python**: 3.11.6
-   **pip**: 24.2

#### Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/SimoneAvellino/CMiner
    ```

2. Move into the repository folder:

    ```bash
    cd CMiner
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Install the library in `editable` mode:

    ```bash
    pip install -e .
    ```

## Quick start

Mine all patterns from 2 to 3 nodes, present in at least 50% of graphs:

```bash
CMiner /path/to/db.data -s 0.5 -l 2 -u 3
```

Cluster a graph database into 4 clusters:

```bash
CMiner /path/to/db.data -c 4
```

Cluster with automatic K selection (Silhouette method) and verbose logs:

```bash
CMiner /path/to/db.data -c auto --verbose 1
```

Cluster with a richer pipeline (semantic enrichment + flexible-subgraph
embedding, capped at size 5, with kmeans++ init):

```bash
CMiner /path/to/db.data -c 4 \
    --enrichment_strategy semantic_label_clustering \
    --embedding_strategy flexible_subgraph min_size=3 max_size=5 \
    --clustering_strategy kmedoids init_method=kmeans++ \
    --verbose 1
```

For all options, examples and the strategy-driven clustering pipeline
(enrichment → embedding → clustering, with per-step progress logs and
auto-K tuning), see the per-mode READMEs linked above.

## Required argument (both modes)

-   `db_file`: Absolute path to the graph database file.

The two flags `-s` / `--support` (mining) and `-c` / `--num_clusters`
(clustering) are mutually exclusive: exactly one mode must be selected.
