<!-- @format -->

# CMiner

CMiner is an algorithm for mining patterns from graphs using a user-defined support technique. This implementation provides a command-line interface for both pattern mining and graph clustering.

## Installation

#### Prerequisites

Make sure you have the following requirements to run the project:

- **Python**: Version 3.11.6
- **pip**: Version  24.2

#### Installation steps

1. Clone the repository:

    ```bash
    git clone https://github.com/SimoneAvellino/CMiner
    ```

1. Download the repository from https://github.com/SimoneAvellino/CMiner.
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

## Usage

[//]: #
[//]: # 'Once installed, CMiner can be used in three different ways:'
[//]: #
[//]: # '1. **Command Line Interface (CLI)**:'
[//]: # '    Run directly from the command line with the following syntax:'

```bash
CMiner <db_file> -s <support> [mining_options]
CMiner <db_file> -c <num_clusters> [clustering_options]
```

[//]: #
[//]: # "2. **Using Python's `-m` flag**:"
[//]: # '   Alternatively, you can execute CMiner as a Python module:'
[//]: # '```bash'
[//]: # 'python -m CMiner <db_file> <support> [options]'
[//]: # ' ```'
[//]: #
[//]: # '2. **As a Python module**:'
[//]: # '   You can also import CMiner into your Python code and use it programmatically:'
[//]: # '   '
[//]: # '```python'
[//]: # 'from CMiner import CMiner'
[//]: #
[//]: # 'miner = CMiner('
[//]: # "    db_file='/path/to/your/db/graphs.data', # required"
[//]: # '    support=0.5,                            # required'
[//]: # '    min_nodes=1,'
[//]: # "    max_nodes=float('inf'),"
[//]: # '    show_mappings=False,'
[//]: # '    output_path=None,'
[//]: # '    start_patterns=None,'
[//]: # '    is_directed=False,'
[//]: # '    with_frequencies=False,'
[//]: # '    only_closed_patterns=False'
[//]: # ')'
[//]: #
[//]: # 'miner.mine()'
[//]: # '```'
[//]: #

#### Required arguments:

-   `db_file`: Absolute path to the graph database file.
-   `-s`, `--support`: **(Mining mode)** Minimum support for pattern extraction. Specify a value between `0` and `1` to represent a percentage (e.g., `0.2` for 20%) or an absolute number (e.g., `20` for at least 20 graphs). To find patterns in all graphs, use `1` (100%). For patterns in at least one graph, use a value greater than `1` (e.g., `1.1`).
-   `-c`, `--num_clusters`: **(Clustering mode)** Number of clusters to generate, or `auto` to estimate it with the Silhouette method.

`-s` and `-c` are mutually exclusive: exactly one mode must be selected.

#### Mining options (used only with `-s`):

-   `-l`, `--min_nodes`: Minimum number of nodes in the pattern (default: 1).
-   `-u`, `--max_nodes`: Maximum number of nodes in the pattern (default: infinite).
-   `-n`, `--num_nodes`: Exact number of nodes in the pattern (if this option is set, -l and -u are not considered).
-   `-d`, `--is_directed`: Flag to indicate if the graphs are directed (default: 0, undirected).
-   `-m`, `--show_mappings`: Display mappings of found patterns (default: 0, not displayed).
-   `-t`, `--templates_file`: File path to start the search. The index of the nodes must start from 0.
-   `-f`, `--with_frequencies`: Display for each pattern the frequency in each graph. (default: 0, not displayed).
-   `-x`, `--pattern_type`: Flag to indicate the type of pattern that CMiner return. It can be 'all', 'maximum' (default: all) NOTE: this feature is under development, it could have bug.
-   `-o`, `--output_path`: File path to save results, if not set the results are shown in the console.
-   `-w`, `--worker`: Number of parallel workers to mine the patterns.

#### Clustering options (used only with `-c`):

##### Global flags (all clustering strategies)

| Flag | Default | Description |
| --- | --- | --- |
| `-d`, `--is_directed` | `0` | Graph direction flag (0 = undirected, 1 = directed). |
| `-o`, `--output_path` | `None` | Output root folder path for clustering results. CMiner creates a subfolder named `cluster_[filename]`; inside it, one file per cluster (`cluster_i_j`) plus a `README.md`. |
| `--strategy` | `simple_structural` | Distance-matrix strategy: `simple_structural`, `flexible_subgraph` or `ged`. |
| `--init_method` | `random` | Medoid initialization: `random` or `kmeans++`. |
| `--max_iter` | `100` | Maximum clustering iterations. |
| `--tolerance` | `1e-4` | Convergence tolerance for medoid updates. |
| `--verbose` | `0` | Show distance-matrix computation progress logs (`0` = off, `1` = on). |

##### Strategy: `simple_structural`

| Flag | Values | Default | Description |
| --- | --- | --- | --- |
| No strategy-specific flags | - | - | This strategy uses only the global clustering flags. |

##### Strategy: `flexible_subgraph`

| Flag | Values | Default | Description |
| --- | --- | --- | --- |
| `--subgraph_method` | `nodes`, `edges` | `nodes` | Subgraph extraction mode: `nodes` (node-induced) or `edges` (edge-induced). |

##### Strategy: `ged`

| Flag | Values | Default | Description |
| --- | --- | --- | --- |
| `--ged_timeout` | float | `5.0` | Timeout (seconds) for each GED pairwise computation. If `<= 0`, timeout is disabled. |
| `--ged_normalize` | `0`, `1` | `1` | Normalize GED values using a graph-size upper bound (`1` = enabled). |
| `--ged_node_del_cost` | float | `1.0` | Node deletion cost used by GED. |
| `--ged_node_ins_cost` | float | `1.0` | Node insertion cost used by GED. |
| `--ged_edge_del_cost` | float | `1.0` | Edge deletion cost used by GED. |
| `--ged_edge_ins_cost` | float | `1.0` | Edge insertion cost used by GED. |

Current clustering implementation computes a graph distance matrix (via the selected strategy) and applies a medoid-based clustering routine.

Note: `flexible_subgraph` can be computationally expensive on medium/large graphs because it enumerates many subgraph combinations.

Detailed GED documentation is available in `src/CCluster/strategies/GED_STRATEGY.md`.

#### Basic usage example

-   Mine patterns from 2 up to 3 nodes, present in at least 50% of graphs in the database.

```bash
CMiner /path/to/db.data -s 0.5 -l 2 -u 3
```

-   Mine all patterns present in at least 2 graphs in the database that have exactly 5 nodes.

```bash
CMiner /path/to/db.data -s 2 -n 5
```

-   Start graph clustering with 4 clusters:

```bash
CMiner /path/to/db.data -c 4 --init_method kmeans++ --max_iter 200
```

-   Start graph clustering with automatic cluster-count selection (Silhouette):

```bash
CMiner /path/to/db.data -c auto --strategy simple_structural
```

-   Start graph clustering with the new flexible strategy (node-induced subgraphs):

```bash
CMiner /path/to/db.data -c 4 --strategy flexible_subgraph --subgraph_method nodes
```

-   Start graph clustering with the new flexible strategy (edge-induced subgraphs):

```bash
CMiner /path/to/db.data -c 4 --strategy flexible_subgraph --subgraph_method edges
```

-   Start graph clustering with progress logs enabled:

```bash
CMiner /path/to/db.data -c 4 --strategy flexible_subgraph --subgraph_method nodes --verbose 1
```

-   Start graph clustering with Graph Edit Distance (GED):

```bash
CMiner /path/to/db.data -c 4 --strategy ged --ged_timeout 3 --ged_normalize 1
```

#### Template usage examples

Some usage examples from the folder `experiments/Datasets/OntoUML`:

-   Mine all patterns present in at least 2 graphs in the database that match the template defined in `S1.txt`:

```bash
CMiner ./ontographs.data -s 2 -t ./S1.txt -n 3
```

Note: we specify `-n 3` so that only solutions that are exactly the template are returned.

<div style="display: flex; align-items: flex-start; gap: 40px;">

  <div style="background-color: #1e1e1e; padding: 16px; border-radius: 8px; color: white; font-family: monospace;">
  <strong>File:</strong>
  <pre><code>t # 1
v 0 kind
v 1 subkind
v 2 subkind
e 1 0 Generalization
e 2 0 Generalization</code></pre>
</div>

  <div>
    <div style="font-family: sans-serif; font-size: 16px; margin-bottom: 8px;"><strong>Graphically:</strong></div>
    <img src="https://anonymous.4open.science/r/CMiner/img/S1.png" alt="S1 Graph" style="max-width: 300px; border-radius: 4px;"/>
  </div>

</div>

-   Same as before, but this time node labels are not specified:

```bash
CMiner ./ontographs.data -s 2 -t ./S2.txt -n 3
```

<div style="display: flex; align-items: flex-start; gap: 40px;">

  <div style="background-color: #1e1e1e; padding: 16px; border-radius: 8px; color: white; font-family: monospace;">
  <strong>File:</strong>
  <pre><code>t # 1
v 0
v 1
v 2
e 1 0 Generalization
e 2 0 Generalization</code></pre>
</div>

  <div>
    <div style="font-family: sans-serif; font-size: 16px; margin-bottom: 8px;"><strong>Graphically:</strong></div>
    <img src="https://anonymous.4open.science/r/CMiner/img/S2.png" alt="S1 Graph" style="max-width: 300px; border-radius: 4px;"/>
  </div>

</div>

-   You can also partially or completely omit labels for both nodes and edges:

```bash
CMiner ./ontographs.data -s 2 -t ./S3.txt -n 3
```

<div style="display: flex; align-items: flex-start; gap: 40px;">

  <div style="background-color: #1e1e1e; padding: 16px; border-radius: 8px; color: white; font-family: monospace;">
  <strong>File:</strong>
  <pre><code>t # 1
v 0 kind
v 1
v 2
e 1 0
e 2 0</code></pre>
</div>

  <div>
    <div style="font-family: sans-serif; font-size: 16px; margin-bottom: 8px;"><strong>Graphically:</strong></div>
    <img src="https://anonymous.4open.science/r/CMiner/img/S3.png" alt="S1 Graph" style="max-width: 300px; border-radius: 4px;"/>
  </div>

</div>
