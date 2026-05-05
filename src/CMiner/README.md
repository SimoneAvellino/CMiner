<!-- @format -->

# Mining mode (`-s`)

Pattern mining over a graph database with a user-defined support threshold.

```bash
CMiner <db_file> -s <support> [mining_options]
```

## Required arguments

-   `db_file`: Absolute path to the graph database file.
-   `-s`, `--support`: Minimum support for pattern extraction. A value between
    `0` and `1` is interpreted as a percentage (e.g., `0.2` for 20%); a value
    `> 1` is interpreted as an absolute number of graphs (e.g., `20` for at
    least 20 graphs). Use `1` for "all graphs" (100%) and a value `> 1`
    (e.g., `1.1`) for "patterns appearing in at least one graph".

`-s` is mutually exclusive with `-c` (clustering mode).

## Mining options

| Flag | Default | Description |
| --- | --- | --- |
| `-l`, `--min_nodes` | `1` | Minimum number of nodes in the pattern. |
| `-u`, `--max_nodes` | `inf` | Maximum number of nodes in the pattern. |
| `-n`, `--num_nodes` | `None` | Exact number of nodes in the pattern. When set, `-l` and `-u` are ignored. |
| `-d`, `--is_directed` | `0` | `1` if the graphs are directed. |
| `-m`, `--show_mappings` | `0` | Display mappings of found patterns. |
| `-t`, `--templates_file` | `None` | File path to start the search. The index of the nodes must start from `0`. |
| `-f`, `--with_frequencies` | `0` | Display, for each pattern, its frequency in each graph. |
| `-x`, `--pattern_type` | `all` | `all` or `maximum`. Note: `maximum` is under development. |
| `-o`, `--output_path` | `None` | File path where results are saved. If unset, results are printed to stdout. |
| `-w`, `--worker` | `1` | Number of parallel workers. |

## Basic usage examples

Mine patterns from 2 up to 3 nodes, present in at least 50% of graphs:

```bash
CMiner /path/to/db.data -s 0.5 -l 2 -u 3
```

Mine all patterns present in at least 2 graphs that have exactly 5 nodes:

```bash
CMiner /path/to/db.data -s 2 -n 5
```

## Template usage examples

Some usage examples from the folder `experiments/Datasets/OntoUML`:

-   Mine all patterns present in at least 2 graphs in the database that match
    the template defined in `S1.txt`:

```bash
CMiner ./ontographs.data -s 2 -t ./S1.txt -n 3
```

Note: `-n 3` ensures that only solutions exactly matching the template are
returned.

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
    <img src="https://anonymous.4open.science/r/CMiner/img/S2.png" alt="S2 Graph" style="max-width: 300px; border-radius: 4px;"/>
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
    <img src="https://anonymous.4open.science/r/CMiner/img/S3.png" alt="S3 Graph" style="max-width: 300px; border-radius: 4px;"/>
  </div>

</div>
