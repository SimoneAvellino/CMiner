# CMiner

CMiner is an algorithm for mining patterns from graphs using a user-defined support technique. This implementation provides a command-line interface for running the algorithm, with configurable options like minimum and maximum nodes, support, and search approach.



## Installation

### Prerequisites

Make sure you have the following requirements to run the project:

- **Python**: Version 3.11.6
- **pip**: Version  24.2 or higher


### Installation steps

1. Download the repository from https://github.com/SimoneAvellino/CMiner.
2. Move into the repository folder:
    ```bash
    cd CMiner
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the library in `editable` mode:
    ```bash
    pip install -e .
    ```

## Usage

[//]: # ()
[//]: # (Once installed, CMiner can be used in three different ways:)

[//]: # ()
[//]: # (1. **Command Line Interface &#40;CLI&#41;**:)

[//]: # (    Run directly from the command line with the following syntax:)

```bash
CMiner <db_file> <support> [options]
 ```

[//]: # ()
[//]: # (2. **Using Python's `-m` flag**:)

[//]: # (   Alternatively, you can execute CMiner as a Python module:)

[//]: # (```bash)

[//]: # (python -m CMiner <db_file> <support> [options])

[//]: # ( ```)

[//]: # ()
[//]: # (2. **As a Python module**:)

[//]: # (   You can also import CMiner into your Python code and use it programmatically:)

[//]: # (   )
[//]: # (```python)

[//]: # (from CMiner import CMiner)

[//]: # ()
[//]: # (miner = CMiner&#40;)

[//]: # (    db_file='/path/to/your/db/graphs.data', # required)

[//]: # (    support=0.5,                            # required)

[//]: # (    min_nodes=1,)

[//]: # (    max_nodes=float&#40;'inf'&#41;,)

[//]: # (    show_mappings=False,)

[//]: # (    output_path=None,)

[//]: # (    start_patterns=None,)

[//]: # (    is_directed=False,)

[//]: # (    with_frequencies=False,)

[//]: # (    only_closed_patterns=False)

[//]: # (&#41;)

[//]: # ()
[//]: # (miner.mine&#40;&#41;)

[//]: # (```)

[//]: # ()



### Required arguments:
- `db_file`: Absolute path to the graph database file.
- `support`: **Minimum support for pattern extraction**: Specify a value between `0` and `1` to represent a percentage (e.g., `0.2` for 20%) or an absolute number (e.g., `20` for at least 20 graphs). To find patterns in all graphs, use `1` (100%). For patterns in at least one graph, use a value greater than `1` (e.g., `1.1`).


### Additional options:
- `-l`, `--min_nodes`: Minimum number of nodes in the pattern (default: 1).
- `-u`, `--max_nodes`: Maximum number of nodes in the pattern (default: infinite).
- `-n`, `--num_nodes`: Exact number of nodes in the pattern (if this option is set, -l and -u are not considered).
- `-d`, `--directed`: Flag to indicate if the graphs are directed (default: 0, not directed).
- `-m`, `--show_mappings`: Display mappings of found patterns (default: 0, not displayed).
- `-t`, `--templates_path`: File paths to start the search. The index of the nodes must start from 0.
- `-f`, `--with_frequencies`: Display for each pattern the frequency in each graph. (default: 0, not displayed).
- `-x`, `--pattern_type`: Flag to indicate the type of pattern that CMiner return. It can be 'all', 'maximum' (default: all).
- `-o`, `--output_path`: File path to save results, if not set the results are shown in the console. 

### Basic usage example

- Mine patterns from 2 up to 3 nodes, present in at least 50% of graphs in the database.

```bash
CMiner /path/to/db.data 0.5 -l 2 -u 3
 ```

- Mine all patterns present in at least 2 graphs in the database that have exactly 5 nodes.

```bash
CMiner /path/to/db.data 2 -n 5
```

### Template usage examples

Some usage examples from the folder `test/Datasets/OntoUML`:

- Mine all patterns present in at least 2 graphs in the database that match the template defined in `S1.txt`:

```bash
CMiner ./ontographs.data 2 -t ./S1.txt -n 3
```

Note: we specify `-n 3` so that only solutions that are exactly the template are returned.

<div style="display: flex; align-items: flex-start; gap: 40px;">

  <div style="background-color: #1e1e1e; padding: 16px; border-radius: 8px; color: white; font-family: monospace;">
  <strong>File:</strong>
  <pre><code>v 0 kind
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

- Same as before, but this time node labels are not specified:

```bash
CMiner ./ontographs.data 2 -t ./S2.txt -n 3
```
<div style="display: flex; align-items: flex-start; gap: 40px;">

  <div style="background-color: #1e1e1e; padding: 16px; border-radius: 8px; color: white; font-family: monospace;">
  <strong>File:</strong>
  <pre><code>v 0
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

- You can also partially or completely omit labels for both nodes and edges:


```bash
CMiner ./ontographs.data 2 -t ./S3.txt -n 3
```
<div style="display: flex; align-items: flex-start; gap: 40px;">

  <div style="background-color: #1e1e1e; padding: 16px; border-radius: 8px; color: white; font-family: monospace;">
  <strong>File:</strong>
  <pre><code>v 0 kind
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
