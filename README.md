# CMiner

CMiner is an algorithm for mining patterns from graphs using a user-defined support technique. This implementation provides a command-line interface for running the algorithm, with configurable options like minimum and maximum nodes, support, and search approach.



## Installation

### Prerequisites

Make sure you have the following requirements to run the project:

- **Python**: Version 3.11.6 or higher
- **pip**: Version  24.2 or higher


### Installation steps

1. Download the repository from https://anonymous.4open.science/r/CMiner.
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

### Usage example

Some usage examples from the folder `test/Datasets/toy-db`:

- Mine patterns from 2 up to 3 nodes, present in at least 50% of graphs in the database.

```bash
CMiner /test/Datasets/toy-db/db1.data 0.5 -l 2 -u 3
 ```

- Mine all patterns present in at least 2 graphs in the database.

```bash
CMiner ./test/Datasets/toy-db/db3.data 2
 ```

- Mine all patterns present in at least 2 graphs in the database that have the template inside the file `pattern.txt`

```bash
CMiner ./test/Datasets/toy-db/db3.data 2 -t ./test/Datasets/toy-db/pattern.txt
```
Content of `./test/Datasets/toy-db/pattern.txt`
```bash
v 0 red
v 1 yellow
e 0 1 white
 ```