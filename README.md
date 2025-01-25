# CMiner

CMiner is an algorithm for mining patterns from graphs using a user-defined support technique. This implementation provides a command-line interface for running the algorithm, with configurable options like minimum and maximum nodes, support, and search approach.

## Prerequisites

Make sure you have the following requirements to run the project:

- **Python**: Version 3.11 or higher
- **pip**: Version  24.2 or higher

## Installation

### Prerequisites

- Python 3.x
- `pip` (Python package manager)

### Installation steps

1. Download the repository from https://anonymous.4open.science/r/CMiner.
2. Move into the repository folder.

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
- `support`: **Minimum support for pattern extraction**: Specify a value between `0` and `1` for percentage (e.g., `0.2` for 20%) or an absolute number (e.g., `20` for at least 20 graphs).


### Additional options:
- `-l`, `--min_nodes`: Minimum number of nodes in the pattern (default: 1).
- `-u`, `--max_nodes`: Maximum number of nodes in the pattern (default: infinite).
- `-n`, `--num_nodes`: Exact number of nodes in the pattern (if not set -l and -u are considered).
- `-d`, `--directed`: Flag to indicate if the graphs are directed (default: 0).
- `-m`, `--show_mappings`: Display mappings of found patterns (default: 0).
- `-t`, `--templates_path`: File paths to start the search.
- `-f`, `--with_frequencies`: Flag to indicate if the frequencies of the patterns for each graph must be shown (default: 0).

[//]: # (- `-o`, `--output_path`: File path to save results, if not set the results are shown in the console.)
[//]: # (- `-c`, `--only_closed_patterns`: Flag to indicate if only closed patterns should be shown &#40;default: False&#41;.)

### Usage example

Some usage examples from the folder `test/Datasets/toy-db`:

- Mine patterns from 2 up to 3 nodes, present in at least 50% of graphs in the database.

```bash
CMiner /test/Datasets/toy-db/db1.data 0.5 -l 4 -u 8
 ```

- Mining all patterns present in at least 2 graphs in the database.

```bash
CMiner ./test/Datasets/toy-db/db3.data 2
 ```

- Mining all patterns present in at least 2 graphs in the database that have the pattern inside the file `pattern.txt`

```bash
CMiner ./test/Datasets/toy-db/db3.data 2 -t ./test/Datasets/toy-db/pattern.txt
```
Content of `./test/Datasets/toy-db/pattern.txt`
```bash
v 0 red
v 1 yellow
e 0 1 white
 ```