# NetworkX
NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

# CONNECTORS

Implementation of connectors for NetworkX for creating a graph from a file.
The files' types could be: csv, json

## CSV case

<b> 1. nodes.csv</b>

| node_id | labels | prop_a |  ........ | prop_m |
|:--------|:------:|-------:|----------:|-------:|         
| 0       |  A:B   |     10 |  ........ |     18 |
| 1       |   C    |     21 |  ........ |     32 |
| ....    |  ....  |   .... |  ........ |   .... |
| ....    |  ....  |   .... |  ........ |   .... |
| n       |   D    |     12 |  ........ |     14 |

<b> 2. edges.csv</b>

| source | target | type  | prop_x |  ........ | prop_l |
|:------:|:------:|:-----:|-------:|----------:|-------:|
|   0    |   1    |   E   |    10  |  ........ |    18  |
|   1    |   2    |   F   |    21  |  ........ |    32  |
|  ...   |  ...   |  ...  |   ...  |  ........ |   ...  |
|  ...   |  ...   |  ...  |   ...  |  ........ |   ...  |
|   q    |   n    |   G   |    12  |  ........ |    14  |


<b>NOTE</b>: 


The networks edges could be created by the function `networkx.from_pandas_dataframe`

    from_pandas_dataframe(df, source, target, edge_attr=None,create_using=None)

    where:
       - df: DataFrame created by pandas.read_csv('edges.csv')
       - source: name of the column in df representing the source nodes
       - target: name of the column in df representing the target nodes
       - edge_attr: string or list of the columns in df representing the edge attributes.
         in our case: ['type', 'prop_x', ...., 'prop_l']
       - create_using: Use specified graph for result. 
          -- The default value is Graph() [undirected graph]
          -- DiGraph() [directed graph]

We could get several problems if the size of the csv file is higher than 2GB.

Nodes properties could be added by the function `set_node_attributes`

Therefore, we need to read the nodes.csv file, and then transform it into a python dictionary having such form:

    attrs = {
        0: {'labels': [A,B], 'prop_a': 10, ...., 'prop_m': 18},
        1: {'labels': ['C'], 'prop_a': 21, ...., 'prop_m': 32},
        ....
        n: {'labels': ['D'], 'prop_a': 12, ...., 'prop_m': 14}
    }

At this point:

    nx.set_node_attributes(G, attrs)



## JSON case

The network has to be formed:

    {
       'directed': False, 
       'multigraph': False, 
       'graph': {}, 
       'nodes': [
           {
              'id': 'A', 
              'p1': 20, 
                 ..., 
              'pn':21
           },
           {
               'id': 'B',
               'p1': 10,
                  ..., 
               'pn':25
           }
       ], 
       'links': [
           {
              'source': 'A',
              'target': 'B',
              'pa': 3, 
                   ..., 
              'pm':32
           }
       ]
    }

Starting from the above file, the network could be created by using the function `node_link_data`

    node_link_graph(data, directed=False, multigraph=True, *, source='source', target='target', name='id', key='key', link='links')

    where:
       - data: dict, 
         node-link formatted graph data

       - directed: bool, optional (default=False), 
         if True, and direction not specified in data, return a directed graph.

       - multigraph: bool, optional (default=True), 
         if True, and multigraph not specified in data, return a multigraph.

       - source: string, optional (default='source'), 
         a string that provides the ‘source’ attribute name for storing NetworkX-internal graph data.
       
       - target: string, optional (default='target')
         a string that provides the ‘target’ attribute name for storing NetworkX-internal graph data.

       - name: string, optional (default='id')
         a string that provides the ‘name’ attribute name for storing NetworkX-internal graph data.

       - key: string, optional (default='key')
         a string that provides the ‘key’ attribute name for storing NetworkX-internal graph data.
         key is only used for multigraphs
 
       - link: string, optional (default='links')