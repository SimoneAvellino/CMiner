import time
import argparse
import sys
from CCluster import CCluster


def _num_clusters_arg(value):
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "num_clusters must be a positive integer or 'auto'."
        ) from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("num_clusters must be greater than zero.")

    return parsed


def _build_base_parser():
    parser = argparse.ArgumentParser(description="CMiner algorithm")
    parser.add_argument("db_file", type=str, help="Path to graph db")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "-s",
        "--support",
        type=float,
        help="Minimum support for mining",
        default=None,
    )
    mode_group.add_argument(
        "-c",
        "--num_clusters",
        type=_num_clusters_arg,
        help="Number of clusters for graph clustering, or 'auto'",
        default=None,
    )
    return parser


def _build_mining_parser():
    parser = _build_base_parser()
    parser.add_argument(
        "-l", "--min_nodes", type=int, help="Minimum number of nodes", default=1
    )
    parser.add_argument(
        "-u",
        "--max_nodes",
        type=int,
        help="Maximum number of nodes",
        default=float("inf"),
    )
    parser.add_argument(
        "-n", "--num_nodes", type=int, help="Number of nodes", default=None
    )
    parser.add_argument(
        "-m", "--show_mappings", type=int, help="Show pattern mappings", default=0
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Output file", default=None
    )
    parser.add_argument(
        "-t", "--templates_file", type=str, help="Starting template file", default=None
    )
    parser.add_argument(
        "-d",
        "--is_directed",
        type=int,
        help="Specify if the graph is directed",
        default=0,
    )
    parser.add_argument(
        "-f",
        "--with_frequencies",
        type=int,
        help="Show the relative frequencies of the pattern",
        default=0,
    )
    parser.add_argument(
        "-x", "--pattern_type", type=str, help="[all | maximum]", default="all"
    )
    parser.add_argument(
        "-w", "--worker", type=int, help="Number of parallel workers", default=1
    )
    return parser


def _coerce_value(s: str):
    """Best-effort cast of a CLI string token to bool / int / float / None / str."""
    sl = s.lower()
    if sl == "true":
        return True
    if sl == "false":
        return False
    if sl == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_strategy(values):
    """Parse a strategy CLI argument of the form

        NAME [key1=val1 key2=val2 ...]

    into ``(name, {key: coerced_value, ...})``.
    """
    if not values:
        raise argparse.ArgumentTypeError("strategy must specify a name")
    name = values[0]
    params = {}
    for kv in values[1:]:
        if "=" not in kv:
            raise argparse.ArgumentTypeError(
                f"strategy parameters must be in key=value form, got: {kv!r}"
            )
        key, value = kv.split("=", 1)
        params[key.strip()] = _coerce_value(value.strip())
    return name, params


# Strategy names accepted by the resolvers in CCluster. Listed here only for
# the --help text; the actual validation happens inside CCluster.
_ENRICHMENT_NAMES = ("noop", "semantic_label_clustering")
_EMBEDDING_NAMES = ("simple_structural", "flexible_subgraph")
_CLUSTERING_NAMES = ("kmedoids",)


def _build_clustering_parser():
    parser = _build_base_parser()
    parser.add_argument(
        "-d",
        "--is_directed",
        type=int,
        help="Specify if the graph is directed",
        default=0,
    )
    parser.add_argument(
        "-o", "--output_path", type=str, help="Output file", default=None
    )
    parser.add_argument(
        "--enrichment_strategy",
        nargs="+",
        metavar=("NAME", "KEY=VAL"),
        default=["noop"],
        help=(
            "Semantic-enrichment strategy and its parameters. Available "
            f"names: {', '.join(_ENRICHMENT_NAMES)}. "
            "Pass parameters as space-separated key=value tokens, e.g. "
            "'--enrichment_strategy semantic_label_clustering "
            "model_name=all-MiniLM-L6-v2 max_k=8'."
        ),
    )
    parser.add_argument(
        "--embedding_strategy",
        nargs="+",
        metavar=("NAME", "KEY=VAL"),
        default=["simple_structural"],
        help=(
            "Embedding strategy and its parameters. Available names: "
            f"{', '.join(_EMBEDDING_NAMES)}. Example: "
            "'--embedding_strategy flexible_subgraph subgraph_method=nodes "
            "min_size=3 max_size=5'."
        ),
    )
    parser.add_argument(
        "--clustering_strategy",
        nargs="+",
        metavar=("NAME", "KEY=VAL"),
        default=["kmedoids"],
        help=(
            "Clustering strategy and its parameters. Available names: "
            f"{', '.join(_CLUSTERING_NAMES)}. Example: "
            "'--clustering_strategy kmedoids init_method=kmeans++ "
            "max_iter=200 tolerance=1e-4'."
        ),
    )
    parser.add_argument(
        "--auto_k_max",
        type=int,
        help=(
            "Upper bound for the silhouette grid search when -c auto is used. "
            "Default: ceil(sqrt(n_graphs))."
        ),
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1],
        help="Show pipeline progress logs (0: off, 1: on)",
        default=0,
    )
    return parser


# Clustering parameters that map to first-class kwargs of ``CCluster``
# (they are shared with the auto-K silhouette grid search). Anything not in
# this allow-list is forwarded to ``clustering_params`` as a free-form dict.
_CLUSTERING_KWARGS = {"init_method", "max_iter", "tolerance"}


def main_function():
    argv = sys.argv[1:]
    has_support_mode = any(
        arg == "-s" or arg.startswith("--support") for arg in argv
    )
    has_cluster_mode = any(
        arg == "-c" or arg.startswith("--num_clusters") for arg in argv
    )

    if has_support_mode:
        args = _build_mining_parser().parse_args()
        from CMiner.CMiner import CMiner

        if args.num_nodes is not None:
            args.min_nodes = args.num_nodes
            args.max_nodes = args.num_nodes

        miner = CMiner(
            args.db_file,
            support=args.support,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            show_mappings=args.show_mappings,
            output_path=args.output_path,
            templates_file=args.templates_file,
            directed_graph=args.is_directed,
            with_frequencies=args.with_frequencies,
            pattern_type=args.pattern_type,
            workers=args.worker,
        )

        start_time = time.time()

        try:
            miner.mine()
        except KeyboardInterrupt:
            print("\n-> Ctrl+C detected. Closing miner...")
            miner.close()
        finally:
            end_time = time.time()
            print(f"\n-> Execution time: {end_time - start_time} seconds")
        return

    if not has_cluster_mode:
        _build_base_parser().parse_args()
        return

    args = _build_clustering_parser().parse_args()

    enrich_name, enrich_params = _parse_strategy(args.enrichment_strategy)
    embed_name, embed_params = _parse_strategy(args.embedding_strategy)
    cluster_name, cluster_params = _parse_strategy(args.clustering_strategy)

    # Propagate the global --verbose flag into all three strategy contexts
    # (per-strategy 'verbose=...' override still wins).
    verbose = bool(args.verbose)
    enrich_params.setdefault("verbose", verbose)
    embed_params.setdefault("verbose", verbose)
    cluster_params.setdefault("verbose", verbose)

    # Promote well-known clustering knobs from cluster_params to first-class
    # CCluster constructor kwargs (the orchestrator also reuses them inside
    # the auto-K silhouette grid search).
    promoted_kwargs = {
        key: cluster_params.pop(key)
        for key in list(cluster_params)
        if key in _CLUSTERING_KWARGS
    }

    clusterer = CCluster(
        db_file=args.db_file,
        num_clusters=args.num_clusters,
        directed_graph=args.is_directed,
        output_path=args.output_path,
        enrichment_strategy=enrich_name,
        embedding_strategy=embed_name,
        clustering_strategy=cluster_name,
        auto_k_max=args.auto_k_max,
        enrichment_params=enrich_params,
        embedding_params=embed_params,
        clustering_params=cluster_params,
        **promoted_kwargs,
    )

    start_time = time.time()

    try:
        clusterer.cluster()
    except NotImplementedError as exc:
        print(f"\n-> Clustering not available yet: {exc}")
    except KeyboardInterrupt:
        print("\n-> Ctrl+C detected. Closing clustering...")
    finally:
        end_time = time.time()
        print(f"\n-> Execution time: {end_time - start_time} seconds")
