import warnings

# Must be set before any import that transitively loads urllib3, otherwise
# the warning is emitted during urllib3.__init__ before the filter is active.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import os
import time
import argparse
import sys

from CCluster import CCluster
from CCluster.CCluster import CLUSTERING_REGISTRY, EMBEDDING_REGISTRY, ENRICHMENT_REGISTRY


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


def _build_clustering_parser():
    # Strategy names are derived directly from the registries so the help text
    # stays in sync automatically when new strategies are added.
    enrichment_names = ", ".join(ENRICHMENT_REGISTRY)
    embedding_names = ", ".join(EMBEDDING_REGISTRY)
    clustering_names = ", ".join(CLUSTERING_REGISTRY)

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
            f"names: {enrichment_names}. "
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
            f"{embedding_names}. Example: "
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
            f"{clustering_names}. Example: "
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
    parser.add_argument(
        "--grid_compute",
        action="store_true",
        help=(
            "Run every combination of enrichment × embedding × clustering strategies "
            "and write results to separate subfolders inside -o. "
            "-o is required when this flag is set."
        ),
    )
    return parser


# Clustering parameters that map to first-class kwargs of ``CCluster``
# (they are shared with the auto-K silhouette grid search). Anything not in
# this allow-list is forwarded to ``clustering_params`` as a free-form dict.
_CLUSTERING_KWARGS = {"init_method", "max_iter", "tolerance"}

# ------------------------------------------------------------------ grid compute


def _combo_folder_name(enr, emb, cls):
    return f"enr={enr}__emb={emb}__cls={cls}"


def _write_combo_readme(combo_folder, db_file, num_clusters, is_directed, enr, emb, cls):
    """Write a README.md describing this combination and how to reproduce it.

    Descriptions are read directly from the strategy instances so this
    function never needs to be updated when new strategies are added.
    """
    enr_desc = ENRICHMENT_REGISTRY[enr].description
    emb_desc = EMBEDDING_REGISTRY[emb].description
    cls_desc = CLUSTERING_REGISTRY[cls].description

    parent_output = os.path.dirname(combo_folder)

    cmd = (
        f"CMiner {db_file} -c {num_clusters} -d {is_directed} \\\n"
        f"    --enrichment_strategy {enr} \\\n"
        f"    --embedding_strategy {emb} \\\n"
        f"    --clustering_strategy {cls} \\\n"
        f"    -o {parent_output}"
    )

    lines = [
        f"# Experiment: {enr} + {emb} + {cls}",
        "",
        "## Pipeline Configuration",
        "",
        "| Stage | Strategy |",
        "|---|---|",
        f"| Enrichment | `{enr}` |",
        f"| Embedding | `{emb}` |",
        f"| Clustering | `{cls}` |",
        "",
        "## Strategy Details",
        "",
        f"### Enrichment — `{enr}`",
        "",
        enr_desc,
        "",
        f"### Embedding — `{emb}`",
        "",
        emb_desc,
        "",
        f"### Clustering — `{cls}`",
        "",
        cls_desc,
        "",
        "## Command to Replicate",
        "",
        "```bash",
        cmd,
        "```",
        "",
    ]

    readme_path = os.path.join(combo_folder, "README.md")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _write_grid_summary_readme(output_path, db_file, num_clusters, is_directed, results):
    """Write the top-level README summarising all combinations and their outcome."""
    import datetime

    today = datetime.date.today().isoformat()

    lines = [
        "# Grid Compute Summary",
        "",
        f"**Database:** `{db_file}`  ",
        f"**num\\_clusters:** `{num_clusters}`  ",
        f"**is\\_directed:** `{is_directed}`  ",
        f"**Generated:** {today}  ",
        "",
        "## Results",
        "",
        "| # | Enrichment | Embedding | Clustering | Time (s) | Status | Folder |",
        "|---|---|---|---|---|---|---|",
    ]

    for i, (enr, emb, cls, elapsed, status, folder_name) in enumerate(results, start=1):
        elapsed_str = f"{elapsed:.1f}" if elapsed is not None else "—"
        if status == "ok":
            status_cell = "✓ ok"
        else:
            short = status[:60] + "…" if len(status) > 60 else status
            status_cell = f"✗ {short}"
        lines.append(
            f"| {i} | `{enr}` | `{emb}` | `{cls}` "
            f"| {elapsed_str} | {status_cell} | `{folder_name}` |"
        )

    ok_count = sum(1 for *_, s, _ in results if s == "ok")
    fail_count = len(results) - ok_count

    lines += [
        "",
        f"**Total combinations:** {len(results)}  ",
        f"**Succeeded:** {ok_count}  ",
        f"**Failed:** {fail_count}  ",
        "",
        "## How to Re-run a Single Experiment",
        "",
        "Each subfolder contains a `README.md` with the exact `CMiner` command",
        "to replicate that specific combination with its default parameters.",
        "",
        "## Folder Structure",
        "",
        "```",
        f"{os.path.basename(output_path)}/",
        "├── README.md                   ← this file",
        "├── enr=<e>__emb=<b>__cls=<c>/",
        "│   ├── README.md               ← per-experiment description & command",
        "│   ├── tsne.png                ← t-SNE scatter plot coloured by cluster",
        "│   └── cluster_<db_name>/      ← cluster files (one per cluster)",
        "│       ├── cluster_0_<n>",
        "│       ├── cluster_1_<n>",
        "│       └── README.md",
        "└── ...",
        "```",
        "",
    ]

    readme_path = os.path.join(output_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _run_grid_compute(args):
    """Iterate every combination drawn from the live strategy registries.

    The distance matrix only depends on the (enrichment, embedding) pair, so
    it is computed ONCE per pair and reused across all clustering strategies.
    """
    if not args.output_path:
        print(
            "Error: --grid_compute requires -o / --output_path to be specified.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.output_path, exist_ok=True)

    n_enr = len(ENRICHMENT_REGISTRY)
    n_emb = len(EMBEDDING_REGISTRY)
    n_cls = len(CLUSTERING_REGISTRY)
    total = n_enr * n_emb * n_cls
    col_cls = max(len(c) for c in CLUSTERING_REGISTRY)

    print(f"grid_compute  {n_enr} enrichment × {n_emb} embedding × {n_cls} clustering = {total} runs")
    print(f"output        {args.output_path}")
    print()

    verbose = bool(args.verbose)
    results = []
    combo_idx = 0
    t_total = time.time()

    for enr in ENRICHMENT_REGISTRY:
        for emb in EMBEDDING_REGISTRY:

            # --- distance matrix (computed once per enr+emb pair) ----------
            t_matrix = time.time()
            matrix_error = None
            distance_matrix = None
            graph_names = None
            base_db = None

            print(f"  enr={enr}  emb={emb}", end="  ", flush=True)

            try:
                base = CCluster(
                    db_file=args.db_file,
                    num_clusters=args.num_clusters,
                    directed_graph=args.is_directed,
                    enrichment_strategy=enr,
                    embedding_strategy=emb,
                    auto_k_max=args.auto_k_max,
                    enrichment_params={"verbose": verbose},
                    embedding_params={"verbose": verbose},
                    quiet=True,
                )
                distance_matrix, graph_names = base.prepare_distance_matrix()
                base_db = base.db
            except KeyboardInterrupt:
                elapsed = time.time() - t_matrix
                print(f"\ninterrupted (matrix). Partial results saved.")
                for cls in CLUSTERING_REGISTRY:
                    combo_idx += 1
                    fn = _combo_folder_name(enr, emb, cls)
                    results.append((enr, emb, cls, elapsed, "interrupted", fn))
                _write_grid_summary_readme(
                    args.output_path, args.db_file, args.num_clusters,
                    args.is_directed, results,
                )
                sys.exit(0)
            except Exception as exc:  # noqa: BLE001
                matrix_error = type(exc).__name__ + ": " + str(exc)

            elapsed_matrix = time.time() - t_matrix

            if matrix_error:
                print(f"matrix ERROR: {matrix_error}")
                for cls in CLUSTERING_REGISTRY:
                    combo_idx += 1
                    fn = _combo_folder_name(enr, emb, cls)
                    results.append((enr, emb, cls, elapsed_matrix, matrix_error, fn))
                print()
                continue

            print(f"matrix {elapsed_matrix:.1f}s")

            # --- clustering strategies (one line each) ---------------------
            for cls in CLUSTERING_REGISTRY:
                combo_idx += 1
                folder_name = _combo_folder_name(enr, emb, cls)
                combo_folder = os.path.join(args.output_path, folder_name)
                os.makedirs(combo_folder, exist_ok=True)
                _write_combo_readme(
                    combo_folder, args.db_file, args.num_clusters,
                    args.is_directed, enr, emb, cls,
                )

                t_cls = time.time()
                status = "ok"
                try:
                    clusterer = CCluster(
                        db_file=args.db_file,
                        num_clusters=args.num_clusters,
                        directed_graph=args.is_directed,
                        output_path=combo_folder,
                        clustering_strategy=cls,
                        auto_k_max=args.auto_k_max,
                        clustering_params={"verbose": verbose},
                        quiet=True,
                    )
                    clusterer.db = base_db
                    clusterer.cluster_and_emit(distance_matrix, graph_names)
                except KeyboardInterrupt:
                    elapsed_cls = time.time() - t_cls
                    print(f"\ninterrupted at [{combo_idx}/{total}]. Partial results saved.")
                    results.append(
                        (enr, emb, cls, elapsed_matrix + elapsed_cls,
                         "interrupted", folder_name)
                    )
                    _write_grid_summary_readme(
                        args.output_path, args.db_file, args.num_clusters,
                        args.is_directed, results,
                    )
                    sys.exit(0)
                except Exception as exc:  # noqa: BLE001
                    status = type(exc).__name__ + ": " + str(exc)

                elapsed_cls = time.time() - t_cls
                results.append(
                    (enr, emb, cls, elapsed_matrix + elapsed_cls, status, folder_name)
                )

                # One compact line per clustering run
                if status == "ok":
                    k_info = ""
                    if clusterer.auto_k_result is not None:
                        k, sil = clusterer.auto_k_result
                        k_info = f"k={k:<3}  sil={sil:.4f}"
                    elif not isinstance(args.num_clusters, str):
                        k_info = f"k={args.num_clusters}"
                    tsne_tag = "  [tsne.png]" if os.path.exists(
                        os.path.join(combo_folder, "tsne.png")
                    ) else ""
                    print(
                        f"    [{combo_idx:>{len(str(total))}}/{total}]"
                        f"  {cls:<{col_cls}}  {k_info}  {elapsed_cls:.1f}s  ✓{tsne_tag}"
                    )
                else:
                    short_err = status[:60] + "…" if len(status) > 60 else status
                    print(
                        f"    [{combo_idx:>{len(str(total))}}/{total}]"
                        f"  {cls:<{col_cls}}  ERROR: {short_err}"
                    )

            print()

    _write_grid_summary_readme(
        args.output_path, args.db_file, args.num_clusters,
        args.is_directed, results,
    )

    ok_count = sum(1 for *_, s, _ in results if s == "ok")
    elapsed_total = time.time() - t_total
    print(f"done  {ok_count}/{total} succeeded  {elapsed_total:.1f}s total")
    print(f"summary → {os.path.join(args.output_path, 'README.md')}")


# ------------------------------------------------------------------ entry point

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

    if args.grid_compute:
        _run_grid_compute(args)
        return

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
