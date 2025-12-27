import time
import argparse
from CMiner import CMiner


def main_function():
    parser = argparse.ArgumentParser(description="CMiner algorithm")
    parser.add_argument("db_file", type=str, help="Path to graph db")
    parser.add_argument("support", type=float, help="Support")
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

    args = parser.parse_args()

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
