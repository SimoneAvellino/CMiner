"""Default no-op enrichment strategy.

Acts as boilerplate / safe default for the CCluster pipeline: it returns the
input graphs unchanged. Replace with a real :class:`EnrichGraphSemanticsStrategy`
implementation to actually augment the semantic content of the database.
"""

from typing import List

from .enrichment_strategy import EnrichGraphSemanticsStrategy, EnrichmentContext


class NoOpEnrichmentStrategy(EnrichGraphSemanticsStrategy):
    """Identity enrichment strategy: returns ``context.db_graphs`` untouched."""

    @property
    def name(self) -> str:
        return "noop"

    def enrich(self, context: EnrichmentContext) -> List:
        verbose = bool(context.strategy_params.get("verbose", False))
        if verbose:
            print(
                "[noop_enrichment] No semantic enrichment applied "
                f"({len(context.db_graphs)} graphs passed through)."
            )
        return context.db_graphs
