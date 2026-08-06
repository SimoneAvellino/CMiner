import threading

from .Pattern import Pattern
from .SolutionSaver import (
    ConsoleSolutionSaver,
    FileSolutionSaver,
    QueueSolutionSaver,
    StringSolutionSaver,
)


class DFSStack(list):

    def __init__(self, min_nodes, max_nodes, output_options):
        """
        Stack that only keeps patterns that can be extended.
        """
        super().__init__()
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        # Set that keeps track of already computed patterns.
        # Entries are compact (hash, prefix) fingerprints,
        # so the set stays small. Collision would cause a rare silent pattern skip.
        self.found_patterns = set()
        self.output_options = output_options
        # In case of closed pattern mining, keep track of the last popped pattern
        # in case of backtracking it is printed.
        self.last_popped_pattern = None
        # Memoize BitMatrices
        self.bit_matrices = {}
        # Streaming mode lets callers consume patterns one-by-one instead of
        # building a giant result string/list in memory.
        self.streaming_output = output_options.get("streaming_output", False)
        self.keep_structured_patterns = output_options.get(
            "keep_structured_patterns", not self.streaming_output
        )
        # solution saver
        if output_options["output_path"]:
            self.solution_saver = FileSolutionSaver(
                output_options["output_path"],
                output_options["show_mappings"],
                output_options["with_frequencies"],
            )
        elif self.streaming_output:
            self.solution_saver = QueueSolutionSaver(
                output_options["show_mappings"], output_options["with_frequencies"]
            )
        elif output_options.get("string_output", False):
            self.solution_saver = StringSolutionSaver(
                output_options["show_mappings"], output_options["with_frequencies"]
            )
        else:
            self.solution_saver = ConsoleSolutionSaver(
                output_options["show_mappings"], output_options["with_frequencies"]
            )
        # Store pattern objects for structured access
        self.output_structured_patterns: list[Pattern] = []
        # Lock to guarantee exclusive stack access in concurrent scenarios
        self._lock = threading.RLock()

    def is_empty(self):
        with self._lock:
            return len(self) == 0

    def pop(self, index=-1, backtracking=False) -> Pattern:
        """
        Pop the last element from the stack.
        """
        with self._lock:
            if (
                self.output_options["pattern_type"] == "maximum"
                and backtracking
                and self.last_popped_pattern
            ):
                self.output(self.last_popped_pattern)

            pattern = super().pop(index)
            self.last_popped_pattern = pattern

            return pattern

    def try_pop(self, backtracking=False) -> Pattern | None:
        """
        Safely pop the last element from the stack, returning None if empty.
        """
        with self._lock:
            if len(self) == 0:
                return None
            return self.pop(index=-1, backtracking=backtracking)

    @staticmethod
    def _fingerprint(code: str) -> tuple[int, str]:
        """
        Compact fingerprint of a canonical code: full hash + short prefix.
        A false "already stacked" hit needs both to collide (~2^-80).
        """
        return (hash(code), code[:16])

    def push(self, pattern: Pattern):
        """
        Push the pattern into the stack.
        """
        with self._lock:
            if (
                len(pattern.nodes()) <= self.max_nodes
                and not self.was_stacked(pattern)
                and pattern.frequency() > 0
            ):
                super().append(pattern)
                self.found_patterns.add(self._fingerprint(pattern.canonical_code()))
                if self.output_options["pattern_type"] != "maximum":
                    self.output(pattern)
            else:
                del pattern

    def was_stacked(self, pattern: Pattern):
        """
        Check if the pattern is already computed.

        NOTE: If two pattern are isomorphic, they have the same code.
              There could be two different patterns with the same code,
              so the code is used to prune the search space, but the
              isomorphism check is still needed.
        """
        fingerprint = self._fingerprint(pattern.canonical_code())
        with self._lock:
            return fingerprint in self.found_patterns

    def output(self, pattern: Pattern):
        if len(pattern.nodes()) < self.min_nodes:
            return
        with self._lock:
            self.solution_saver.save(pattern)
            if self.keep_structured_patterns:
                self.output_structured_patterns.append(pattern)

    def get_string_results(self) -> str:
        """
        Return the accumulated results string when using StringSolutionSaver.
        Raises TypeError if the current saver is not a StringSolutionSaver.
        """
        if not isinstance(self.solution_saver, StringSolutionSaver):
            raise TypeError(
                "get_string_results() is only available when string_output=True."
            )
        return self.solution_saver.get_results()

    def iter_results(self, yield_strings: bool = True):
        """
        Yield results one pattern at a time when streaming_output is enabled.

        Parameters
        ----------
        yield_strings : bool
            If True (default), yields formatted strings. If False, yields the
            raw Pattern objects so the caller can format or store them itself.

        Raises
        ------
        TypeError
            If the current saver is not a QueueSolutionSaver.
        """
        if not isinstance(self.solution_saver, QueueSolutionSaver):
            raise TypeError(
                "iter_results() is only available when streaming_output=True."
            )
        for pattern in self.solution_saver:
            if yield_strings:
                yield self.solution_saver.patter_to_str(pattern)
            else:
                yield pattern

    def close(self):
        """
        Close the solution saver.
        """
        self.solution_saver.close()
