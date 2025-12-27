import threading

from .Pattern import Pattern
from .SolutionSaver import ConsoleSolutionSaver, FileSolutionSaver


class DFSStack(list):

    def __init__(self, min_nodes, max_nodes, output_options):
        """
        Stack that only keeps patterns that can be extended.
        """
        super().__init__()
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        # Dictionary that keeps track of already computed pattern.
        # key   := pattern_code
        # value := list of patterns
        self.found_patterns = set()
        self.output_options = output_options
        # In case of closed pattern mining, keep track of the last popped pattern
        # in case of backtracking it is printed.
        self.last_popped_pattern = None
        # Memoize BitMatrices
        self.bit_matrices = {}
        # solution saver
        if output_options["output_path"]:
            self.solution_saver = FileSolutionSaver(
                output_options["output_path"],
                output_options["show_mappings"],
                output_options["with_frequencies"],
            )
        else:
            self.solution_saver = ConsoleSolutionSaver(
                output_options["show_mappings"], output_options["with_frequencies"]
            )
        # Lock to guarantee exclusive stack access in concurrent scenarios
        self._lock = threading.RLock()

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
                code = pattern.canonical_code()
                if code not in self.found_patterns:
                    self.found_patterns.add(code)
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
        code = pattern.canonical_code()
        with self._lock:
            return code in self.found_patterns

    def output(self, pattern: Pattern):
        if len(pattern.nodes()) < self.min_nodes:
            return
        with self._lock:
            self.solution_saver.save(pattern)

    def close(self):
        """
        Close the solution saver.
        """
        self.solution_saver.close()
