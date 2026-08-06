from abc import ABC, abstractmethod
import threading
import queue


class SolutionSaver(ABC):

    def __init__(self, show_mappings: bool = False, show_frequencies: bool = False):
        """
        Initialize the SolutionSaver.
        """
        self.pattern_count = 0
        self.show_mappings = show_mappings
        self.show_frequencies = show_frequencies

    @abstractmethod
    def save(self, pattern: "Pattern"):
        """
        Abstract method to save a solution.
        """
        pass

    def close(self):
        """
        Abstract method to close the solution saver.
        Subclasses should implement this to ensure the queue is empty before closing.
        """
        pass


class FileSolutionSaver(SolutionSaver):
    """
    This class is responsible for saving solutions to a file in a separate thread.
    """

    def __init__(
        self, filename: str, show_mappings: bool = False, show_frequencies: bool = False
    ):
        super().__init__(show_mappings, show_frequencies)
        self.filename = filename
        self.queue = queue.Queue()
        self.thread = None
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def patter_to_str(self, pattern: "Pattern") -> str:
        """
        Converts a pattern to a string representation.
        """
        console_str = ""
        file_str = ""
        self.pattern_count += 1
        console_str = f"t # {self.pattern_count}\n"
        console_str += pattern.__str__()
        console_str += f"s {pattern.support()}\n"
        console_str += f"f {pattern.frequency()}\n"

        file_str = console_str
        if self.show_frequencies or self.show_mappings:
            console_str += f"\ninfo:\n"
            console_str += (
                f"{pattern.granular_frequencies_str()}\n"
                if self.show_frequencies and not self.show_mappings
                else ""
            )
            console_str += f"{pattern.mappings_str()}\n" if self.show_mappings else ""

            file_str += f"\ninfo:\n"
            file_str += (
                f"{pattern.granular_frequencies_str()}\n"
                if self.show_frequencies and not self.show_mappings
                else ""
            )
            file_str += (
                f"{pattern.pattern_mappings.__str__()}\n" if self.show_mappings else ""
            )

        console_str += "----------\n"
        file_str += "----------\n"
        return console_str, file_str

    def save(self, solution: str):
        """
        Adds a solution to the queue to be saved.
        """
        self.queue.put(solution)

    def close(self):
        """
        Stops the background thread and closes the file.
        """
        self.queue.join()
        self.running = False
        self.queue.put(None)  # Sentinel value to signal shutdown
        if self.thread:
            self.thread.join()

    def _run(self):
        """
        Background thread method to save solutions to the file.
        """
        with open(self.filename, "w") as f:
            while self.running:
                pattern = self.queue.get()
                if pattern is None:  # Sentinel value to exit the loop
                    break
                console_str, file_str = self.patter_to_str(pattern)
                print(console_str)
                f.write(file_str)
                self.queue.task_done()


class ConsoleSolutionSaver(SolutionSaver):
    """
    This class is responsible for saving solutions to the console.
    """

    def __init__(self, show_mappings: bool = False, show_frequencies: bool = False):
        super().__init__(show_mappings, show_frequencies)

    def patter_to_str(self, pattern: "Pattern") -> str:
        """
        Converts a pattern to a string representation.
        """

        self.pattern_count += 1
        output = f"t # {self.pattern_count}\n"
        output += pattern.__str__()
        output += f"s {pattern.support()}\n"
        output += f"f {pattern.frequency()}\n"

        # print()
        # for g, maps in pattern.pattern_mappings.patterns_mappings.items():
        #     output += f"g {g}\n"
        #     for m in maps:
        #         output += f"    "
        #         for k, v in m._retrieve_edge_mapping().items():
        #             output += f"{k} -> {v}, "
        #         output += "\n"

        if self.show_frequencies or self.show_mappings:
            output += f"\ninfo:\n"
            output += f"code: {pattern.canonical_code()}\n"
            output += (
                f"{pattern.granular_frequencies_str()}\n"
                if self.show_frequencies and not self.show_mappings
                else ""
            )
            output += f"{pattern.mappings_str()}\n" if self.show_mappings else ""
        output += "----------\n"
        return output

    def save(self, pattern: "Pattern"):
        """
        Prints the solution to the console.
        """
        print(self.patter_to_str(pattern))

    def close(self):
        """
        No action needed for console saver.
        """
        pass


class StringSolutionSaver(SolutionSaver):
    """
    Collects solutions into an in-memory string instead of printing or writing to file.
    Thread-safe: can be used with concurrent workers.
    """

    def __init__(self, show_mappings: bool = False, show_frequencies: bool = False):
        super().__init__(show_mappings, show_frequencies)
        self._parts: list[str] = []
        self._lock = threading.Lock()

    def patter_to_str(self, pattern: "Pattern") -> str:
        self.pattern_count += 1
        output = f"t # {self.pattern_count}\n"
        output += pattern.__str__()
        output += f"s {pattern.support()}\n"
        output += f"f {pattern.frequency()}\n"
        if self.show_frequencies or self.show_mappings:
            output += f"\ninfo:\n"
            output += (
                f"{pattern.granular_frequencies_str()}\n"
                if self.show_frequencies and not self.show_mappings
                else ""
            )
            output += f"{pattern.mappings_str()}\n" if self.show_mappings else ""
        output += "----------\n"
        return output

    def save(self, pattern: "Pattern"):
        s = self.patter_to_str(pattern)
        with self._lock:
            self._parts.append(s)

    def get_results(self) -> str:
        """Return all collected pattern strings joined together."""
        with self._lock:
            return "".join(self._parts)

    def close(self):
        pass


class QueueSolutionSaver(SolutionSaver):
    """
    Streams patterns through a thread-safe queue instead of accumulating them.

    This saver is meant to be consumed by a single iterator (typically from
    DFSStack.iter_results / CMinerAPI.mine_stream). Each produced Pattern is
    enqueued as-is; the consumer formats it on demand, so the library never
    builds the full output string in memory.
    """

    def __init__(
        self,
        show_mappings: bool = False,
        show_frequencies: bool = False,
        maxsize: int = 0,
    ):
        super().__init__(show_mappings, show_frequencies)
        # maxsize=0 (default) keeps the queue unbounded,
        # maxsize!=0 blocks mining workers when the queue is full to not fill RAM.
        self._queue = queue.Queue(maxsize=maxsize)
        self._closed = False

    def save(self, pattern: "Pattern"):
        """Enqueue a pattern for the streaming consumer."""
        self._queue.put(pattern)

    def close(self):
        """Signal the consumer that no more patterns will be produced."""
        if not self._closed:
            self._closed = True
            # avoid blocking forever the producer if the consumer is slow or stopped.
            while True:
                try:
                    self._queue.put(None, timeout=0.5)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except queue.Empty:
                        pass

    def patter_to_str(self, pattern: "Pattern") -> str:
        """
        Converts a pattern to the same string representation used by
        StringSolutionSaver. The consumer calls this, so pattern numbering
        follows the order in which patterns are dequeued.
        """
        self.pattern_count += 1
        output = f"t # {self.pattern_count}\n"
        output += pattern.__str__()
        output += f"s {pattern.support()}\n"
        output += f"f {pattern.frequency()}\n"
        if self.show_frequencies or self.show_mappings:
            output += f"\ninfo:\n"
            output += (
                f"{pattern.granular_frequencies_str()}\n"
                if self.show_frequencies and not self.show_mappings
                else ""
            )
            output += f"{pattern.mappings_str()}\n" if self.show_mappings else ""
        output += "----------\n"
        return output

    def __iter__(self):
        return self

    def __next__(self) -> "Pattern":
        pattern = self._queue.get()
        if pattern is None:
            raise StopIteration
        return pattern
