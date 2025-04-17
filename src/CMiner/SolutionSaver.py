from abc import ABC, abstractmethod
import threading
import queue
# from CMiner.CMiner import Pattern

class SolutionSaver(ABC):
    
    def __init__(self, is_directed: bool = False, show_mappings: bool = False, show_frequencies: bool = False):
        """
        Initialize the SolutionSaver.
        """
        self.pattern_count = 0
        self.is_directed = is_directed
        self.show_mappings = show_mappings
        self.show_frequencies = show_frequencies
    
    @abstractmethod
    def save(self, pattern: 'Pattern'):
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

    def __init__(self, filename: str, is_directed: bool = False, show_mappings: bool = False, show_frequencies: bool = False):
        super().__init__(is_directed, show_mappings, show_frequencies)
        self.filename = filename
        self.queue = queue.Queue()
        self.thread = None
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
        
    def patter_to_str(self, pattern: 'Pattern') -> str:
        """
        Converts a pattern to a string representation.
        """
        console_str = ""
        file_str = ""
        self.pattern_count += 1
        console_str = f"t # {self.pattern_count}\n"
        console_str += pattern.directed_pattern_str() if self.is_directed else pattern.undirected_pattern_str()
        console_str += f"s {pattern.support()}\n"
        console_str += f"f {pattern.frequency()}\n"

        file_str = console_str
        if self.show_frequencies or self.show_mappings:
            console_str += f"\ninfo:\n"
            console_str += f"{pattern.granular_frequencies_str()}\n" if self.show_frequencies and not self.show_mappings else ""
            console_str += f"{pattern.mappings_str()}\n" if self.show_mappings else ""

            file_str += f"\ninfo:\n"
            file_str += f"{pattern.granular_frequencies_str()}\n" if self.show_frequencies and not self.show_mappings else ""
            file_str += f"{pattern.mappings_str(mapping_info = True)}\n" if self.show_mappings else ""
        
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
        with open(self.filename, 'w') as f:
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
    def __init__(self, is_directed: bool = False, show_mappings: bool = False, show_frequencies: bool = False):
        super().__init__(is_directed, show_mappings, show_frequencies)
        
    def patter_to_str(self, pattern: 'Pattern') -> str:
        """
        Converts a pattern to a string representation.
        """
        self.pattern_count += 1
        output = f"t # {self.pattern_count}\n"
        output += pattern.directed_pattern_str() if self.is_directed else pattern.undirected_pattern_str()
        output += f"s {pattern.support()}\n"
        output += f"f {pattern.frequency()}\n"
        if self.show_frequencies or self.show_mappings:
            output += f"\ninfo:\n"
            output += f"{pattern.granular_frequencies_str()}\n" if self.show_frequencies and not self.show_mappings else ""
            output += f"{pattern.mappings_str()}\n" if self.show_mappings else ""
        output += "----------\n"
        return output

    def save(self, pattern: 'Pattern'):
        """
        Prints the solution to the console.
        """
        print(self.patter_to_str(pattern))

    def close(self):
        """
        No action needed for console saver.
        """
        pass