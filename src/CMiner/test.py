# clear && python -m cProfile -s time -o ./Desktop/CMiner_times/s_40.prof -m CMiner ./Desktop/CMiner/test/Datasets/OntoUML-db/graphs.data 50

import pstats
import sys

# Controlla che sia stato passato un argomento
if len(sys.argv) < 2:
    print("Uso: python script.py <percorso_file.prof>")
    sys.exit(1)

# Legge il percorso del file .prof dalla riga di comando
prof_file = sys.argv[1]

try:
    # Carica i risultati del profiling
    p = pstats.Stats(prof_file)
    p.strip_dirs()
    # p.print_stats().sort_stats('cumtime')
    # Ordina per tempo totale di esecuzione
    # p.strip_dirs().sort_stats('cumtime').print_stats('mine|was_stacked|apply_node_extension|apply_edge_extension|find_edge_extensions|find_node_extensions')
    p.print_callees('mine')
    # p.print_callees('all_edges_of_subgraph')
    p.print_callees('find_node_extensions')
    p.print_callees('find_edge_extensions')

    # p.print_callees('get_edge_labels_with_duplicate')
    p.print_callees('add')



except FileNotFoundError:
    print(f"Errore: File '{prof_file}' non trovato.")
except Exception as e:
    print(f"Errore durante l'analisi del file: {e}")

