import json
import sys
import re

def load_graphs(graphs_file):
    """
    Carica il database dei grafi dal file ontographs.data.
    
    In questo formato ogni grafo è definito in una sezione che inizia con una riga 
    del tipo "t # <id> <nome_grafo>" (case insensitive). Le righe successive definiscono i nodi
    e gli archi:
      - I nodi sono definiti da righe che iniziano con "v " e hanno il formato:
            v <id_nodo> <etichetta_nodo>
      - Gli archi sono definiti da righe che iniziano con "e " e hanno il formato:
            e <id_nodo_source> <id_nodo_target> <etichetta_arco>
    Una riga vuota indica la fine della definizione del grafo corrente.
    
    Restituisce un dizionario con struttura:
      {
         "nome_grafo": {
             "nodes": { "id_nodo": "etichetta_nodo", ... },
             "edges": [ { "source": "id_nodo_source", "target": "id_nodo_target", "label": "etichetta_arco" }, ... ]
         },
         ...
      }
    """
    graphs = {}
    current_graph = None
    nodes = {}
    edges = []
    with open(graphs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_graph is not None:
                    graphs[current_graph] = {"nodes": nodes, "edges": edges}
                    current_graph = None
                    nodes = {}
                    edges = []
                continue
            if line.lower().startswith("t #"):
                if current_graph is not None:
                    graphs[current_graph] = {"nodes": nodes, "edges": edges}
                current_graph = line.split(' ')[-1]  # Nome del grafo
                nodes = {}
                edges = []
            elif line.lower().startswith("v "):
                parts = line.split()
                if len(parts) >= 3:
                    node_id = parts[1]
                    label = " ".join(parts[2:])
                    nodes[node_id] = label
            elif line.lower().startswith("e "):
                parts = line.split()
                if len(parts) >= 4:
                    source = parts[1]
                    target = parts[2]
                    label = " ".join(parts[3:])
                    edges.append({
                        "source": source,
                        "target": target,
                        "label": label
                    })
        if current_graph is not None:
            graphs[current_graph] = {"nodes": nodes, "edges": edges}
    return graphs

def parse_cminer_output(cminer_file):
    """
    Effettua il parsing del file di output di cminer, estraendo per ciascun pattern:
      - La riga identificativa del pattern ("t ...")
      - I vertici del pattern ("v <id> <etichetta>")
      - Gli archi ("e <id1> <id2> <etichetta>"), salvati a scopo informativo
      - Il supporto ("s ...") e la frequenza ("f ...")
      - La sezione "info:" che, per ciascun grafo target, indica il numero di matching e i mapping.
      
    Viene estratto il mapping dei nodi dalla notazione:
         ({0->429, 1->87}, {(1, 0, 0)->('87', '429', 205)})
    che restituisce il dizionario: {"0": "429", "1": "87"}.
    
    Restituisce una lista di pattern, ognuno rappresentato da un dizionario.
    """
    patterns = []
    current_pattern = None
    mapping_section = False

    with open(cminer_file, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("t "):
                if current_pattern is not None:
                    patterns.append(current_pattern)
                current_pattern = {
                    "pattern_id": line,
                    "vertices": {},
                    "edges": [],
                    "support": None,
                    "frequency": None,
                    "mappings": []
                }
                mapping_section = False
            elif line.startswith("v "):
                parts = line.split()
                vid = parts[1]
                label = " ".join(parts[2:])
                current_pattern["vertices"][vid] = label
            elif line.startswith("e "):
                parts = line.split()
                source = parts[1]
                target = parts[2]
                label = " ".join(parts[3:])
                current_pattern["edges"].append({
                    "source": source,
                    "target": target,
                    "label": label
                })
            elif line.startswith("s "):
                current_pattern["support"] = line.split()[1]
            elif line.startswith("f "):
                current_pattern["frequency"] = line.split()[1]
            elif line.startswith("info:"):
                mapping_section = True
            elif mapping_section:
                mapping_file_match = re.match(r"^(.*?)\s+(\d+)$", line)
                if mapping_file_match:
                    mapping_filename = mapping_file_match.group(1).strip()
                    count = int(mapping_file_match.group(2))
                    current_pattern["mappings"].append({
                        "filename": mapping_filename,
                        "count": count,
                        "mappings": []
                    })
                else:
                    mapping_match = re.search(r"\(\{([^}]+)\},", line)
                    if mapping_match:
                        nodes_map_str = mapping_match.group(1)
                        node_mappings = {}
                        for item in nodes_map_str.split(" "):
                            item = item.strip()
                            # rimuovi le virgole
                            item = item.replace(",", "")                            
                            if "->" in item:
                                src, tgt = item.split("->")
                                node_mappings[src.strip()] = tgt.strip()
                        mapping_entry = {"node_mapping": node_mappings}
                        if current_pattern["mappings"]:
                            current_pattern["mappings"][-1]["mappings"].append(mapping_entry)
    if current_pattern is not None:
        patterns.append(current_pattern)
    return patterns

def evaluate_mappings(patterns, graphs, directed_edges=False):
    """
    Per ciascun pattern e per ogni mapping, verifica che:
      1. Per ogni nodo del pattern (definito con "v") l'etichetta attesa corrisponda all'etichetta 
         del nodo target ottenuto dalla mappatura (tramite ontographs.data).
      2. Per ogni arco del pattern (definito con "e <source> <target> <etichetta>"),
         la mappatura dei nodi determini un arco nel grafo target.
         Se l'arco non viene trovato nella direzione originale, si controlla se esiste
         anche l'inverso (dato che alcuni archi possono essere interpretati in maniera non orientata).
    
    Restituisce una lista di dizionari, ognuno relativo ad un mapping, che includono:
      - "pattern": informazioni sul pattern (pattern_id, vertices, edges, support, frequency)
      - "graph": il nome del grafo target
      - "mapping": il dizionario dei nodi mappati (es. {"0": "429", "1": "87"})
      - "correct": True se tutti i controlli passano, False altrimenti
      - "details": un elenco di messaggi relativi ad eventuali discrepanze.
    """
    results = []
    for pat in patterns:
        # Includo le informazioni sul pattern per poterle consultare manualmente
        pattern_info = {
            "pattern_id": pat.get("pattern_id"),
            "vertices": pat.get("vertices"),
            "edges": pat.get("edges"),
            "support": pat.get("support"),
            "frequency": pat.get("frequency")
        }
        for mapping_block in pat.get("mappings", []):
            graph_filename = mapping_block["filename"]
            graph = graphs.get(graph_filename)
            if not graph:
                results.append({
                    "pattern": pattern_info,
                    "graph": graph_filename,
                    "mapping": None,
                    "correct": False,
                    "details": [f"Il grafo '{graph_filename}' non è stato trovato nel database."]
                })
                continue
            graph_nodes = graph.get("nodes", {})
            graph_edges = graph.get("edges", [])
            for mapping in mapping_block.get("mappings", []):
                node_map = mapping.get("node_mapping", {})
                mapping_result = {
                    "pattern": pattern_info,
                    "graph": graph_filename,
                    "mapping": node_map,
                    "correct": True,
                    "details": []
                }
                # Controllo dei nodi
                for p_node, p_label in pat.get("vertices", {}).items():
                    t_node = node_map.get(p_node)
                    if t_node is None:
                        mapping_result["correct"] = False
                        mapping_result["details"].append(
                            f"Nodo del pattern '{p_node}' non presente nel mapping."
                        )
                    else:
                        target_label = graph_nodes.get(t_node)
                        if target_label != p_label:
                            mapping_result["correct"] = False
                            mapping_result["details"].append(
                                f"Per il nodo '{p_node}', atteso '{p_label}' ma trovato '{target_label}' (nodo target '{t_node}')."
                            )
                # Controllo degli archi
                for edge in pat.get("edges", []):
                    p_source = edge["source"]
                    p_target = edge["target"]
                    p_label = edge["label"]
                    t_source = node_map.get(p_source)
                    t_target = node_map.get(p_target)
                    if t_source is None or t_target is None:
                        mapping_result["correct"] = False
                        mapping_result["details"].append(
                            f"Impossibile valutare l'arco da '{p_source}' a '{p_target}' perché manca il mapping di uno dei nodi."
                        )
                    else:
                        found = False
                        for target_edge in graph_edges:
                            # Verifica sia nella direzione originale che invertita
                            if ((target_edge["source"] == t_source and target_edge["target"] == t_target) or
                                (not directed_edges and target_edge["source"] == t_target and target_edge["target"] == t_source)) and \
                                target_edge["label"] == p_label:
                                found = True
                                break
                        if not found:
                            mapping_result["correct"] = False
                            mapping_result["details"].append(
                                f"Arco del pattern da '{p_source}' a '{p_target}' con label '{p_label}', mappato in target come '{t_source}-{t_target}', non trovato {'(considerate entrambe le direzioni)' if not directed_edges else ''} nel grafo '{graph_filename}'."
                            )
                results.append(mapping_result)
    return results

def main():
    # Esecuzione:
    # python validate_mapping.py output.txt ontographs.data result.txt
    if len(sys.argv) != 5:
        print("Uso: python validate_mapping.py <cminer_output_file> <graphs_file> <result_file> <directed_edges>")
        sys.exit(1)
        
    cminer_file = sys.argv[1]
    graphs_file = sys.argv[2]
    result_file = sys.argv[3]
    directed_edges = sys.argv[4].lower() == 'true'

    graphs = load_graphs(graphs_file)
    patterns = parse_cminer_output(cminer_file)
    eval_results = evaluate_mappings(patterns, graphs, directed_edges)
    
    wrong_mappings = [res for res in eval_results if not res["correct"]]

    with open(result_file, 'w', encoding='utf-8') as fout:
        if len(wrong_mappings) == 0:
            fout.write("Tutti i mapping sono corretti.\n")
        else:
            json.dump(wrong_mappings, fout, indent=2, ensure_ascii=False)

    print(f"Valutazione completata. Risultati salvati in '{result_file}'.")

if __name__ == "__main__":
    main()
