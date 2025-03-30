#!/usr/bin/env python3
"""
Single Dataset Graph Preprocessing
-----------------------------------
Questo script accetta da riga di comando:
1) numero di thread,
2) path al dataset dei metadati,
3) path al dataset di retrieval.

Poi crea un solo grafo di metadati (basato sugli articoli unici e le relazioni
autori in comune, stesso anno, stessa rivista, ecc.), e lo salva su disco.
"""

import json
import pickle
import time

import networkx as nx
from datasets import load_from_disk
from itertools import chain
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Nome dello split su cui calcolare le relazioni (eventualmente parametrizzabile).
FNAME = "train"

# =============================================================================
# Funzioni di Utilità
# =============================================================================

def dict_of_lists_to_list_of_dicts(dict_of_lists):
    """
    Converte un dizionario di liste in una lista di dizionari.
    """
    keys = list(dict_of_lists.keys())
    lists = list(dict_of_lists.values())
    length = len(lists[0])
    if not all(len(lst) == length for lst in lists):
        raise ValueError("All lists must be della stessa lunghezza")
    return [dict(zip(keys, values)) for values in zip(*lists)]


def turn_a_list_of_lists_into_a_set(lst):
    """
    Appiattisce una lista di liste in un insieme di elementi unici.
    """
    flat_list = list(chain.from_iterable(lst))
    unique_set = set(flat_list)
    print(f"Vecchi elementi: {len(flat_list)}, nuovi elementi unici: {len(unique_set)}")
    return unique_set


def treat_topk_articles_as_unique_articles(fname, db_metadata, retr_info_ds):
    """
    Estrae articoli unici basandosi sugli indici ottenuti dal dataset di retrieval.
    """
    articles_indexes = retr_info_ds[fname]["retrieval_relative_indices"]
    unique_indexes = turn_a_list_of_lists_into_a_set(articles_indexes)
    # Rimuovi eventuali -1, se ci sono
    unique_indexes = [x for x in unique_indexes if x != -1]
    return db_metadata[fname][unique_indexes]


def remove_redundant_lists(lst):
    """
    Rimuove liste duplicate da una lista di liste.
    """
    return [list(rel) for rel in set(tuple(rel) for rel in lst)]


def process_relation_index(i, fname, db_metadata, retr_info_ds, relation_handler):
    """
    Elabora le relazioni per un singolo indice dell'array retrieval_relative_indices.
    """
    articles = db_metadata[fname][retr_info_ds[fname]["retrieval_relative_indices"][i]]
    articles = dict_of_lists_to_list_of_dicts(articles)
    local_relations = []
    for doc in articles:
        for other_doc in articles:
            if doc["PMID"] != other_doc["PMID"]:
                if relation_handler.same_year(doc["Year"], other_doc["Year"]):
                    local_relations.append([
                        doc["PMID"],
                        other_doc["PMID"],
                        relation_handler.metadata_relations_to_id["same_year"]
                    ])
                if relation_handler.same_conference(doc["JournalName"], other_doc["JournalName"]):
                    local_relations.append([
                        doc["PMID"],
                        other_doc["PMID"],
                        relation_handler.metadata_relations_to_id["same_journal"]
                    ])
                if relation_handler.same_author_1_in_common(
                    doc["Metadata"]["AuthorList"], other_doc["Metadata"]["AuthorList"]
                ):
                    local_relations.append([
                        doc["PMID"],
                        other_doc["PMID"],
                        relation_handler.metadata_relations_to_id["author_in_common"]
                    ])
                if relation_handler.same_chemical_term_1_in_common(
                    doc["Metadata"]["ChemicalTerms"], other_doc["Metadata"]["ChemicalTerms"]
                ):
                    local_relations.append([
                        doc["PMID"],
                        other_doc["PMID"],
                        relation_handler.metadata_relations_to_id["chemical_term_in_common"]
                    ])
                if relation_handler.same_mesh_term_3_in_common(
                    doc["Metadata"]["MeshTerms"], other_doc["Metadata"]["MeshTerms"]
                ):
                    local_relations.append([
                        doc["PMID"],
                        other_doc["PMID"],
                        relation_handler.metadata_relations_to_id["mesh_term_in_common"]
                    ])
    return local_relations


def create_metadata_relations_parallel(fname, db_metadata, retr_info_ds, relation_handler, num_workers=4):
    """
    Crea le relazioni sui metadati in parallelo usando ProcessPoolExecutor.
    """
    indices = retr_info_ds[fname]["retrieval_relative_indices"]
    all_relations = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_relation_index, i, fname, db_metadata, retr_info_ds, relation_handler)
            for i in range(len(indices))
        ]
        for future in tqdm(as_completed(futures), total=len(indices), desc="Parallel creating metadata relations"):
            try:
                all_relations.extend(future.result())
            except Exception as e:
                print(f"Error in processing index: {e}")
    return all_relations

# =============================================================================
# Classe RelationHandler
# =============================================================================

class RelationHandler:
    """
    Gestisce la mappatura e la creazione delle relazioni tra articoli.
    """
    def __init__(self):
        # Relazioni "metadati" (autori, anno, rivista, ecc.)
        # (Abbiamo rimosso le relazioni cliniche di DDB, se non servono.)
        self.metadata_relations = [
            "author_in_common", "mesh_term_in_common", "chemical_term_in_common",
            "same_journal", "same_year"
        ]
        # Mappiamo le relazioni su interi (ID) per la costruzione del grafo
        self.metadata_relations_to_id = {
            w: i for i, w in enumerate(self.metadata_relations)
        }

    def encounters(self, list1, list2):
        return [l for l in list1 if l in list2]

    def at_least_N_things_in_common(self, list1, list2, n):
        list1 = list1 or []
        list2 = list2 or []
        return len(self.encounters(list1, list2)) >= n

    def same_author_1_in_common(self, authors1, authors2):
        return self.at_least_N_things_in_common(authors1, authors2, 1)

    def same_chemical_term_1_in_common(self, chem_list1, chem_list2):
        return self.at_least_N_things_in_common(chem_list1, chem_list2, 1)

    def same_mesh_term_2_in_common(self, mesh_list1, mesh_list2):
        return self.at_least_N_things_in_common(mesh_list1, mesh_list2, 2)

    def same_mesh_term_3_in_common(self, mesh_list1, mesh_list2):
        return self.at_least_N_things_in_common(mesh_list1, mesh_list2, 3)

    def same_conference(self, conf1, conf2):
        return conf1 == conf2

    def same_year(self, year1, year2):
        return year1 == year2

# =============================================================================
# Funzione per costruire il grafo e salvarlo
# =============================================================================

def construct_metadata_graph(article_pmids, metadata_relations, output_path, relation_handler):
    """
    Costruisce il grafo dei metadati da una lista di relazioni e lo salva.
    
    - article_pmids: lista di tutti i PMID (o ID unici) presenti
    - metadata_relations: relazioni calcolate (sorgente, destinazione, ID relazione)
    - output_path: dove salvare il grafo in formato pickle
    """
    # Mappiamo i PMID in ID numerici
    concept2id = {pmid: i for i, pmid in enumerate(article_pmids)}
    # Ricaviamo l'elenco di relazioni come stringhe
    id2relation = relation_handler.metadata_relations
    relation2id = relation_handler.metadata_relations_to_id
    
    graph = nx.MultiDiGraph()

    # Creiamo gli archi
    for (subj_pmid, obj_pmid, rel_id) in metadata_relations:
        try:
            s = concept2id[subj_pmid]
            o = concept2id[obj_pmid]
        except KeyError:
            # Se per qualche motivo uno dei PMID non è presente nel concept2id, saltiamo
            continue
        # Aggiungiamo l'arco (subj -> obj) con attributo 'rel'
        graph.add_edge(s, o, rel=rel_id, weight=1.0)
        # Se vogliamo un grafo bidirezionale, aggiungiamo l'altro verso
        graph.add_edge(o, s, rel=rel_id, weight=1.0)

    # Salviamo su disco
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    return graph


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    # -----------------------------------------------------------
    # Lettura parametri da linea di comando
    # -----------------------------------------------------------
    num_threads = int(input("Specify number of threads: "))

    while True:
        try:
            dataset_metadata_path = input("Specify dataset for metadata relations: ")
            db_w_metadata = load_from_disk(dataset_metadata_path)
            dataset_retr_path = input("Specify dataset for retrieval information: ")
            retr_info_ds = load_from_disk(dataset_retr_path)
            break
 
        except FileNotFoundError:
            print(FileNotFoundError)
            print("Please specify a valid dataset path.")
            continue
        
    # -----------------------------------------------------------
    # Estrazione articoli unici
    # -----------------------------------------------------------
    unique_articles = treat_topk_articles_as_unique_articles(FNAME, db_w_metadata, retr_info_ds)
    # Ricaviamo la lista dei PMID (o ID unici)
    pmid_list = unique_articles["PMID"]

    # -----------------------------------------------------------
    # Creazione delle relazioni (in parallelo)
    # -----------------------------------------------------------
    relation_handler = RelationHandler()
    print(f"Creating metadata relations for split '{FNAME}' using {num_threads} threads...")
    relations = create_metadata_relations_parallel(FNAME, db_w_metadata, retr_info_ds, relation_handler, num_threads)

    # Rimozione duplicati
    orig_len = len(relations)
    relations = remove_redundant_lists(relations)
    final_len = len(relations)
    print(f"Relazioni totali: {orig_len}, dopo rimozione duplicati: {final_len}")

    # Salvataggio relazioni su file .json (facoltativo; utile per debug)
    relations_json_path = "metadata_relations.json"
    with open(relations_json_path, "w") as f:
        json.dump(relations, f, ensure_ascii=False)
    print(f"Saved relations to {relations_json_path}")

    # -----------------------------------------------------------
    # Creazione del grafo dei metadati e salvataggio
    # -----------------------------------------------------------
    graph_path = "metadata_graph.pickle"
    print(f"Building graph and saving to {graph_path}...")
    g = construct_metadata_graph(
        article_pmids=pmid_list,
        metadata_relations=relations,
        output_path=graph_path,
        relation_handler=relation_handler
    )

    end_time = time.time()
    print(f"Graph construction completed in {end_time - start_time:.2f} seconds.")
    print(f"Grafo salvato in {graph_path}")
