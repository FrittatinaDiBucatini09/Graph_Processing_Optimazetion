#!/usr/bin/env python3
"""
Graph Preprocessing Script
---------------------------
Questo script carica i dataset MedMCQA e PubMedQA, esegue il preprocessing
dei metadati, costruisce grafi di conoscenza a partire da relazioni definite in un database (DDB)
e salva i risultati su disco.

Autore: [Il Tuo Nome]
Data: [Data]
"""

import os
import json
import pickle
import traceback
from collections import defaultdict
from itertools import chain

import numpy as np
import networkx as nx
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict
from huggingface_hub import login

# =============================================================================
# Configurazione e Caricamento Datasets
# =============================================================================

# Informazioni sui repository e sui dataset
REPO_ID_MEDCQA = "MGFiD-Group/MedMCQA_utils"
REPO_ID_PUBMEDQA = "MGFiD-Group/PubMedQA_utils"
METADATANAME = "database_w_metadata"
RETRIEVALINFONAME = "retrieval_infos"

# Split e nomi dei file
FNAMES = ["train"]
FNAME = "train"

# Caricamento dei dataset da disco
db_w_metadata_medcqa = load_from_disk("./data/hf/db_w_metadata_medcqa.hf")
retr_info_ds_medcqua = load_from_disk("./data/hf/retr_info_ds_medcqua.hf")
db_w_metadata_pubmedqa = load_from_disk("./data/hf/db_w_metadata_pubmedqa.hf")
retr_info_ds_pubmedqa = load_from_disk("./data/hf/retr_info_ds_pubmedqa.hf")

# =============================================================================
# Funzioni di Utilità
# =============================================================================

def dict_of_lists_to_list_of_dicts(dict_of_lists):
    """
    Converte un dizionario di liste in una lista di dizionari.
    
    Esempio:
        Input: {"name": ["Alice", "Bob"], "age": [30, 25]}
        Output: [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    
    Args:
        dict_of_lists (dict): Dizionario con liste di uguale lunghezza.
    
    Returns:
        list: Lista di dizionari.
    
    Raises:
        ValueError: Se le liste non sono tutte della stessa lunghezza.
    """
    keys = list(dict_of_lists.keys())
    lists = list(dict_of_lists.values())
    length = len(lists[0])
    if not all(len(lst) == length for lst in lists):
        raise ValueError("All lists must be of the same length")
    return [dict(zip(keys, values)) for values in zip(*lists)]


def turn_a_list_of_lists_into_a_set(lst):
    """
    Appiattisce una lista di liste in un insieme di elementi unici.
    
    Args:
        lst (list of lists): Lista di liste.
    
    Returns:
        set: Insieme di elementi unici.
    """
    flat_list = list(chain.from_iterable(lst))
    unique_set = set(flat_list)
    print(f"old elements: {len(flat_list)}, new elements: {len(unique_set)}")
    return unique_set


def treat_topk_articles_as_unique_articles(fname, db_metadata, retr_info_ds):
    """
    Estrae articoli unici basandosi sugli indici ottenuti dal dataset di retrieval.
    
    Args:
        fname (str): Nome dello split (es. "train").
        db_metadata (dict): Dataset dei metadati.
        retr_info_ds (dict): Dataset delle informazioni di retrieval.
    
    Returns:
        dict: Articoli filtrati in base agli indici unici.
    """
    articles_indexes = retr_info_ds[fname]["retrieval_relative_indices"]
    unique_indexes = turn_a_list_of_lists_into_a_set(articles_indexes)
    unique_indexes = list(filter(lambda x: x != -1, unique_indexes))
    return db_metadata[fname][unique_indexes]


def remove_newlines_from_list(input_list):
    """
    Rimuove eventuali caratteri di newline da ogni stringa in una lista.
    
    Args:
        input_list (list): Lista di stringhe.
    
    Returns:
        list: Lista di stringhe senza newline.
    """
    return [s.strip() for s in input_list]


def get_article_pointers(path):
    """
    Legge i puntatori agli articoli da un file.
    
    Args:
        path (str): Percorso del file.
    
    Returns:
        list: Lista di puntatori.
    """
    with open(path) as fin:
        lines = fin.readlines()
    return [line.strip() for line in lines]


def remove_redundant_lists(lst):
    """
    Rimuove liste duplicate da una lista di liste.
    
    Args:
        lst (list of lists): Lista di liste.
    
    Returns:
        list of lists: Lista senza duplicati.
    """
    return [list(rel) for rel in set(tuple(rel) for rel in lst)]


def load_metadata_relations(path):
    """
    Carica le relazioni dei metadati da un file JSON.
    
    Args:
        path (str): Percorso del file.
    
    Returns:
        list: Lista delle relazioni.
    """
    with open(path) as f:
        all_relas = json.load(f)
    # Se il file è un dizionario, restituisce i valori; altrimenti, restituisce l'oggetto stesso.
    return list(all_relas.values()) if isinstance(all_relas, dict) else all_relas


# =============================================================================
# Classe RelationHandler
# =============================================================================

class RelationHandler:
    """
    Gestisce la mappatura e la creazione delle relazioni tra articoli.
    """
    def __init__(self):
        self.merged_relations = [
            "belongs_to_the_category_of", "is_a_category", "may_cause",
            "is_a_subtype_of", "is_a_risk_factor_of", "is_associated_with",
            "may_contraindicate", "interacts_with", "belongs_to_the_drug_family_of",
            "belongs_to_drug_super-family", "is_a_vector_for", "may_be_allelic_with",
            "see_also", "is_an_ingradient_of", "may_treat"
        ]
        self.relas_dict = {
            "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 5, "10": 6,
            "12": 7, "16": 8, "17": 9, "18": 10, "20": 11, "26": 12,
            "30": 13, "233": 14
        }
        self.metadata_relations = [
            "author_in_common", "mesh_term_in_common", "chemical_term_in_common",
            "same_journal", "same_year", "name_entities_with_abstracts_relation",
            "god_document_dummy_node_connections", "qa_context_nodes"
        ]
        self.metadata_relations_to_id = {
            w: i + len(self.merged_relations) for i, w in enumerate(self.metadata_relations)
        }
        self.final_merged_relations = self.merged_relations + self.metadata_relations

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

    def create_metadata_relations(self, fname, db_metadata, retr_info_ds):
        """
        Crea relazioni sui metadati per lo split specificato.
        
        Args:
            fname (str): Nome dello split (es. "train").
            db_metadata (dict): Metadati degli articoli.
            retr_info_ds (dict): Informazioni di retrieval.
        
        Returns:
            list: Lista di relazioni.
        """
        json_data = []
        indices = retr_info_ds[fname]["retrieval_relative_indices"]
        for i in tqdm(range(len(indices)), desc="Creating metadata relations"):
            articles = db_metadata[fname][indices[i]]
            articles = dict_of_lists_to_list_of_dicts(articles)
            for doc in articles:
                for other_doc in articles:
                    if doc["PMID"] != other_doc["PMID"]:
                        if self.same_year(doc["Year"], other_doc["Year"]):
                            json_data.append([
                                doc["PMID"],
                                other_doc["PMID"],
                                self.metadata_relations_to_id["same_year"]
                            ])
                        if self.same_conference(doc["JournalName"], other_doc["JournalName"]):
                            json_data.append([
                                doc["PMID"],
                                other_doc["PMID"],
                                self.metadata_relations_to_id["same_journal"]
                            ])
                        if self.same_author_1_in_common(
                            doc["Metadata"]["AuthorList"],
                            other_doc["Metadata"]["AuthorList"]
                        ):
                            json_data.append([
                                doc["PMID"],
                                other_doc["PMID"],
                                self.metadata_relations_to_id["author_in_common"]
                            ])
                        if self.same_chemical_term_1_in_common(
                            doc["Metadata"]["ChemicalTerms"],
                            other_doc["Metadata"]["ChemicalTerms"]
                        ):
                            json_data.append([
                                doc["PMID"],
                                other_doc["PMID"],
                                self.metadata_relations_to_id["chemical_term_in_common"]
                            ])
                        if self.same_mesh_term_3_in_common(
                            doc["Metadata"]["MeshTerms"],
                            other_doc["Metadata"]["MeshTerms"]
                        ):
                            json_data.append([
                                doc["PMID"],
                                other_doc["PMID"],
                                self.metadata_relations_to_id["mesh_term_in_common"]
                            ])
        return json_data


# =============================================================================
# DDB Knowledge Graph Setup
# =============================================================================

def load_ddb():
    """
    Carica i nomi e le relazioni dal database DDB.
    
    Returns:
        tuple: (relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name)
    """
    with open("data/ddb/ddb_names.json") as f:
        all_names = json.load(f)
    with open("data/ddb/ddb_relas.json") as f:
        all_relas = json.load(f)
    
    relas_lst = [val for key, val in all_relas.items()]
    
    ddb_ptr_to_preferred_name = {}
    ddb_ptr_to_name = defaultdict(list)
    ddb_name_to_ptr = {}
    
    for key, val in all_names.items():
        item_name = key
        item_ptr = val[0]
        item_preferred = val[1]
        if item_preferred == "1":
            ddb_ptr_to_preferred_name[item_ptr] = item_name
        ddb_name_to_ptr[item_name] = item_ptr
        ddb_ptr_to_name[item_ptr].append(item_name)
    
    return relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name


def construct_graph(relas_lst, id2concept, relation_handler):
    """
    Costruisce il grafo di conoscenza DDB.
    
    Args:
        relas_lst (list): Lista delle relazioni DDB.
        id2concept (list): Lista degli ID concettuali.
        relation_handler (RelationHandler): Istanze per la gestione delle relazioni.
    
    Returns:
        tuple: (concept2id, id2relation, relation2id, graph)
    """
    concept2id = {w: i for i, w in enumerate(id2concept)}
    id2relation = relation_handler.merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}
    graph = nx.MultiDiGraph()
    for relation in relas_lst:
        subj = concept2id[relation[0]]
        obj = concept2id[relation[1]]
        rel = relation_handler.relas_dict[relation[2]]
        weight = 1.0
        graph.add_edge(subj, obj, rel=rel, weight=weight)
        graph.add_edge(
            obj, subj,
            rel=rel + len(relation_handler.final_merged_relations),
            weight=weight
        )
    output_path = "./data/ddb/ddb.graph"
    pickle.dump(graph, open(output_path, "wb"))
    return concept2id, id2relation, relation2id, graph


# =============================================================================
# Preprocessing per PubMed
# =============================================================================

def save_articles_metadata(fname, articles, dataset_name):
    """
    Salva i puntatori e gli abstract degli articoli in file di testo.
    
    Args:
        fname (str): Nome dello split (es. "train").
        articles (dict): Metadati degli articoli.
        dataset_name (str): Nome del dataset ("medmcqa" o "pubmedqa").
    """
    ptr_path = f"./data/{dataset_name}/metadata/{fname}_ptrs.txt"
    vocabs_path = f"./data/{dataset_name}/metadata/{fname}_vocabs.txt"
    
    with open(ptr_path, "w") as fout:
        for doc in articles["PMID"]:
            print(doc, file=fout)
    
    with open(vocabs_path, "w", encoding="utf-8") as fout:
        for doc in tqdm(articles["Abstract"], desc=f"Saving {dataset_name} abstracts"):
            print(doc, file=fout)


def preprocess_articles():
    """
    Processa gli articoli per MedMCQA e PubMedQA.
    
    Returns:
        tuple: (md_id2concept_medmcqa, md_id2concept_pubmedqa)
    """
    articles_medmcqa = treat_topk_articles_as_unique_articles(FNAME, db_w_metadata_medcqa, retr_info_ds_medcqua)
    articles_pubmedqa = treat_topk_articles_as_unique_articles(FNAME, db_w_metadata_pubmedqa, retr_info_ds_pubmedqa)
    
    save_articles_metadata(FNAME, articles_medmcqa, "medmcqa")
    save_articles_metadata(FNAME, articles_pubmedqa, "pubmedqa")
    
    md_id2concept_medmcqa = get_article_pointers(f"./data/medmcqa/metadata/{FNAME}_ptrs.txt")
    md_id2concept_pubmedqa = get_article_pointers(f"./data/pubmedqa/metadata/{FNAME}_ptrs.txt")
    return md_id2concept_medmcqa, md_id2concept_pubmedqa


def create_and_save_relations(fname, db_metadata, retr_info_ds, dataset_name, relation_handler):
    """
    Crea e salva le relazioni sui metadati per un dataset.
    
    Args:
        fname (str): Nome dello split (es. "train").
        db_metadata (dict): Metadati degli articoli.
        retr_info_ds (dict): Informazioni di retrieval.
        dataset_name (str): Nome del dataset ("medmcqa" o "pubmedqa").
        relation_handler (RelationHandler): Istanze per la gestione delle relazioni.
    """
    relations = relation_handler.create_metadata_relations(fname, db_metadata, retr_info_ds)
    original_len = len(relations)
    relations = remove_redundant_lists(relations)
    removed = original_len - len(relations)
    print(f"relations_{dataset_name}_{fname}: {original_len} totali, {removed} duplicati rimossi, {len(relations)} finali")
    out_path = f"./data/{dataset_name}/metadata/{fname}_metadata_relations.json"
    with open(out_path, "w") as fout:
        json.dump(relations, fout, ensure_ascii=False)


def construct_metadata_graph(md_id2concept, md_relas_lst, output_path, relation_handler):
    """
    Costruisce il grafo dei metadati per le relazioni PubMed.
    
    Args:
        md_id2concept (list): Lista di puntatori agli articoli.
        md_relas_lst (list): Lista delle relazioni di metadati.
        output_path (str): Percorso per salvare il grafo.
        relation_handler (RelationHandler): Istanze per la gestione delle relazioni.
    
    Returns:
        tuple: (concept2id, id2relation, relation2id, graph)
    """
    # Gli ID dei concetti per il grafo dei metadati iniziano dopo quelli DDB
    concept2id = {w: i + len(id2concept) for i, w in enumerate(md_id2concept)}
    id2relation = relation_handler.metadata_relations
    relation2id = {w: i + len(relation_handler.merged_relations) for i, w in enumerate(id2relation)}
    graph = nx.MultiDiGraph()
    
    for relation in tqdm(md_relas_lst, desc="Constructing metadata graph"):
        try:
            subj = concept2id[str(relation[0])]
            obj = concept2id[str(relation[1])]
            rel = relation[2]
            weight = 1.0
            graph.add_edge(subj, obj, rel=rel, weight=weight)
            graph.add_edge(obj, subj, rel=rel + len(relation_handler.final_merged_relations), weight=weight)
        except Exception as e:
            print(f"Error processing relation {relation}: {e}")
            traceback.print_exc()
    pickle.dump(graph, open(output_path, "wb"))
    return concept2id, id2relation, relation2id, graph


# =============================================================================
# Esecuzione Principale
# =============================================================================

if __name__ == "__main__":
    # Inizializza il RelationHandler
    relation_handler = RelationHandler()

    # DDB Knowledge Graph Setup
    relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name = load_ddb()
    ddb_ptr_lst = list(ddb_ptr_to_preferred_name.keys())
    id2concept = ddb_ptr_lst
    concept2id, id2relation, relation2id, KG = construct_graph(relas_lst, id2concept, relation_handler)

    # Preprocessing degli articoli e salvataggio dei metadati
    md_id2concept_medmcqa, md_id2concept_pubmedqa = preprocess_articles()

    # Creazione e salvataggio delle relazioni per MedMCQA e PubMedQA
    for fname in FNAMES:
        create_and_save_relations(fname, db_w_metadata_medcqa, retr_info_ds_medcqua, "medmcqa", relation_handler)
        create_and_save_relations(fname, db_w_metadata_pubmedqa, retr_info_ds_pubmedqa, "pubmedqa", relation_handler)

    # Caricamento delle relazioni per la costruzione del grafo dei metadati
    md_relas_lst_medmcqa = load_metadata_relations(f"./data/medmcqa/metadata/{FNAME}_metadata_relations.json")
    md_relas_lst_pubmedqa = load_metadata_relations(f"./data/pubmedqa/metadata/{FNAME}_metadata_relations.json")

    # Costruzione dei grafi dei metadati per MedMCQA e PubMedQA
    md_concept2id_medmcqa, md_id2relation_medmcqa, md_relation2id_medmcqa, md_KG_medmcqa = construct_metadata_graph(
        md_id2concept_medmcqa, md_relas_lst_medmcqa, "./data/medmcqa/metadata/metadata.graph", relation_handler
    )
    md_concept2id_pubmedqa, md_id2relation_pubmedqa, md_relation2id_pubmedqa, md_KG_pubmedqa = construct_metadata_graph(
        md_id2concept_pubmedqa, md_relas_lst_pubmedqa, "./data/pubmedqa/metadata/metadata.graph", relation_handler
    )
