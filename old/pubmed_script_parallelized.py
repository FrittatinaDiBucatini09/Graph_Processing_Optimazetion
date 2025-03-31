#!/usr/bin/env python3
"""
Graph Preprocessing Script
---------------------------
Questo script carica i dataset MedMCQA e PubMedQA, esegue il preprocessing
dei metadati, costruisce grafi di conoscenza a partire da relazioni definite in un database (DDB)
e salva i risultati su disco.

Autore: :)
Data: :(
"""

import os
import sys
import json
import pickle
import traceback
import time

import numpy as np
import networkx as nx

from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import login
from itertools import chain
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_NUM_THREADS = 4

# =============================================================================
# Configurazione e Caricamento Datasets
# =============================================================================

REPO_ID_MEDCQA = "MGFiD-Group/MedMCQA_utils"
REPO_ID_PUBMEDQA = "MGFiD-Group/PubMedQA_utils"
METADATANAME = "database_w_metadata"
RETRIEVALINFONAME = "retrieval_infos"

FNAMES = ["train"]
FNAME = "train"

db_w_metadata_medcqa = load_from_disk("./data/hf/db_w_metadata_medcqa.hf")
retr_info_ds_medcqua = load_from_disk("./data/hf/retr_info_ds_medcqua.hf")
db_w_metadata_pubmedqa = load_from_disk("./data/hf/db_w_metadata_pubmedqa.hf")
retr_info_ds_pubmedqa = load_from_disk("./data/hf/retr_info_ds_pubmedqa.hf")

start = time.time()

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
        raise ValueError("All lists must be of the same length")
    return [dict(zip(keys, values)) for values in zip(*lists)]


def turn_a_list_of_lists_into_a_set(lst):
    """
    Appiattisce una lista di liste in un insieme di elementi unici.
    """
    flat_list = list(chain.from_iterable(lst))
    unique_set = set(flat_list)
    print(f"old elements: {len(flat_list)}, new elements: {len(unique_set)}")
    return unique_set


def treat_topk_articles_as_unique_articles(fname, db_metadata, retr_info_ds):
    """
    Estrae articoli unici basandosi sugli indici ottenuti dal dataset di retrieval.
    """
    articles_indexes = retr_info_ds[fname]["retrieval_relative_indices"]
    unique_indexes = turn_a_list_of_lists_into_a_set(articles_indexes)
    unique_indexes = list(filter(lambda x: x != -1, unique_indexes))
    return db_metadata[fname][unique_indexes]


def remove_newlines_from_list(input_list):
    """
    Rimuove eventuali caratteri di newline da ogni stringa in una lista.
    """
    return [s.strip() for s in input_list]


def get_article_pointers(path):
    """
    Legge i puntatori agli articoli da un file.
    """
    with open(path) as fin:
        lines = fin.readlines()
    return [line.strip() for line in lines]


def remove_redundant_lists(lst):
    """
    Rimuove liste duplicate da una lista di liste.
    """
    return [list(rel) for rel in set(tuple(rel) for rel in lst)]


def load_metadata_relations(path):
    """
    Carica le relazioni dei metadati da un file JSON.
    """
    with open(path) as f:
        all_relas = json.load(f)
    return list(all_relas.values()) if isinstance(all_relas, dict) else all_relas


# =============================================================================
# Classe RelationHandler
# =============================================================================

class RelationHandler:
    """
    Gestisce la mappatura e la creazione delle relazioni tra articoli.
    """
    def __init__(self):
        # Relazioni "cliniche" provenienti da DDB
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
        # Relazioni "metadati" (autori, anno, rivista, ecc.)
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


# =============================================================================
# Funzioni parallele per creare relazioni metadati
# =============================================================================

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
# DDB Knowledge Graph Setup
# =============================================================================

def load_ddb():
    """
    Carica i nomi e le relazioni dal database DDB.
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
    Processa gli articoli per MedMCQA e PubMedQA e salva i metadati.
    """
    articles_medmcqa = treat_topk_articles_as_unique_articles(FNAME, db_w_metadata_medcqa, retr_info_ds_medcqua)
    articles_pubmedqa = treat_topk_articles_as_unique_articles(FNAME, db_w_metadata_pubmedqa, retr_info_ds_pubmedqa)
    
    save_articles_metadata(FNAME, articles_medmcqa, "medmcqa")
    save_articles_metadata(FNAME, articles_pubmedqa, "pubmedqa")
    
    md_id2concept_medmcqa = get_article_pointers(f"./data/medmcqa/metadata/{FNAME}_ptrs.txt")
    md_id2concept_pubmedqa = get_article_pointers(f"./data/pubmedqa/metadata/{FNAME}_ptrs.txt")
    return md_id2concept_medmcqa, md_id2concept_pubmedqa


def construct_metadata_graph(md_id2concept, md_relas_lst, output_path, relation_handler, global_id2concept):
    """
    Costruisce il grafo dei metadati per le relazioni PubMed/MedMCQA.
    """
    # Gli ID dei concetti per il grafo dei metadati iniziano dopo quelli DDB
    concept2id = {w: i + len(global_id2concept) for i, w in enumerate(md_id2concept)}
    id2relation = relation_handler.metadata_relations
    relation2id = {
        w: i + len(relation_handler.merged_relations)
        for i, w in enumerate(id2relation)
    }
    
    graph = nx.MultiDiGraph()
    for relation in tqdm(md_relas_lst, desc=f"Constructing metadata graph {output_path}"):
        try:
            subj = concept2id[str(relation[0])]
            obj = concept2id[str(relation[1])]
        except KeyError:
            # Salta la relazione se l'ID non è nel dizionario
            continue
        rel = relation[2]
        weight = 1.0
        graph.add_edge(subj, obj, rel=rel, weight=weight)
        graph.add_edge(obj, subj, rel=rel + len(relation_handler.final_merged_relations), weight=weight)

    pickle.dump(graph, open(output_path, "wb"))
    return concept2id, id2relation, relation2id, graph


# =============================================================================
# Esecuzione Principale
# =============================================================================

if __name__ == "__main__":
    # Inizializza il RelationHandler
    relation_handler = RelationHandler()
    
    # Determina il numero di thread da riga di comando (o default)
    if len(sys.argv) < 2:
        num_threads = DEFAULT_NUM_THREADS
    elif int(sys.argv[1]) > 0:
        num_threads = int(sys.argv[1])
    else:
        num_threads = DEFAULT_NUM_THREADS

    # DDB Knowledge Graph Setup
    relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name = load_ddb()
    ddb_ptr_lst = list(ddb_ptr_to_preferred_name.keys())
    id2concept = ddb_ptr_lst  # concetti (PTR) del DB DDB

    concept2id, id2relation, relation2id, KG = construct_graph(relas_lst, id2concept, relation_handler)

    # Preprocessing articoli e salvataggio metadata
    md_id2concept_medmcqa, md_id2concept_pubmedqa = preprocess_articles()

    # Creazione e salvataggio relazioni metadati (MedMCQA, PubMedQA)
    for fname in FNAMES:
        # Relazioni per MedMCQA
        relations_medmcqa = create_metadata_relations_parallel(
            fname, db_w_metadata_medcqa, retr_info_ds_medcqua, relation_handler, num_threads
        )
        original_len = len(relations_medmcqa)
        relations_medmcqa = remove_redundant_lists(relations_medmcqa)
        removed = original_len - len(relations_medmcqa)
        print(f"relations_medmcqa_{fname}: {original_len} totali, {removed} duplicati rimossi, {len(relations_medmcqa)} finali")
        out_path = f"./data/medmcqa/metadata/{fname}_metadata_relations.json"
        with open(out_path, "w") as fout:
            json.dump(relations_medmcqa, fout, ensure_ascii=False)

        # Relazioni per PubMedQA
        relations_pubmedqa = create_metadata_relations_parallel(
            fname, db_w_metadata_pubmedqa, retr_info_ds_pubmedqa, relation_handler, num_threads
        )
        original_len = len(relations_pubmedqa)
        relations_pubmedqa = remove_redundant_lists(relations_pubmedqa)
        removed = original_len - len(relations_pubmedqa)
        print(f"relations_pubmedqa_{fname}: {original_len} totali, {removed} duplicati rimossi, {len(relations_pubmedqa)} finali")
        out_path = f"./data/pubmedqa/metadata/{fname}_metadata_relations.json"
        with open(out_path, "w") as fout:
            json.dump(relations_pubmedqa, fout, ensure_ascii=False)

    # Carica le relazioni per costruire i grafi metadati
    md_relas_lst_medmcqa = load_metadata_relations(f"./data/medmcqa/metadata/{FNAME}_metadata_relations.json")
    md_relas_lst_pubmedqa = load_metadata_relations(f"./data/pubmedqa/metadata/{FNAME}_metadata_relations.json")

    # Costruzione dei grafi dei metadati
    md_concept2id_medmcqa, md_id2relation_medmcqa, md_relation2id_medmcqa, md_KG_medmcqa = construct_metadata_graph(
        md_id2concept_medmcqa,
        md_relas_lst_medmcqa,
        "./data/medmcqa/metadata/metadata.graph",
        relation_handler,
        id2concept  # passiamo i concetti DDB per calcolare l'offset
    )
    md_concept2id_pubmedqa, md_id2relation_pubmedqa, md_relation2id_pubmedqa, md_KG_pubmedqa = construct_metadata_graph(
        md_id2concept_pubmedqa,
        md_relas_lst_pubmedqa,
        "./data/pubmedqa/metadata/metadata.graph",
        relation_handler,
        id2concept
    )

    end = time.time()
    print(f"Tempo totale: {end - start:.2f} secondi")
