#!/usr/bin/env python3
"""
Single Dataset Graph Preprocessing
-----------------------------------
This script requires three command-line parameters:
1) number of threads,
2) path to the metadata dataset,
3) path to the retrieval dataset.

It then creates a metadata graph (based on the unique articles and relationships:
shared authors, same year, same journal, etc.), and saves it to disk.
"""

import json
import pickle
import time

import networkx as nx
from datasets import load_from_disk
from itertools import chain
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

# Name of the split on which to compute the relationships (can be customized).
FNAME = "train"


# =============================================================================
# Utility Functions
# =============================================================================

def dict_of_lists_to_list_of_dicts(dict_of_lists):
    """
    Converts a dictionary of lists into a list of dictionaries.
    For example:
        Input: {"name": ["Alice", "Bob"], "age": [30, 25]}
        Output: [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    """
    keys = list(dict_of_lists.keys())
    lists = list(dict_of_lists.values())
    length = len(lists[0])
    if not all(len(lst) == length for lst in lists):
        raise ValueError("All lists must be of the same length.")
    return [dict(zip(keys, values)) for values in zip(*lists)]


def turn_a_list_of_lists_into_a_set(lst):
    """
    Flattens a list of lists into a set of unique elements.
    Prints the difference between the total number of items vs. unique elements.
    """
    flat_list = list(chain.from_iterable(lst))
    unique_set = set(flat_list)
    print(f"Old elements: {len(flat_list)}, unique new elements: {len(unique_set)}")
    return unique_set


def treat_topk_articles_as_unique_articles(fname, db_metadata, retr_info_ds):
    """
    Extracts unique articles based on the indices obtained from the retrieval dataset.
    Filters out any -1 entries if they exist.
    """
    articles_indexes = retr_info_ds[fname]["retrieval_relative_indices"]
    unique_indexes = turn_a_list_of_lists_into_a_set(articles_indexes)
    # Remove any -1 if present
    unique_indexes = [x for x in unique_indexes if x != -1]
    return db_metadata[fname][unique_indexes]


def remove_redundant_lists(lst):
    """
    Removes duplicate lists from a list of lists by converting them into tuples
    and placing them into a set, then converting them back into lists.
    """
    return [list(rel) for rel in set(tuple(rel) for rel in lst)]


def process_relation_index(i, fname, db_metadata, retr_info_ds, relation_handler):
    """
    Processes relationships for a single index of the retrieval_relative_indices array.
    Iterates through the articles in db_metadata, comparing them pairwise to find
    matches in year, journal, authors, etc.
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


def create_metadata_relations_parallel(fname, db_metadata, retr_info_ds, relation_handler, num_workers):
    """
    Creates metadata relationships in parallel using ProcessPoolExecutor.
    Iterates over each index in retrieval_relative_indices and merges the results.
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
# RelationHandler Class
# =============================================================================

class RelationHandler:
    """
    Manages the mapping and creation of relationships between articles.
    It focuses on "metadata" relationships only (authors, year, journal, etc.).
    """

    def __init__(self):
        # "Metadata" relationships (authors, year, journal, etc.)
        self.metadata_relations = [
            "author_in_common", "mesh_term_in_common", "chemical_term_in_common",
            "same_journal", "same_year"
        ]
        # Map each relationship to an integer (ID) for building the graph
        self.metadata_relations_to_id = {
            w: i for i, w in enumerate(self.metadata_relations)
        }

    def encounters(self, list1, list2):
        return [elem for elem in list1 if elem in list2]

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
# Function to build and save the graph
# =============================================================================

def construct_metadata_graph(article_pmids, metadata_relations, output_path, relation_handler):
    """
    Builds the metadata graph from a list of relationships and saves it.
    
    :param article_pmids: list of all PMIDs (or unique IDs) present
    :param metadata_relations: calculated relationships (source, destination, relationship ID)
    :param output_path: path where the graph in pickle format will be saved
    :param relation_handler: an instance of RelationHandler to reference relationship IDs
    """
    # Map PMIDs to numeric IDs
    concept2id = {pmid: i for i, pmid in enumerate(article_pmids)}
    # Retrieve the list of relationship strings
    id2relation = relation_handler.metadata_relations
    relation2id = relation_handler.metadata_relations_to_id
    
    graph = nx.MultiDiGraph()

    # Create edges
    for (subj_pmid, obj_pmid, rel_id) in metadata_relations:
        try:
            s = concept2id[subj_pmid]
            o = concept2id[obj_pmid]
        except KeyError:
            # If for any reason one of the PMIDs is not in concept2id, skip it
            continue
        # Add edge (subj -> obj) with 'rel' attribute
        graph.add_edge(s, o, rel=rel_id, weight=1.0)
        # For a bidirectional graph, add the reverse edge as well
        graph.add_edge(o, s, rel=rel_id, weight=1.0)

    # Save to disk
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    return graph


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    # -----------------------------------------------------------
    # Reading command-line parameters
    # -----------------------------------------------------------
    
    # Check for number of threads
    while True:
        try:
            num_threads = int(input("Specify number of threads: "))
        except ValueError:
            print("The number of threads must be an integer.")
            continue
        if num_threads <= 0:
            print("The number of threads must be greater than 0.")
            continue
        break

    # Check for dataset availability
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
    # Extracting unique articles
    # -----------------------------------------------------------
    unique_articles = treat_topk_articles_as_unique_articles(FNAME, db_w_metadata, retr_info_ds)
    # Get the list of PMIDs (or unique IDs)
    pmid_list = unique_articles["PMID"]

    # -----------------------------------------------------------
    # Creating relationships (in parallel)
    # -----------------------------------------------------------
    relation_handler = RelationHandler()
    print(f"Creating metadata relations for split '{FNAME}' using {num_threads} threads...")
    relations = create_metadata_relations_parallel(FNAME, db_w_metadata, retr_info_ds, relation_handler, num_threads)

    # Removing duplicates
    orig_len = len(relations)
    relations = remove_redundant_lists(relations)
    final_len = len(relations)
    print(f"Total relationships: {orig_len}, after removing duplicates: {final_len}")

    # Saving relationships to .json (optional; useful for debugging)
    relations_json_path = "metadata_relations.json"
    with open(relations_json_path, "w") as f:
        json.dump(relations, f, ensure_ascii=False)
    print(f"Saved relationships to {relations_json_path}")

    # -----------------------------------------------------------
    # Building and saving the metadata graph
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
    print(f"\nGraph construction completed in {end_time - start_time:.2f} seconds.")
    print(f"\nGraph saved to {graph_path}")