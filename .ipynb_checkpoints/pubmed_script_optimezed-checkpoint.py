import os
import sys
import json
import pickle
import traceback

import numpy as np
import networkx as nx

from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, load_from_disk, DatasetDict
from huggingface_hub import login
from itertools import chain

####################
# Get the datasets #
####################

# Datasets' names
REPO_ID_MEDCQA = "MGFiD-Group/MedMCQA_utils"
REPO_ID_PUBMEDQA = "MGFiD-Group/PubMedQA_utils"
METADATANAME = "database_w_metadata"
RETRIEVALINFONAME = "retrieval_infos"

# Some vars
FNAMES = ["train"]
FNAME = 'train'


# The pubmed dataset
# db_w_metadata_medcqa = load_dataset(REPO_ID_MEDCQA, METADATANAME)
db_w_metadata_medcqa = load_from_disk("./data/hf/db_w_metadata_medcqa.hf")

# Custom dataset that connects the previous two.
# - ID
#   The enum index of each entry of the medqa dataset
# - retrieval_relative_indices
#   A list of PMIDs which are relevant to the question
# retr_info_ds_whole_medcqa = load_dataset(REPO_ID_MEDCQA, RETRIEVALINFONAME)
# retr_info_ds_medcqua = DatasetDict({
#     'train': retr_info_ds_whole_medcqa['train'].select(range(500)),
#     'validation': retr_info_ds_whole_medcqa['validation'].select(range(100))
# })
retr_info_ds_medcqua = load_from_disk("./data/hf/retr_info_ds_medcqua.hf")

# Same things but for pubmed_qa
# retr_info_ds_whole_medcqa will have the same structure as the one above but for the entries of pubmed_qa
# db_w_metadata_medcqa should be literally the same of the one above, but i don't want to check, so double it is
# db_w_metadata_pubmedqa = load_dataset(REPO_ID_PUBMEDQA, METADATANAME)
db_w_metadata_pubmedqa = load_from_disk("./data/hf/db_w_metadata_pubmedqa.hf")

# retr_info_ds_whole_pubmedqa = load_dataset(REPO_ID_PUBMEDQA, RETRIEVALINFONAME)
# retr_info_ds_pubmedqa = DatasetDict({
#     'train': retr_info_ds_whole_pubmedqa['train'].select(range(500)),
# })
retr_info_ds_pubmedqa = load_from_disk("./data/hf/retr_info_ds_pubmedqa.hf")



###########################
# General helpful methods #
###########################

def dict_of_lists_to_list_of_dicts(dict_of_lists):
    '''
    Turn a dict of lists into a list of dicts
    e.g.:
    dict_of_lists = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [30, 25, 35],
        "city": ["New York", "Los Angeles", "Chicago"]
    }
    to
    [{'name': 'Alice', 'age': 30, 'city': 'New York'},
    {'name': 'Bob', 'age': 25, 'city': 'Los Angeles'},
    {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}]

    This is to handle what comes out from treat_topk_articles_as_unique_articles
    which has this format:
    {'PMID': [32665527, 8599280, 12822472, 30647141],
    'Year': ['2021', '1996', '2003', '2019'],
    '''
    # Get all keys and their corresponding lists
    keys = list(dict_of_lists.keys())
    lists = list(dict_of_lists.values())
    # Check if all lists are of the same length
    length = len(lists[0])
    if not all(len(lst) == length for lst in lists):
        raise ValueError("All lists must be of the same length")

    # Use zip to group elements and create the list of dictionaries
    return [dict(zip(keys, values)) for values in zip(*lists)]


def turn_a_list_of_lists_into_a_set(lst):
    '''
    Transforms a list of lists into as single chonky set, thus obtain a single set
    without duplicates.
    e.g.:
    [[1,3],[4,3],[44,1]]
    old elements: 6, new elements: 4
    {1, 3, 4, 44}
    '''
    new_lst = list(chain.from_iterable(lst))
    unique_lst = set(new_lst)
    print(f"old elements: {len(new_lst)}, new elements: {len(unique_lst)}")
    return unique_lst


# Function to treat top K articles as unique articles
def treat_topk_articles_as_unique_articles(fname, db_metadata, retr_info_ds):
    # Get the list of article indexes for each question
    articles_indexes = retr_info_ds[fname]['retrieval_relative_indices']
    # Remove duplicate articles
    unique_articles_indexes = turn_a_list_of_lists_into_a_set(articles_indexes)
    # Filter out invalid article indexes (-1)
    unique_articles_indexes = list(filter(lambda x: x != -1, unique_articles_indexes))
    # Return the unique articles from the metadata dataset
    return db_metadata[fname][unique_articles_indexes]


def remove_newlines_from_list(input_list):
    return [s.strip() for s in input_list]


def get_article_pointers(path):
    md_id2concept = []
    with open(path) as fin:
        lines = fin.readlines()
        md_id2concept = [l for l in remove_newlines_from_list(lines)]
    return md_id2concept


def remove_redundant_lists(lst):
    # Converte direttamente le liste in un set di tuple
    return [list(rel) for rel in set(tuple(rel) for rel in lst)]

def load_metadata_relations(path):
    with open(path) as f:
        all_relas = json.load(f)
    relas_lst = []
    for val in all_relas:
        relas_lst.append(val)

    return relas_lst


class RelationHandler():
    def __init__(self):
        '''
        Create the mapping of the relaions in the ddb dataset having the dict relas_dict being:
        - Key => The id of the relation in the original dbb dataset
        - value => The index of the entry to which it belongs to in the `merged_relations` list

        Then, extend it with the relation the pubmed dataset will further provide
        '''
        self.merged_relations = [
            'belongs_to_the_category_of',
            'is_a_category',
            'may_cause',
            'is_a_subtype_of',
            'is_a_risk_factor_of',
            'is_associated_with',
            'may_contraindicate',
            'interacts_with',
            'belongs_to_the_drug_family_of',
            'belongs_to_drug_super-family',
            'is_a_vector_for',
            'may_be_allelic_with',
            'see_also',
            'is_an_ingradient_of',
            'may_treat'
        ]

        self.relas_dict = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 5, "10": 6, "12": 7, "16": 8, "17": 9, "18": 10, "20": 11, "26": 12, "30": 13, "233": 14}

        self.metadata_relations = [
            'author_in_common',
            'mesh_term_in_common',
            'chemical_term_in_common',
            'same_journal',
            'same_year',
            'name_entities_with_abstracts_relation',
            'god_document_dummy_node_connections',
            'qa_context_nodes',
        ]
        
        self.metadata_relations_to_id = {w : i+len(self.merged_relations) for i, w in enumerate(self.metadata_relations)}
        self.final_merged_relations = self.merged_relations + self.metadata_relations

    def encounters(self, list1, list2):
        return [l for l in list1 if l in list2]

    def at_least_N_things_in_common(self, list1, list2, n):
        if list1 is None:
            list1 = []
        if list2 is None:
            list2 = []
        e = self.encounters(list1, list2)
        return len(e) >= n

    def same_author_1_in_common(self, authors1, authors2):
        return self.at_least_N_things_in_common(authors1, authors2, 1)

    def same_chemical_term_1_in_common(self, chemicallist1, chemicallist2):
        return self.at_least_N_things_in_common(chemicallist1, chemicallist2, 1)

    def same_mesh_term_2_in_common(self, meshtermlist1, meshtermlist2):
        return self.at_least_N_things_in_common(meshtermlist1, meshtermlist2, 2)

    def same_mesh_term_3_in_common(self, meshtermlist1, meshtermlist2):
        return self.at_least_N_things_in_common(meshtermlist1, meshtermlist2, 3)

    def same_conference(self, conf1, conf2):
        return conf1 == conf2

    def same_year(self, year1, year2):
        return year1==year2
    
    def create_metadata_relations(self, fname, db_metadata, retr_info_ds):
        json_data = []
        for i in tqdm(range(len(retr_info_ds[fname]['retrieval_relative_indices']))):
            articles = db_metadata[fname][retr_info_ds[fname]['retrieval_relative_indices'][i]]
            articles = dict_of_lists_to_list_of_dicts(articles)

            for doc in articles:
                for other_doc in articles:
                    # Get the PMIDs and metadata
                    doc_pmid = doc['PMID']
                    other_doc_pmid = other_doc['PMID']
                    doc_md = doc['Metadata']
                    other_doc_md = other_doc['Metadata']

                    # check the relations
                    if doc_pmid != other_doc_pmid:
                        if self.same_year(doc['Year'], other_doc['Year']):
                            json_data.append([doc_pmid, other_doc_pmid, self.metadata_relations_to_id['same_year']])

                        if self.same_conference(doc['JournalName'], other_doc['JournalName']):
                            json_data.append([doc_pmid, other_doc_pmid, self.metadata_relations_to_id['same_journal']])

                        if self.same_author_1_in_common(doc_md['AuthorList'], other_doc_md['AuthorList']):
                            json_data.append([doc_pmid, other_doc_pmid, self.metadata_relations_to_id['author_in_common']])

                        if self.same_chemical_term_1_in_common(doc_md['ChemicalTerms'], other_doc_md['ChemicalTerms']):
                            json_data.append([doc_pmid, other_doc_pmid, self.metadata_relations_to_id['chemical_term_in_common']])

                        if self.same_mesh_term_3_in_common(doc_md['MeshTerms'], other_doc_md['MeshTerms']):
                            json_data.append([doc_pmid, other_doc_pmid, self.metadata_relations_to_id['mesh_term_in_common']])
                            
        return json_data
    
relation_handler =  RelationHandler()



################
# DDB KG Setup #
################

def load_ddb():
    '''
    Opne the json ddb_names and get all its entries
    They have this format
    "Allylbarbital": ["30938", "0"]
    - Key => the ddb item name
    - val[0] => the ddb ID of the item
    - val[1] => if the item key is the preferred name (1) or not (0) for the entry

    Open the json ddb_relas and get all its entries
    They have this format
    "39": ["9836", "16", "2"]
    - Key => general index of each entry
    - val[0] => Source DDB ID
    - val[1] => Target DDB ID
    - val[2] => Relation type

    The method returns :
    - relas_lst => the list of all the ddb relations
    - ddb_ptr_to_name => a dict in the form key:ItemName, value:ItemID(the pointer)
    - ddb_name_to_ptr => a dict in the form key:ItemID(the pointer), value:ItemName
    - ddb_ptr_to_preferred_name => a dict in the form key:ItemName,
                                value:ItemID(the pointer); this only if the
                                val[1] of the ddb_names is 1
    '''
    with open(f'data/ddb/ddb_names.json') as f:
        all_names = json.load(f)
    with open(f'data/ddb/ddb_relas.json') as f:
        all_relas = json.load(f)
    relas_lst = []
    for key, val in all_relas.items():
        relas_lst.append(val)

    # ddb pointer to preffered name
    ddb_ptr_to_preferred_name = {}
    # ddb pointer to name
    ddb_ptr_to_name = defaultdict(list)
    # ddb name to pointer
    ddb_name_to_ptr = {}

    for key, val in all_names.items():
        item_name = key
        item_ptr = val[0]
        item_preferred = val[1]
        if item_preferred == "1":
            ddb_ptr_to_preferred_name[item_ptr] = item_name
        ddb_name_to_ptr[item_name] = item_ptr
        ddb_ptr_to_name[item_ptr].append(item_name)

    return (relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name)

relas_lst, ddb_ptr_to_name, ddb_name_to_ptr, ddb_ptr_to_preferred_name = load_ddb()


ddb_ptr_lst, ddb_names_lst = [], []
for key, val in ddb_ptr_to_preferred_name.items():
    ddb_ptr_lst.append(key)
    ddb_names_lst.append(val)

id2concept = ddb_ptr_lst


def construct_graph():
    '''
    Get:
    - concept2id => the dict ItemID:Index
    - id2relation => the list of relations
    - relation2id => the dict relation:Index

    Then, for each relation in ddb get the subj (ID of the source item), obj (ID of
    the target item) and the rel (the string explaining the relationship)
    Finally, proceede to create the graph
    '''
    concept2id = {w: i for i, w in enumerate(id2concept)}
    id2relation = relation_handler.merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}
    graph = nx.MultiDiGraph()
    attrs = set()
    for relation in relas_lst:
        subj = concept2id[relation[0]]
        obj = concept2id[relation[1]]
        rel = relation_handler.relas_dict[relation[2]]
        weight = 1.
        graph.add_edge(subj, obj, rel=rel, weight=weight)
        attrs.add((subj, obj, rel))
        graph.add_edge(obj, subj, rel=rel + len(relation_handler.final_merged_relations), weight=weight)
        attrs.add((obj, subj, rel + len(relation_handler.final_merged_relations)))
    output_path = f"./data/ddb/ddb.graph"
    #nx.write_gpickle(graph, output_path) deprecated function
    pickle.dump(graph, open(output_path, 'wb'))
    return concept2id, id2relation, relation2id, graph

concept2id, id2relation, relation2id, KG = construct_graph()



####################################
# Sort of preprocessing for PUBMED #
####################################

for fname in FNAMES:
    # For medmcqa
    articles_medmcqa = treat_topk_articles_as_unique_articles(fname, db_w_metadata_medcqa, retr_info_ds_medcqua)
    # For pubmedqa
    articles_pubmedqa = treat_topk_articles_as_unique_articles(fname, db_w_metadata_pubmedqa, retr_info_ds_pubmedqa)

    # Rimuovi tqdm nei loop per scrivere articoli se l'operazione è rapida
    with open(f"./data/medmcqa/metadata/{fname}_ptrs.txt", "w") as fout:
        for doc in articles_medmcqa['PMID']:
            print(doc, file=fout)


    with open(f"./data/medmcqa/metadata/{fname}_vocabs.txt", "w", encoding='utf-8') as fout:
        for doc in tqdm(articles_medmcqa['Abstract']):
            print(doc, file=fout)

    with open(f"./data/pubmedqa/metadata/{fname}_ptrs.txt", "w") as fout:
        for doc in tqdm(articles_pubmedqa['PMID']):
            print (doc, file=fout)

    with open(f"./data/pubmedqa/metadata/{fname}_vocabs.txt", "w", encoding='utf-8') as fout:
        for doc in tqdm(articles_pubmedqa['Abstract']):
            print(doc, file=fout)

    

# - md_concept2id
#   A long ass dict with all the PMIDs that were present in "{fname}_ptrs.txt"`
#   More specifically it will have entries in the format "PMID": enum_index
#   e.g.: 12345: 2 will mean that the PMID=12345 will be the second entry
md_id2concept_medmcqa = get_article_pointers(f"./data/medmcqa/metadata/{FNAME}_ptrs.txt")
md_id2concept_pubmedqa = get_article_pointers(f"./data/pubmedqa/metadata/{FNAME}_ptrs.txt")


##################################
# Construct the PUBMED relations #
##################################

# For each split
for fname in FNAMES:
    # Create the realtion for medcqa
    relations_medmcqa = relation_handler.create_metadata_relations(fname, db_w_metadata_medcqa, retr_info_ds_medcqua)

    # Remove redundancy 
    old_len = len(relations_medmcqa)
    k = remove_redundant_lists(relations_medmcqa)
    removed_files = old_len - len(k)
    print(f"relations_medmcqa_{fname} had {old_len} relations, removed {removed_files} relations, remain {len(k)} relations")

    with open(f"./data/medmcqa/metadata/{fname}_metadata_relations.json", "w") as fout:
        json.dump(k, fout, ensure_ascii=False)


    # Create the realtion for pubmedqa
    relations_pubmedqa = relation_handler.create_metadata_relations(fname, db_w_metadata_pubmedqa, retr_info_ds_pubmedqa)

    # Remove redundancy 
    old_len = len(relations_pubmedqa)
    k = remove_redundant_lists(relations_pubmedqa)
    removed_files = old_len - len(k)
    print(f"relations_pubmedqa_{fname} had {old_len} relations, removed {removed_files} relations, remain {len(k)} relations")

    with open(f"./data/pubmedqa/metadata/{fname}_metadata_relations.json", "w") as fout:
        json.dump(k, fout, ensure_ascii=False)

# Load the relations of the split FNAME
md_relas_lst_medmcqa = load_metadata_relations(f"./data/medmcqa/metadata/{FNAME}_metadata_relations.json")
md_relas_lst_pubmedqa = load_metadata_relations(f"./data/pubmedqa/metadata/{FNAME}_metadata_relations.json")



##############################
# Construct the PUBMED graph #
##############################

# MEDMCQA
def construct_metadata_graph(md_id2concept, md_relas_lst, output_path):
    '''
    Get:
    - concept2id => the dict PMID:Index(it is i+len(id2concept) so like after the
                    previous ddb entries)
    - id2relation => the list of relations
    - relation2id => the dict relation:Index

    Then, for each relation that we got between the pubmed articles get the subj
    (PMID of the source article), obj (PMID of the target article) and the rel (the
    string explaining the relationship)
    Finally, proceede to create the graph
    '''
    concept2id = {w: i+len(id2concept) for i, w in enumerate(md_id2concept)}
    id2relation = relation_handler.metadata_relations
    relation2id = {w : i+len(relation_handler.merged_relations) for i, w in enumerate(id2relation)}
    graph = nx.MultiDiGraph()
    attrs = set()

    for relation in tqdm(md_relas_lst):

        try: 
            subj = concept2id[str(relation[0])]
            obj = concept2id[str(relation[1])]
            rel = relation[2]
        
            weight = 1.
            graph.add_edge(subj, obj, rel=rel, weight=weight)
            attrs.add((subj, obj, rel))
            graph.add_edge(obj, subj, rel=rel + len(relation_handler.final_merged_relations), weight=weight)
            attrs.add((obj, subj, rel + len(relation_handler.final_merged_relations)))
        except Exception as e:
            print(f"Error processing relation {relation}: {e}")
            traceback.print_exc()


    pickle.dump(graph, open(output_path, 'wb'))
    return concept2id, id2relation, relation2id, graph

# MEDMCQA
md_concept2id_medmcqa, md_id2relation_medmcqa, md_relation2id_medmcqa, md_KG_medmcqa = construct_metadata_graph(md_id2concept_medmcqa, md_relas_lst_medmcqa, "./data/medmcqa/metadata/metadata.graph")
# PUBMEDQA
md_concept2id_pubmedqa, md_id2relation_pubmedqa, md_relation2id_pubmedqa, md_KG_pubmedqa = construct_metadata_graph(md_id2concept_pubmedqa, md_relas_lst_pubmedqa, "./data/pubmedqa/metadata/metadata.graph")
