from .newsgroups20 import Newsgroups20_Dataset
from .agnews import AgNews_Dataset
from .rncp import Rncp_Dataset


# Modification: adding cont and useBow arguments
def load_dataset(dataset_name, data_path, normal_class, tokenizer='spacy', use_tfidf_weights=False,
                 append_sos=False, append_eos=False, clean_txt=False, cont=None,useBow=False):
    """Loads the dataset."""

    # Modification: adding agnews and rncp datasets
    implemented_datasets = ('agnews', 'newsgroups20','rncp')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'agnews':
        dataset = AgNews_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                 use_tfidf_weights=use_tfidf_weights, append_sos=append_sos,
                                 append_eos=append_eos, clean_txt=clean_txt, cont=cont,useBow=useBow)
    
    if dataset_name == 'rncp':
        dataset = Rncp_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                 use_tfidf_weights=use_tfidf_weights, append_sos=append_sos,
                                 append_eos=append_eos, clean_txt=clean_txt, cont=cont,useBow=useBow)

    if dataset_name == 'newsgroups20':
        dataset = Newsgroups20_Dataset(root=data_path, normal_class=normal_class, tokenizer=tokenizer,
                                       use_tfidf_weights=use_tfidf_weights, append_sos=append_sos,
                                       append_eos=append_eos, clean_txt=clean_txt, cont=cont,useBow=useBow)

    return dataset
