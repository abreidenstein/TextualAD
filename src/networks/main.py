from .cvdd_Net import CVDDNet
from .bert import BERT
# Modification: Importing fasttext
from gensim.models.fasttext import FastText
from base.embedding import MyEmbedding
from utils.word_vectors import load_word_vectors
# Modification: Importing pickle
from six.moves import cPickle as pickle
# Modification: Importing torch
import torch
from transformers import ElectraModel

# Modification: adding cont argument
def build_network(net_name, dataset, embedding_size=None, pretrained_model=None, update_embedding=True,
                  embedding_reduction='none', use_tfidf_weights=False, normalize_embedding=False,
                  word_vectors_cache='../data/word_vectors_cache', attention_size=100, n_attention_heads=1, cont=None):
    """Builds the neural network."""

    implemented_networks = ('embedding', 'cvdd_Net')
    assert net_name in implemented_networks

    net = None
    vocab_size = dataset.encoder.vocab_size
    # Set embedding

    # Load pre-trained model if specified
    if pretrained_model is not None:
        # if word vector model
        if pretrained_model in ['GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en', 'FastText_fr']:
            word_vectors, embedding_size = load_word_vectors(pretrained_model, embedding_size, word_vectors_cache)
            embedding = MyEmbedding(vocab_size, embedding_size, update_embedding, embedding_reduction,
                                    use_tfidf_weights, normalize_embedding)
            # Init embedding with pre-trained word vectors

            for i, token in enumerate(dataset.encoder.vocab):
                embedding.weight.data[i] = word_vectors[token]

        # if language model
        if pretrained_model in ['bert']:
            embedding = BERT()
        # to be able to use a pre-trained electra model
        if pretrained_model in ['Electra_pretrained']:
            model = ElectraModel.from_pretrained("google/electra-small-discriminator",cache_dir='../data/electra_cache',output_hidden_states=False)
            model.to('cuda')
            def embedding(x):
                x = torch.transpose(x,0,1)
                return(torch.mean(model(x)[0], dim=1))# take the mean of the vectors' representations

            

        """ Modification: Importing other embeddings """
        # If other embeddings
        if pretrained_model in ['FastText_dataset', 'FastText_class', 'PPMI_dataset', 'PPMI_class', 'DATE_dataset','DATE_class','DATE_class_noRMD','DATE_dataset_noRMD']:
            embedding = MyEmbedding(vocab_size, embedding_size, update_embedding, embedding_reduction,
                                    use_tfidf_weights, normalize_embedding)
            # Getting the file with pretrained word vectors
            pretrained_file = '../data/word_vectors_saved'
            if dataset.dataset_name == 'agnews':
                pretrained_file += '/ag_od/train/'
                class_list = ['business', 'sci', 'sports', 'world']
            if dataset.dataset_name == 'newsgroup20':
                pretrained_file += '/20ng_od/train/'
                class_list = ['comp', 'misc', 'pol', 'rec', 'rel', 'sci']
            if dataset.dataset_name == 'rncp':
                pretrained_file += '/rncp_od/train/'
                class_list = ['1-environnement','2-defense','3-patrimoine','4-economie',
                        '5-recherche','6-nautisme','7-aeronautique','8-securite',
                        '9-multimedia','10-humanitaire','11-nucleaire','12-enfance',
                        '13-saisonnier','14-assistance','15-sport','16-ingenierie']

            if pretrained_model.startswith('FastText'):    
                if pretrained_model == 'FastText_dataset':
                    pretrained_file += pretrained_model
                if pretrained_model == 'FastText_class':
                    if cont is None: cont=0
                    if cont is not None:
                        pretrained_file += str(cont) + '/'
                    pretrained_file += 'FastText_'
                    pretrained_file += class_list[dataset.normal_class]

                # Create a fasttext model and initialize if with the pre-trained word vectors
                fastext_model = FastText.load(pretrained_file)
                for i, token in enumerate(dataset.encoder.vocab):
                    embedding.weight.data[i] = torch.FloatTensor(fastext_model.wv[token])

            if pretrained_model.startswith('PPMI'):
                if pretrained_model == 'PPMI_dataset':
                    pretrained_file += pretrained_model
                if pretrained_model == 'PPMI_class':
                    if cont is None: cont=0
                    if cont is not None:
                        pretrained_file += str(cont) + '/'
                    pretrained_file += 'PPMI_'
                    pretrained_file += class_list[dataset.normal_class]
                with open(pretrained_file, 'rb') as f:
                    external_voc = pickle.load(f)
                    word_vectors = pickle.load(f)
                unk = len(external_voc)
                    
                # Init embedding with pre-trained word vectors from PPMI
                for i, token in enumerate(dataset.encoder.vocab):
                    embedding.weight.data[i] = torch.FloatTensor(word_vectors[external_voc.get(token, unk - 1)])

            if pretrained_model.startswith('DATE'):
                # no word vectors, but contextual vectors computed for each document directly and saved before running OCSVM with Date representations
                embedding = pretrained_model
                

    else:
        if embedding_size is not None:
            embedding = MyEmbedding(vocab_size, embedding_size, update_embedding, embedding_reduction,
                                    use_tfidf_weights, normalize_embedding)
        else:
            raise Exception('If pretrained_model is None, embedding_size must be specified')

    # Load network
    if net_name == 'embedding':
        net = embedding
    if net_name == 'cvdd_Net':
        net = CVDDNet(embedding, attention_size=attention_size, n_attention_heads=n_attention_heads)

    return net
