import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.misc import print_text_samples,print_text_samples_with_scores
from baselines.ocsvm import OCSVM
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
# Modification: adding dataset
@click.option('--dataset_name', type=click.Choice(['newsgroups20', 'agnews','rncp']))
@click.option('--xp_path', type=click.Path(exists=True))
@click.option('--data_path', type=click.Path(exists=True))
@click.option('--results_file', type=click.Path(exists=True), default=None,
              help='File to output results of exepriments if launched by launcher file.')
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--kernel', type=click.Choice(['linear', 'poly', 'rbf']), default='linear', help='Kernel for the OC-SVM')
@click.option('--nu', type=float, default=0.1, help='OC-SVM hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--tokenizer', default='spacy', type=click.Choice(['spacy', 'bert','electra']), help='Select text tokenizer.')
@click.option('--clean_txt', is_flag=True, help='Specify if text should be cleaned in a pre-processing step.')
@click.option('--embedding_size', type=int, default=100, help='Size of the word vector embedding.')
# Modification: adding and renaming embedding options
@click.option('--pretrained_model', default=None,
              type=click.Choice([None, 'GloVe_6B', 'GloVe_42B', 'GloVe_840B', 'GloVe_twitter.27B', 'FastText_en', 'FastText_fr',
                                 'bert', 'FastText_dataset', 'FastText_class', 'PPMI_dataset', 'PPMI_class','DATE_dataset','DATE_class','DATE_class_noRMD','DATE_dataset_noRMD','Electra_pretrained']),
              help='Load pre-trained word vectors or language models to initialize the word embeddings.')
@click.option('--embedding_reduction', type=click.Choice(['none', 'mean', 'max']), default='mean',
              help='Specify if and how word embeddings should be reduced/aggregated.')
@click.option('--use_tfidf_weights', type=bool, default=False, help='Specify if tf-idf weights should be applied.')
@click.option('--normalize_embedding', is_flag=True, help='Specify if mean sentence embeddings should be normalized.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
# Modification: adding contamination option
@click.option('--cont', type=int, default=None,
              help='Specify which contaminated data to use')
# Modification: adding cont argument + renaming pretrained_model
def main(dataset_name, xp_path, data_path, results_file, load_config, load_model, seed, kernel, nu, tokenizer, clean_txt,
         embedding_size, pretrained_model, embedding_reduction, use_tfidf_weights, normalize_embedding,
         n_jobs_dataloader, normal_class, cont):
    """
    One-Class SVM for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Simplified logger with just the main parameters
    loggerSimp = logging.getLogger('simp')
    loggerSimp.setLevel(logging.INFO)
    formatterSimp = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_fileSimp = results_file
    file_handlerSimp = logging.FileHandler(log_fileSimp)
    file_handlerSimp.setLevel(logging.INFO)
    file_handlerSimp.setFormatter(formatterSimp)
    loggerSimp.addHandler(file_handlerSimp)

    # Print paths
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    if cfg.settings['pretrained_model'] == 'Electra_pretrained':
        cfg.settings['tokenizer'] = 'electra'
    logger.info('Tokenizer: %s' % cfg.settings['tokenizer'])
    logger.info('Clean text in pre-processing: %s' % cfg.settings['clean_txt'])
    logger.info('Word vector embedding size: %d' % cfg.settings['embedding_size'])
    # Modification: renaming pretrained_model
    logger.info('Load pre-trained word vectors: %s' % cfg.settings['pretrained_model'])
    logger.info('Reduction of word embeddings: %s' % cfg.settings['embedding_reduction'])
    logger.info('Use tf-idf weights: %s' % cfg.settings['use_tfidf_weights'])
    logger.info('Normalize embedding: %s' % cfg.settings['normalize_embedding'])

    # Print OC-SVM configuration
    logger.info('OC-SVM kernel: %s' % cfg.settings['kernel'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    loggerSimp.info('Dataset: %s' % dataset_name)
    loggerSimp.info('Normal class: %d' % normal_class)
    loggerSimp.info('Network: ocsvm')
    loggerSimp.info('Pre-trained model: %s' % cfg.settings['pretrained_model'])
    loggerSimp.info('Use tf-idf weights: %s' % cfg.settings['use_tfidf_weights'])
    loggerSimp.info('OC-SVM kernel: %s' % cfg.settings['kernel'])
    loggerSimp.info('Nu-paramerter: %.2f' % cfg.settings['nu'])
    loggerSimp.info('Cont: %d' % cfg.settings['cont'])

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Set seed for reproducibility
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])
        loggerSimp.info('Seed: %d' % cfg.settings['seed'])

    # Use 'cpu' as device for loading embeddings
    if cfg.settings['pretrained_model'] == 'Electra_pretrained':
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    # Modification: adding cont argument
    
    dataset = load_dataset(dataset_name, data_path, normal_class, cfg.settings['tokenizer'],
                           cfg.settings['use_tfidf_weights'], clean_txt=cfg.settings['clean_txt'], cont=cont)
    
    

    # Initialize OC-SVM model and set word embedding
    # Modification: adding cont argument + renaming pretrained_model
    
    ocsvm = OCSVM(cfg.settings['kernel'], cfg.settings['nu'])
    ocsvm.set_embedding(dataset,
                        embedding_size=cfg.settings['embedding_size'],
                        pretrained_word_vectors=cfg.settings['pretrained_model'],
                        embedding_reduction=cfg.settings['embedding_reduction'],
                        use_tfidf_weights=cfg.settings['use_tfidf_weights'],
                        normalize_embedding=cfg.settings['normalize_embedding'],
                        device=device,
                        cont=cont)
    
    # If specified, load model parameters from already trained model
    if load_model:
        ocsvm.load_model(import_path=load_model, device=device)
        logger.info('Loading model from %s.' % load_model)

    # Train model on dataset
    ocsvm.train(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    

    # Test model
    ocsvm.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)
    

    # Print most anomalous and most normal test samples
    # indices, labels, scores = zip(*ocsvm.results['test_scores'])
    # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    # idx_sorted = indices[np.argsort(scores)]  # sorted from lowest to highest anomaly score
    # idx_normal = idx_sorted[:50].tolist()
    # idx_outlier = idx_sorted[-50:].tolist()[::-1]
    # print_text_samples(dataset.test_set, dataset.encoder, idx_normal,
    #                    export_file=xp_path + '/normals', title='Most normal examples')
    # print_text_samples(dataset.test_set, dataset.encoder, idx_outlier,
    #                    export_file=xp_path + '/outliers', title='Most anomalous examples')
    # print_text_samples_with_scores(dataset.test_set, dataset.encoder, idx_sorted.tolist(), scores, np.argsort(scores),
    #                    labels, export_file=xp_path + '/resultsWithScores', title='Examples with Scores')

    # Save results, model, and configuration
    ocsvm.save_results(export_json=xp_path + '/results.json')
    ocsvm.save_model(export_path=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()
