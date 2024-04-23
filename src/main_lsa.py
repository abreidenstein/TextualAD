import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.misc import print_text_samples
from baselines.lsa import LSA
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
@click.command()
@click.option('--dataset_name', type=click.Choice(['newsgroups20', 'agnews','rncp']))
@click.option('--xp_path', type=click.Path(exists=True))
@click.option('--data_path', type=click.Path(exists=True))
@click.option('--results_file', type=click.Path(exists=True), default=None,
              help='File to output results of exepriments if launched by launcher file.')
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--n_components', type=int, default=10, help='Number of components for LSA/Truncated SVD')
@click.option('--tokenizer', default='spacy', type=click.Choice(['spacy', 'bert']), help='Select text tokenizer.')
@click.option('--clean_txt', is_flag=True, help='Specify if text should be cleaned in a pre-processing step.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--cont', type=int, default=None,
              help='Specify which contaminated data to use')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')


def main(dataset_name, xp_path, data_path, results_file, load_config, normal_class, cont,clean_txt,tokenizer,n_components,seed):
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
    logger.info('Tokenizer: %s' % cfg.settings['tokenizer'])
    logger.info('Clean text in pre-processing: %s' % cfg.settings['clean_txt'])
    loggerSimp.info('Cont: %d' % cfg.settings['cont'])
    
    # Print LSA configuration
    logger.info('n_components: %.2f' % cfg.settings['n_components'])

    loggerSimp.info('Dataset: %s' % dataset_name)
    loggerSimp.info('Normal class: %d' % normal_class)
    loggerSimp.info('Network: lsa')
    
    loggerSimp.info('n_components: %.2f' % cfg.settings['n_components'])

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Set seed for reproducibility
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])
        loggerSimp.info('Seed: %d' % cfg.settings['seed'])


    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, cfg.settings['tokenizer'],
                           False, clean_txt=cfg.settings['clean_txt'], cont=cont,useBow=True)

    # Initialize OC-SVM model and set word embedding
    lsa = LSA(cfg.settings['n_components'],cfg.settings['seed'])

    # Train model on dataset
    lsa.train(dataset)

    # Test model
    lsa.test(dataset)

    # Print most anomalous and most normal test samples
    indices, labels, scores = zip(*lsa.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_sorted = indices[np.argsort(scores)]  # sorted from lowest to highest anomaly score
    idx_normal = idx_sorted[:50].tolist()
    idx_outlier = idx_sorted[-50:].tolist()[::-1]
    print_text_samples(dataset.test_set, dataset.encoder, idx_normal,
                       export_file=xp_path + '/normals', title='Most normal examples')
    print_text_samples(dataset.test_set, dataset.encoder, idx_outlier,
                       export_file=xp_path + '/outliers', title='Most anomalous examples')

    # Save results, model, and configuration
    lsa.save_results(export_json=xp_path + '/results.json')
    lsa.save_model(export_path=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')


if __name__ == '__main__':
    main()
