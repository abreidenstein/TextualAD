#%%
import argparse
import logging
import os
import pickle as pkl
import sys
from datetime import datetime
from tqdm import tqdm
import click

# Adding simpletransformers from Date to the path
sys.path.append('date/simpletransformers')
sys.path.append('date')
import date

from simpletransformers.language_modeling.language_modeling_model import LanguageModelingModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
disc_hid_layers = 4
disc_hid_size = 256
gen_hid_layers = 1
gen_hid_size = 16
lowercase = 1
sched_params_opt = "plateau"
weight_decay = 0.1
disc_drop = 0.5
eval_anomaly = 1875
warmup = 1000
exp_prefix = "tests"
anomaly_batch_size = 16
train_batch_size = 16
eval_batch_size = 16
optimizer = "AdamW"
min_lr = 1e-4
max_lr = 1e-5
rtd_loss_weight = 50
rmd_loss_weight = 100
mlm_loss_weight = 1
sched_patience = 100000
eval_anomaly_after = 1
train_just_generator = 0
seq_len = 128
replace_tokens = 0
tensorboard_dir = "date/runs"
mlm_lr_ratio = 1
dump_histogram = 0

# To use a particular masks settings, the file create_masks should be used first to generate the needed masks and the resulting .pkl used in this file.

@click.command()

@click.option('--dataset_name', type=click.Choice(['newsgroups20', 'agnews','rncp']))
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--results_file', type=click.Path(exists=True), default=None,
              help='File to output results of exepriments if launched by launcher file.')
@click.option('--nb_masks', type=int, default=50, help='Number of different masks used.')
@click.option('--mask_prob', type=float, default=0.5, help='Percentage of the document that are masked.')
@click.option('--random_generator_seed', type=int, default=1,
              help='Set seed of the random generator. If the seed is 0, the generator is not random. This is not a general seed and does not allow reproductibility of experiments')
@click.option('--cont', type=int, default=0, help='Specify which contaminated data to use')
@click.option('--extract_reprs', type=click.Choice(['date','date_dataset', 'electra','electra_dataset', False]), default=False, help='Specify if the representations should be extracted and if the representations should be from date or from Electra (=date without rmd task) and learnt on the class or the dataset (even though the dataset did not give good results for us).')

def main(dataset_name,normal_class,results_file,nb_masks,mask_prob,random_generator_seed,cont,extract_reprs):
    # Get configuration
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Simplified logger with just the main parameters
    loggerSimp = logging.getLogger('simp')
    loggerSimp.setLevel(logging.INFO)
    formatterSimp = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_fileSimp = results_file
    file_handlerSimp = logging.FileHandler(log_fileSimp)
    file_handlerSimp.setLevel(logging.INFO)
    file_handlerSimp.setFormatter(formatterSimp)
    loggerSimp.addHandler(file_handlerSimp)
    loggerSimp.info('Network: date')
    loggerSimp.info('Dataset: %s' % dataset_name)
    loggerSimp.info('Normal class: %d' % normal_class)
    loggerSimp.info('Cont: %d' % cont)
    loggerSimp.info('Mask proba: %.2f' % mask_prob)
    loggerSimp.info('Seed: %d' % random_generator_seed)

    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date_time = now.strftime("%m%d%Y_%H%M%S")

    if sched_params_opt == "plateau":
        sched_params = {}
        sched_params['sched_name'] = 'plateau'
        sched_params['factor'] = 0.1
        sched_params['patience'] = sched_patience
        sched_params['verbose'] = True
        sched_params['threshold'] = 0.001
        sched_params['min_lr'] = min_lr
    else:
        sched_params = None

    run_name = f'{normal_class}_msk{nb_masks}_p{mask_prob}_rgen{random_generator_seed}_{date_time}_cont{cont}'

    print(f'RUN: {run_name}')

    train_args = {
        "fp16": False,
        "use_multiprocessing": False,
        "reprocess_input_data": False,
        "overwrite_output_dir": True,
        "num_train_epochs": 20,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "learning_rate": max_lr,
        "warmup_steps": warmup,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": 1,
        "block_size": seq_len + 2,
        "max_seq_length": seq_len + 2,
        "dataset_type": "simple",
        "logging_steps": 500,
        "evaluate_during_training": False, #set to False to launch multiple runs easier
        "evaluate_during_training_steps": 500,  #was 500
        "evaluate_during_training_steps_anomaly": eval_anomaly,
        "anomaly_batch_size": anomaly_batch_size,
        "evaluate_during_training_verbose": True,
        "use_cached_eval_features": True,
        "sliding_window": True,
        "vocab_size": 52000,
        "eval_anomalies": True,
        "random_generator": random_generator_seed,
        "use_rtd_loss": True,
        "rtd_loss_weight": rtd_loss_weight,
        "rmd_loss_weight": rmd_loss_weight,
        "mlm_loss_weight": mlm_loss_weight,
        "dump_histogram": dump_histogram,
        "eval_anomaly_after": eval_anomaly_after,
        "train_just_generator": train_just_generator,
        "replace_tokens": replace_tokens,
        "extract_scores": 1,
        "subset_name": normal_class,
        "extract_repr": 0,
        "extract_repr_for_OCSVM": extract_reprs != False, #Specify if the document representations of DATE should be saved
        #"dataset_repr" : False, #if True the model for the representations is not class trained but trained with the whole dataset
        # "vanilla_electra": {
        #     "no_masks": masks,
        # },
        # "vanilla_electra": False,
        "train_document": True,
        "tensorboard_dir": f'{tensorboard_dir}/{exp_prefix}/{run_name}',
        #"extract_reps": extract_reps,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "scores_export_path": f"./token_scores/{run_name}/",
        "generator_config": {
            "embedding_size": 128,
            "hidden_size": gen_hid_size,
            "num_hidden_layers": gen_hid_layers,
        },
        "discriminator_config": {
            "hidden_dropout_prob": disc_drop,
            "attention_probs_dropout_prob": disc_drop,
            "embedding_size": 128,
            "hidden_size": disc_hid_size,
            "num_hidden_layers": disc_hid_layers,
        },
        "mlm_lr_ratio": mlm_lr_ratio,
    }
        
    if extract_reprs == "electra":
        # Electra representations (trained on the normal class with no pre-training) are the same as DATE representations but without using the RMD task
        # However DATE representations showed better results in our experiments.
        train_args['rmd_loss_weight'] = 0
    if extract_reprs == "date_dataset" or extract_reprs == "electra_dataset":
        # The model for the representations is not class trained but trained with the whole dataset.
        # As an anomaly detection model DATE is not made to be trained on the whole dataset.
        # From what we experienced representations learnt on the dataset and used with OCSVM afterwards do not provide good results.
        train_args['dataset_repr'] = True
    else:
        train_args['dataset_repr'] = False
    if dataset_name != "rncp":
        # If we use the rncp dataset, the tokenizer should be trained on a french dataset.
        # If no tokenizer is specified, it will be trained on the trainFilesForTokenizer files.
        train_args["tokenizer_name"]="bert-base-uncased"
    
    train_file, test_file, outlier_file, dataset_train_file = readData(dataset_name,normal_class,cont)
    masks_ = pkl.load(open('date/mask_nbOfMasks'+str(nb_masks)+'_coveringProba'+str(mask_prob)+'.pkl', 'rb'))

    #for normal_class in tqdm([normal_class]): # from original code to print a work in progress bar
        #print('-' * 10, '\n', f'SUBSET: {normal_class}', '-' * 10)
    
    #for dataset_representations
    if train_args['extract_repr_for_OCSVM'] and train_args['dataset_repr']:
        model = LanguageModelingModel("electra",
                                    None,
                                    masks=masks_,
                                    args=train_args,
                                    train_files=dataset_train_file,
                                    use_cuda=True,
                                    trainFilesForTokenizer = dataset_train_file #to train tokenizer with the dataset
                                    )
        model.train_model_anomaly(dataset_train_file,
                                eval_file=test_file,
                                eval_file_outlier=outlier_file,
                                sched_params=sched_params,
                                trainFilesForSaving = train_file # files used to learn representations when we want them to be saved
                                )
    else :
        print("pas dataset")
        model = LanguageModelingModel("electra",
                                        None,
                                        masks=masks_,
                                        args=train_args,
                                        train_files=train_file,
                                        use_cuda=True,
                                        trainFilesForTokenizer = dataset_train_file #to train tokenizer with the dataset
                                        )
        
        model.train_model_anomaly(train_file,
                                    eval_file=test_file,
                                    eval_file_outlier=outlier_file,
                                    sched_params=sched_params,
                                    trainFilesForSaving = train_file # files used to learn representations when we want them to be saved
                                    ) 
    loggerSimp.info('------------')



def readData(datasetName,subset,cont):
    if datasetName == "agnews":
        datasetName = "ag"
        if subset == 0:
            subset = 'business'
        elif subset == 1:
            subset = 'sci'
        elif subset == 2:
            subset = 'sports'
        elif subset == 3:
            subset = 'world'
    elif datasetName == "newsgroups20":
        datasetName = "20ng"
        if subset == 0:
            subset = 'comp'
        elif subset == 1:
            subset = 'misc'
        elif subset == 2:
            subset = 'pol'
        elif subset == 3:
            subset = 'rec'
        elif subset == 4:
            subset = 'rel'
        elif subset == 5:
            subset = 'sci'
    elif datasetName == "rncp":
        datasetName = "rncp"
        if subset == 0:
            subset = '1-environnement'
        elif subset == 1:
            subset = '2-defense'
        elif subset == 2:
            subset = '3-patrimoine'
        elif subset == 3:
            subset = '4-economie'
        elif subset == 4:
            subset = '5-recherche'
        elif subset == 5:
            subset = '6-nautisme'
        elif subset == 6:
            subset = '7-aeronautique'
        elif subset == 7:
            subset = '8-securite'
        elif subset == 8:
            subset = '9-multimedia'
        elif subset == 9:
            subset = '10-humanitaire'
        elif subset == 10:
            subset = '11-nucleaire'
        elif subset == 11:
            subset = '12-enfance'
        elif subset == 12:
            subset = '13-saisonnier'
        elif subset == 13:
            subset = '14-assistance'
        elif subset == 14:
            subset = '15-sport'
        elif subset == 15:
            subset = '16-ingenierie'

    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date_time = now.strftime("%m%d%Y_%H%M%S")

    train_file = f"../data/{datasetName}_od/train/{int(cont)}/{subset}.txt"
    test_file = f"../data/{datasetName}_od/test/{subset}.txt"
    outlier_file = f"../data/{datasetName}_od/test/{subset}-outliers.txt"
    dataset_train_file = f"../data/{datasetName}_od/train/{datasetName}_fullTrainSet.txt"
    dataset_test_file = f"../data/{datasetName}_od/test/{datasetName}_fullTestSet.txt"
    return(train_file, test_file, outlier_file,dataset_train_file)
        
#run_exps(datasetName)
if __name__ == '__main__':
    main()