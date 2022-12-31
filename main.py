import logging
import math
import os
import sys
import torch
import json
import numpy as np
import wandb
from dataclasses import dataclass, field
from itertools import chain
import datasets
from datasets import load_dataset, Dataset, DatasetDict
import evaluate
import transformers

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

def bioasq_dict(type_name, sep_token):
    to_label = {
            'yes': 0,
            'no': 1,
        }
    text = []
    label = []
    data = json.load(open('./bioasq/' + type_name + '.json', 'r'))
    for d in data:
        tmp_str = d['question']
        for sent in d['text']:
            tmp_str = tmp_str + f'{sep_token}' + sent
        text.append(tmp_str)
        label.append(to_label[d['anwser']])
    dict = {'text': text, 'labels': label,}
    return dict

def chemprot_dict(type_name):
    to_label = {
        'UPREGULATOR': 0,
        'DOWNREGULATOR': 1,
        'AGONIST': 2,
        'ANTAGONIST': 3,
        'SUBSTRATE': 4,
        'PRODUCT-OF': 5,
        'INDIRECT-DOWNREGULATOR': 6,
        'INDIRECT-UPREGULATOR': 7,
        'INHIBITOR': 8,
        'ACTIVATOR': 9,
    }
    text = []
    label = []
    file = open('./chemprot/' + type_name + '.jsonl', 'r')
    for line in file.readlines():
        data = json.loads(line)
        text.append(data['text'])
        if data['label'] in to_label.keys():
            label.append(to_label[data['label']])
        else:
            label.append(6)
    dict = {'text': text, 'labels': label,}
    return dict

def get_dataset(name, sep_token):
    if name == 'bioasq':
        dict_train = bioasq_dict('train', sep_token);
        dict_test = bioasq_dict('test', sep_token);
    else:
        dict_train = chemprot_dict('train');
        dict_test = chemprot_dict('test');
    dataset = DatasetDict({
        'train': Dataset.from_dict(dict_train),
        'test': Dataset.from_dict(dict_test),
    })
    return dataset

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )   

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
        
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    # training_args = TrainingArguments(output_dir="res_sci", seed=34, do_train=1, do_eval=1, num_train_epochs=1, per_device_train_batch_size=32, per_device_eval_batch_size=32, logging_steps=10)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        raw_datasets = get_dataset(data_args.dataset_name, '<sep>')
   
    # Labels
    to_labelnum = {
        'bio': 2,
        'che': 11,
    }
    num_labels = to_labelnum[data_args.dataset_name[:3]]

    # Load pretrained model and tokenizer
    checkpoint = model_args.model_path
    config = AutoConfig.from_pretrained(checkpoint, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
    
    # Preprocessing the raw_datasets
    # Padding strategy
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=True, max_length=tokenizer.model_max_length, truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    predict_dataset = tokenized_datasets["test"]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        if data_args.dataset_name == 'bioasq':
            return evaluate.load('accuracy').compute(predictions=predictions, references=labels)
        else:
            return evaluate.load('f1').compute(predictions=predictions, references=labels, average="micro")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    if training_args.do_eval:
        eval_results = trainer.evaluate(eval_dataset=predict_dataset)
        print(eval_results)
        f = open('results.txt', mode='a')
        f.write(data_args.dataset_name+' ' + model_args.model_path + ' :\n')
        if data_args.dataset_name == 'bioasq':
            f.write('accuracy: ' + str(eval_results)+'\n')
        else:
            f.write('micro-f1: ' + str(eval_results)+'\n')
        f.close()

if __name__ == "__main__":
    main()