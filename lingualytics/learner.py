import argparse
import csv
import io
import logging
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tabulate import tabulate
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler)
from tqdm.auto import tqdm, trange
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          XLMForSequenceClassification,
                          XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer, XLMTokenizer,
                          get_linear_schedule_with_warmup)

logger = logging.getLogger(__name__)
class CustomDataset(Dataset):
    def __init__(self, input_ids, labels, present=None):
        self.input_ids = input_ids
        self.labels = labels
        self.present = present

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.present:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long), self.present[i]
        else:
            return torch.tensor(self.input_ids[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)

class Learner():
        
    def __init__(self, data_dir='./dataset', output_dir='./output', lr = 5e-5, train_bs = 64, eval_bs = 64, model_type = 'bert', 
                model_name = 'bert-base-multilingual-cased', save_steps = 1, max_seq_length = 256, seed = 42,
                weight_decay = 0.0, adam_epsilon = 1e-8, max_grad_norm = 1.0, num_train_epochs = 5, device = None, dataset=None):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        if dataset is not None:
            self.dataset = dataset
            self.download_dataset()
        self.test_file = self.data_dir / 'test.txt'
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm
        self.num_train_epochs = num_train_epochs
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.seed = seed
        self.model_type = model_type
        self.model_name = model_name
        self.save_steps = save_steps
        self.max_seq_length = max_seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.setup_model()

    def download_dataset(self):
        logger.info(f'Downloading and extracting dataset to {self.data_dir}')
        url = f'https://github.com/lingualytics/py-lingualytics/raw/master/datasets/zip-files/{self.dataset}.zip'
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(self.data_dir)
        self.data_dir = self.data_dir / self.dataset

    def setup_model(self):
        # Set up logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)

        # Set seed
        self.set_seed()

        # Prepare data
        self.labels = self.get_labels()
        num_labels = len(self.labels)

        # Initialize model
        tokenizer_class = {'xlm': XLMTokenizer, 'bert': BertTokenizer, 'xlm-roberta': XLMRobertaTokenizer}
        if self.model_type not in tokenizer_class.keys():
            print('Model type has to be xlm/xlm-roberta/bert')
            exit(0)
        self.tokenizer = tokenizer_class[self.model_type].from_pretrained(
            self.model_name, do_lower_case=True)
        model_class = {'xlm': XLMForSequenceClassification, 'bert': BertForSequenceClassification, 'xlm-roberta': XLMRobertaForSequenceClassification}
        self.model = model_class[self.model_type].from_pretrained(
            self.model_name, num_labels=num_labels)

        self.model.to(self.device)

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def simple_accuracy(self, preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(self, preds, labels):
        acc = self.simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
        precision = precision_score(
            y_true=labels, y_pred=preds, average='weighted')
        recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
        return{
            'acc': acc,
            'f1': f1,
            'acc_and_f1': (acc + f1) / 2,
            'precision': precision,
            'recall': recall
        }

    def read_examples_from_file(self, mode = 'train'):
        file_path = self.data_dir / f'{mode}.txt'
        examples = []
        with open(file_path, 'r') as infile:
            lines = infile.read().strip().split('\n')
        for line in lines:
            x = line.split('\t')
            text = x[0]
            label = x[1]
            examples.append({'text': text, 'label': label})
        if mode == 'test':
            for i in range(len(examples)):
                if examples[i]['text'] == 'not found':
                    examples[i]['present'] = False
                else:
                    examples[i]['present'] = True
        return examples

    def convert_examples_to_features(self, examples,
                                    label_list,
                                    tokenizer):

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []

        for (ex_index, example) in enumerate(examples):

            sentence = example['text']
            label = example['label']

            sentence_tokens = tokenizer.tokenize(sentence)[:self.max_seq_length - 2]
            sentence_tokens = [tokenizer.cls_token] + \
                sentence_tokens + [tokenizer.sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)

            label = label_map[label]
            features.append({'input_ids': input_ids,
                            'label': label})
            if 'present' in example:
                features[-1]['present'] = example['present']

        return features

    def get_labels(self):
        all_path = self.data_dir / 'train.txt'
        labels = []
        with open(all_path, 'r') as infile:
            lines = infile.read().strip().split('\n')

        for line in lines:
            splits = line.split('\t')
            label = splits[-1]
            if label not in labels:
                labels.append(label)
        return labels

    def load_and_cache_examples(self, tokenizer, labels, mode):

        logger.info(f'Creating features from dataset file at {self.data_dir}')
        examples = self.read_examples_from_file(mode)
        features = self.convert_examples_to_features(examples, labels, tokenizer)

        # Convert to Tensors and build dataset
        all_input_ids = [f['input_ids'] for f in features]
        all_labels = [f['label'] for f in features]
        args = [all_input_ids, all_labels]
        if 'present' in features[0]:
            present = [1 if f['present'] else 0 for f in features]
            args.append(present)

        dataset = CustomDataset(*args)
        return dataset

    def collate(self, examples):
        padding_value = 0

        first_sentence = [t[0] for t in examples]
        first_sentence_padded = torch.nn.utils.rnn.pad_sequence(
            first_sentence, batch_first=True, padding_value=padding_value)

        max_length = first_sentence_padded.shape[1]
        first_sentence_attn_masks = torch.stack([torch.cat([torch.ones(len(t[0]), dtype=torch.long), torch.zeros(
            max_length - len(t[0]), dtype=torch.long)]) for t in examples])

        labels = torch.stack([t[1] for t in examples])

        return first_sentence_padded, first_sentence_attn_masks, labels

    def train(self, train_dataset, valid_dataset):
        model, tokenizer, labels = self.model, self.tokenizer, self.labels
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=self.train_bs, collate_fn=self.collate)
        train_bs = self.train_bs

        # Prepare optimizer
        t_total = len(train_dataloader) * self.num_train_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                        lr=self.learning_rate, eps=self.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total // 10, num_training_steps=t_total)

        # Train!
        logger.info('***** Running training *****')
        logger.info('  Num examples = %d', len(train_dataset))
        logger.info('  Num Epochs = %d', self.num_train_epochs)
        logger.info('  Instantaneous batch size per GPU = %d', train_bs)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(self.num_train_epochs), desc='Epoch')
        self.set_seed()
        best_f1_score = 0
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[2]}
                outputs = model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.max_grad_norm)

                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            # Checking for validation accuracy and stopping after drop in accuracy for 3 epochs
            results = self.evaluate('validation')
            if results.get('f1') > best_f1_score and self.save_steps > 0:
                best_f1_score = results.get('f1')
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(self.output_dir)
                tokenizer.save_pretrained(self.output_dir)
                torch.save(self, os.path.join(
                    self.output_dir, 'training_args.bin'))

        return global_step, tr_loss / global_step

    def evaluate(self, mode, prefix=''):
        eval_dataset = self.load_and_cache_examples(self.tokenizer, self.labels, mode=mode)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=self.eval_bs, collate_fn=self.collate)
        results = {}

        # Evaluation
        logger.info('***** Running evaluation %s *****', prefix)
        logger.info('  Num examples = %d', len(eval_dataset))
        logger.info('  Batch size = %d', self.eval_bs)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc='Evaluating'):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[2]}
                '''print(inputs['input_ids'])
                print(inputs['attention_mask'])
                print(inputs['token_type_ids'])'''
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            # print(preds)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        preds = np.argmax(preds, axis=1)
        # print(preds)
        if mode == 'test':
            preds_list = []
            label_map = {i: label for i, label in enumerate(self.labels)}

            for i in range(out_label_ids.shape[0]):
                # print(eval_dataset[i])
                if eval_dataset[i][2] == 0:
                    preds_list.append('not found')
                else:
                    preds_list.append(label_map[preds[i]])

            return preds_list

        else:
            result = self.acc_and_f1(preds, out_label_ids)
            results.update(result)

            logger.info('***** Eval results %s *****', prefix)
            for key in sorted(result.keys()):
                logger.info('  %s = %s', key, str(result[key]))

            return results

    def fit(self):
        # Training

        # logger.info('Training/evaluation parameters %s', args)
        train_dataset = self.load_and_cache_examples(
            self.tokenizer, self.labels, mode='train')
        valid_dataset = self.load_and_cache_examples(
            self.tokenizer, self.labels, mode='validation')
        global_step, tr_loss = self.train(train_dataset, valid_dataset)
        logger.info(' global_step = %s, average loss = %s', global_step, tr_loss)

        # Evaluation
        results = {}

        results = self.evaluate(mode='validation')
        print(tabulate(results.items(),headers=['metric','score']))
        preds = self.evaluate(mode='test')

        # Saving predictions
        output_test_predictions_file = self.output_dir / 'test_predictions.txt'
        with open(output_test_predictions_file, "w") as writer:
            writer.write('\n'.join(preds))
