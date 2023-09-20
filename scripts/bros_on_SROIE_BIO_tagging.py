import datetime
import itertools
import json
import math
import os
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from overrides import overrides
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BrosConfig,
    BrosForTokenClassification,
    BrosProcessor,
)

from datasets import load_dataset, load_from_disk


def linear_scheduler(optimizer, warmup_steps, training_steps, last_epoch=-1):
    """linear_scheduler with warmup from huggingface"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(training_steps - current_step)
            / float(max(1, training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def cosine_scheduler(
    optimizer, warmup_steps, training_steps, cycles=0.5, last_epoch=-1
):
    """Cosine LR scheduler with warmup from huggingface"""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = current_step - warmup_steps
        progress /= max(1, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycles * 2 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def multistep_scheduler(optimizer, warmup_steps, milestones, gamma=0.1, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # calculate a warmup ratio
            return current_step / max(1, warmup_steps)
        else:
            # calculate a multistep lr scaling ratio
            idx = np.searchsorted(milestones, current_step)
            return gamma**idx

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class SROIEBIODataset(Dataset):
    def __init__(
        self, dataset, tokenizer, max_seq_length=512, split="train", bio_format=True
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        self.bio_format = bio_format

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        self.examples = load_dataset(self.dataset)[split]

        self.class_names = ["address", "company", "date", "total"]
        self.bio_class_names = ["O"]
        for class_name in self.class_names:
            self.bio_class_names.extend([f"B_{class_name}", f"I_{class_name}"])
        self.bio_class_name2idx = dict(
            [
                (bio_class_name, idx)
                for idx, bio_class_name in enumerate(self.bio_class_names)
            ]
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]

        width, height = sample["img"].size
        words: List[str] = sample["words"]
        bboxes: List[List[int]] = sample["bboxes"]
        labels: List[str] = sample["labels"]

        sep_bbs = [width, height] * 2
        cls_bbs = [0] * 4

        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        are_box_end_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)

        # encode(tokenize) each word from words (List[str])
        input_ids_list: List[List[int]] = [
            self.tokenizer.encode(e, add_special_tokens=False) for e in words
        ]
        tokens_length_list: List[int] = [len(l) for l in input_ids_list]

        # each word is splited into tokens, and since only have bbox for each word, tokens from same word get same bbox
        # but we want to calculate loss for only "first token of the box", so we make box_first_token masks
        # add 1 in the end, considering [CLS] token that will be added to the beginning
        end_indices = np.array(list(itertools.accumulate(tokens_length_list))) + 1
        st_indices = end_indices - np.array(tokens_length_list)

        end_indices = end_indices[end_indices < self.max_seq_length - 1]
        if len(st_indices) > len(end_indices):
            st_indices = st_indices[: len(end_indices)]

        # end_indices_mask = np.zeros(self.max_seq_length) + 1
        are_box_first_tokens[st_indices] = True
        are_box_end_tokens[end_indices] = True

        # duplicate each word's bbox to length of tokens (of each word)
        # e.g. AAA -> (tokenize) -> A, A, A then copy bbox of AAA 3 times
        bboxes_list: List[List[List[int]]] = [
            [bboxes[idx] for _ in range(len(l))] for idx, l in enumerate(input_ids_list)
        ]

        # do duplicate each word's label to length of tokens (of each word)
        # if the word's label starts with 'B' tag, then convert input_ids' label to ['B', 'I', 'I', ...]
        labels_list: List[List[str]] = []
        for idx, l in enumerate(input_ids_list):
            word_label = labels[idx]
            if word_label.startswith("B_"):
                class_name = word_label.split("_")[1]
                input_ids_label = [word_label] + [
                    "I_" + class_name for _ in range(len(l) - 1)
                ]
            else:
                input_ids_label = [word_label for _ in range(len(l))]
            labels_list.append(input_ids_label)

        # flatten input_ids, bboxes, labels
        input_ids: List[int] = list(itertools.chain.from_iterable(input_ids_list))
        bboxes: List[List[int]] = list(itertools.chain.from_iterable(bboxes_list))
        labels: List[str] = list(itertools.chain.from_iterable(labels_list))

        # sanity check
        assert len(input_ids) == len(bboxes) and len(input_ids) == len(labels)

        ##############################################################
        # For [CLS] and [SEP]

        ### update input_ids with correspoding cls_token_id in the begining and sep_token_id in the end
        input_ids = (
            [self.cls_token_id]
            + input_ids[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )

        # # are_box_first_tokens = [False] + are_box_first_tokens[: self.max_seq_length - 2] + [False]
        # are_box_first_tokens = np.insert(are_box_first_tokens, 0, 0)
        # are_box_first_tokens[-1] = 0

        # # are_box_end_tokens = [False] + are_box_end_tokens[: self.max_seq_length - 2] + [False]
        # are_box_end_tokens = np.insert(are_box_end_tokens, 0, 0) # obj -> index
        # are_box_end_tokens[-1] = 0

        ### update labels
        labels = ["O"] + labels[: self.max_seq_length - 2] + ["O"]
        labels = [self.bio_class_name2idx[l] for l in labels]

        ### update bboxes with correspoding cls_token bbox in the begining and sep_token bbox in the end
        if len(bboxes) == 0:  # When len(json_obj["words"]) == 0 (no OCR result)
            bboxes = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            bboxes = [cls_bbs] + bboxes[: self.max_seq_length - 2] + [sep_bbs]
        ##############################################################

        ##############################################################
        # prepare padded input_ids, bboxes (padded to self.max_seq_length)
        len_ori_input_ids = len(input_ids)

        padded_input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        padded_input_ids[:len_ori_input_ids] = input_ids

        padded_labels = np.zeros(self.max_seq_length, dtype=int)
        padded_labels[:len_ori_input_ids] = np.array(labels)

        attention_mask = np.zeros(self.max_seq_length, dtype=int)
        attention_mask[:len_ori_input_ids] = 1

        # prepare padded_bboxes
        padded_bboxes = np.zeros((self.max_seq_length, 4), dtype=np.float32)

        # convert list to numpy array
        bboxes = np.array(bboxes)

        # save original bboxes in padded_bboxes
        padded_bboxes[:len_ori_input_ids, :] = bboxes
        ##############################################################

        # Normalize bbox -> 0 ~ 1
        padded_bboxes[:, [0, 2]] = padded_bboxes[:, [0, 2]] / width
        padded_bboxes[:, [1, 3]] = padded_bboxes[:, [1, 3]] / height

        padded_input_ids = torch.from_numpy(padded_input_ids)
        padded_bboxes = torch.from_numpy(padded_bboxes)
        padded_labels = torch.from_numpy(padded_labels)
        attention_mask = torch.from_numpy(attention_mask)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
        are_box_end_tokens = torch.from_numpy(are_box_end_tokens)

        return_dict = {
            "input_ids": padded_input_ids,
            "bbox": padded_bboxes,
            "attention_mask": attention_mask,
            "bio_labels": padded_labels,
            "are_box_first_tokens": are_box_first_tokens,
            "are_box_end_tokens": are_box_end_tokens,
        }

        return return_dict


class BROSDataPLModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_batch_size = self.cfg.train.batch_size
        self.val_batch_size = self.cfg.val.batch_size
        self.train_dataset = None
        self.val_dataset = None

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.cfg.val.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        return loader

    @overrides
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        return batch


def parse_from_seq(seq, class_names):
    parsed = [[] for _ in range(len(class_names))]
    for i, label_id_tensor in enumerate(seq):
        label_id = label_id_tensor.item()

        if label_id == 0:  # O
            continue

        class_id = (label_id - 1) // 2
        is_b_tag = label_id % 2 == 1

        if is_b_tag:
            parsed[class_id].append((i,))
        elif len(parsed[class_id]) != 0:
            parsed[class_id][-1] = parsed[class_id][-1] + (i,)

    parsed = [set(indices_list) for indices_list in parsed]

    return parsed


class BROSModelPLModule(pl.LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.optimizer_types = {
            "sgd": SGD,
            "adam": Adam,
            "adamw": AdamW,
        }
        self.loss_func = nn.CrossEntropyLoss()
        self.class_names = None
        self.bio_class_names = None
        self.tokenizer = tokenizer
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx, *args):
        # unpack batch
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]
        bbox_first_token_mask = batch["are_box_first_tokens"]
        labels = batch["bio_labels"]

        # inference model
        prediction = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            bbox_first_token_mask=bbox_first_token_mask,
            labels=labels,
        )

        loss = prediction.loss
        self.log_dict({"train_loss": loss}, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, *args):
        # unpack batch
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]
        are_box_first_tokens = batch["are_box_first_tokens"]
        are_box_end_tokens = batch["are_box_end_tokens"]
        gt_labels = batch["bio_labels"]
        labels = batch["bio_labels"]

        # inference model
        prediction = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            bbox_first_token_mask=are_box_first_tokens,
            labels=labels,
        )

        val_loss = prediction.loss
        pred_labels = torch.argmax(prediction.logits, -1)

        n_batch_gt_classes, n_batch_pred_classes, n_batch_correct_classes = 0, 0, 0
        batch_size = prediction.logits.shape[0]

        for example_idx, (
            pred_label,
            gt_label,
            bbox_first_token_mask,
            box_end_token_mask,
        ) in enumerate(
            zip(pred_labels, gt_labels, are_box_first_tokens, are_box_end_tokens)
        ):
            # validation loss : # calculate validation loss of "box_first_tokens" only
            valid_gt_label = gt_label[bbox_first_token_mask]
            valid_pred_label = pred_label[bbox_first_token_mask]

            gt_parse = parse_from_seq(valid_gt_label, self.class_names)
            pred_parse = parse_from_seq(valid_pred_label, self.class_names)

            """
            (Pdb++) valid_gt_label
            tensor([3, 4, 4, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')

            --> after parse

            (Pdb++) gt_parse
            [{(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)}, {(0, 1, 2, 3)}, {(45,)}, {(113,)}]
            """

            n_gt_classes = sum(
                [len(gt_parse[class_idx]) for class_idx in range(len(self.class_names))]
            )
            n_pred_classes = sum(
                [
                    len(pred_parse[class_idx])
                    for class_idx in range(len(self.class_names))
                ]
            )
            n_correct_classes = sum(
                [
                    len(gt_parse[class_idx] & pred_parse[class_idx])
                    for class_idx in range(len(self.class_names))
                ]
            )
            n_batch_gt_classes += n_gt_classes
            n_batch_pred_classes += n_pred_classes
            n_batch_correct_classes += n_correct_classes

            box_first_token_idx2ori_idx = bbox_first_token_mask.nonzero(as_tuple=True)[0]
            box2token_span_maps = (
                torch.hstack(
                    (
                        (bbox_first_token_mask == True).nonzero(),
                        (box_end_token_mask == True).nonzero(),
                    )
                )
                .cpu()
                .numpy()
            )
            start_token_idx2end_token_idx = {e[0]: e[1] for e in box2token_span_maps}

            pred_cls2text = {name: [] for name in self.class_names}
            gt_cls2text = deepcopy(pred_cls2text)
            correct_cls2text = deepcopy(pred_cls2text)
            incorrect_cls2text = deepcopy(pred_cls2text)
            for cls_idx, cls_name in enumerate(self.class_names):
                # all pred text for cls
                for box_first_token_indices in pred_parse[cls_idx]:
                    ori_indices = (
                        box_first_token_idx2ori_idx[
                            torch.tensor(box_first_token_indices)
                        ]
                        .cpu()
                        .tolist()
                    )
                    text_span = torch.tensor(
                        list(
                            range(
                                ori_indices[0],
                                start_token_idx2end_token_idx[ori_indices[-1]],
                            )
                        )
                    )
                    pred_text = self.tokenizer.decode(input_ids[example_idx][text_span])
                    pred_cls2text[cls_name].append(pred_text)

                # all gt text for cls
                for box_first_token_indices in gt_parse[cls_idx]:
                    ori_indices = (
                        box_first_token_idx2ori_idx[
                            torch.tensor(box_first_token_indices)
                        ]
                        .cpu()
                        .tolist()
                    )
                    text_span = torch.tensor(
                        list(
                            range(
                                ori_indices[0],
                                start_token_idx2end_token_idx[ori_indices[-1]],
                            )
                        )
                    )
                    gt_text = self.tokenizer.decode(input_ids[example_idx][text_span])
                    gt_cls2text[cls_name].append(gt_text)

                # all correct text for cls
                for box_first_token_indices in pred_parse[cls_idx] & gt_parse[cls_idx]:
                    ori_indices = (
                        box_first_token_idx2ori_idx[
                            torch.tensor(box_first_token_indices)
                        ]
                        .cpu()
                        .tolist()
                    )
                    text_span = torch.tensor(
                        list(
                            range(
                                ori_indices[0],
                                start_token_idx2end_token_idx[ori_indices[-1]],
                            )
                        )
                    )
                    correct_text = self.tokenizer.decode(
                        input_ids[example_idx][text_span]
                    )
                    correct_cls2text[cls_name].append(correct_text)

                # all incorrect text for cls (text in gt but not in pred + text not in gt but in pred)
                for box_first_token_indices in pred_parse[cls_idx] ^ gt_parse[cls_idx]:
                    ori_indices = (
                        box_first_token_idx2ori_idx[
                            torch.tensor(box_first_token_indices)
                        ]
                        .cpu()
                        .tolist()
                    )
                    text_span = torch.tensor(
                        list(
                            range(
                                ori_indices[0],
                                start_token_idx2end_token_idx[ori_indices[-1]],
                            )
                        )
                    )
                    incorrect_text = self.tokenizer.decode(
                        input_ids[example_idx][text_span]
                    )
                    incorrect_cls2text[cls_name].append(incorrect_text)

        step_out = {
            "n_batch_gt_classes": n_batch_gt_classes,
            "n_batch_pr_classes": n_batch_pred_classes,
            "n_batch_correct_classes": n_batch_correct_classes,
        }

        self.validation_step_outputs.append(step_out)
        self.log_dict({"val_loss": val_loss}, sync_dist=True, prog_bar=True)
        self.log_dict(step_out, sync_dist=True)

        return step_out

    def on_validation_epoch_end(self):
        all_preds = self.validation_step_outputs

        n_total_gt_classes, n_total_pr_classes, n_total_correct_classes = 0, 0, 0

        for step_out in all_preds:
            n_total_gt_classes += step_out["n_batch_gt_classes"]
            n_total_pr_classes += step_out["n_batch_pr_classes"]
            n_total_correct_classes += step_out["n_batch_correct_classes"]

        precision = (
            0.0
            if n_total_pr_classes == 0
            else n_total_correct_classes / n_total_pr_classes
        )
        recall = (
            0.0
            if n_total_gt_classes == 0
            else n_total_correct_classes / n_total_gt_classes
        )
        f1 = (
            0.0
            if recall * precision == 0
            else 2.0 * recall * precision / (recall + precision)
        )

        self.log_dict(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            sync_dist=True,
        )

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        scheduler = self._get_lr_scheduler(optimizer)
        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def _get_optimizer(self):
        opt_cfg = self.cfg.train.optimizer
        method = opt_cfg.method.lower()

        if method not in self.optimizer_types:
            raise ValueError(f"Unknown optimizer method={method}")

        kwargs = dict(opt_cfg.params)
        kwargs["params"] = self.model.parameters()
        optimizer = self.optimizer_types[method](**kwargs)

        return optimizer

    def _get_lr_scheduler(self, optimizer):
        cfg_train = self.cfg.train
        lr_schedule_method = cfg_train.optimizer.lr_schedule.method
        lr_schedule_params = cfg_train.optimizer.lr_schedule.params

        if lr_schedule_method is None:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1)
        elif lr_schedule_method == "step":
            scheduler = multistep_scheduler(optimizer, **lr_schedule_params)
        elif lr_schedule_method == "cosine":
            total_samples = cfg_train.max_epochs * cfg_train.num_samples_per_epoch
            total_batch_size = cfg_train.batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = cosine_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        elif lr_schedule_method == "linear":
            total_samples = cfg_train.max_epochs * cfg_train.num_samples_per_epoch
            total_batch_size = cfg_train.batch_size * self.trainer.world_size
            max_iter = total_samples / total_batch_size
            scheduler = linear_scheduler(
                optimizer, training_steps=max_iter, **lr_schedule_params
            )
        else:
            raise ValueError(f"Unknown lr_schedule_method={lr_schedule_method}")

        return scheduler

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_path = Path(self.cfg.workspace) / self.cfg.exp_name / self.cfg.exp_version
        model_save_path = (
            Path(self.cfg.workspace)
            / self.cfg.exp_name
            / self.cfg.exp_version
            / "huggingface_model"
        )
        tokenizer_save_path = (
            Path(self.cfg.workspace)
            / self.cfg.exp_name
            / self.cfg.exp_version
            / "huggingface_tokenizer"
        )
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(tokenizer_save_path)


def train(cfg):
    cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints")
    cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs")
    cfg.exp_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # pprint cfg
    print(OmegaConf.to_yaml(cfg))

    # set env
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock with tokenizer
    pl.seed_everything(cfg.seed)

    # Load Tokenizer (going to be used in dataset to to convert texts to input_ids)
    tokenizer = BrosProcessor.from_pretrained(cfg.tokenizer_path).tokenizer

    # prepare SROIE dataset
    train_dataset = SROIEBIODataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        split="train",
    )

    val_dataset = SROIEBIODataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        split="val",
    )

    # make data module & update data_module train and val dataset
    data_module = BROSDataPLModule(cfg)
    data_module.train_dataset = train_dataset
    data_module.val_dataset = val_dataset

    # Load BROS config & pretrained model
    ## update config
    bros_config = BrosConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
    bio_class_names = train_dataset.bio_class_names
    id2label = {idx: name for idx, name in enumerate(bio_class_names)}
    label2id = {name: idx for idx, name in id2label.items()}
    bros_config.id2label = id2label
    bros_config.label2id = label2id

    ## load pretrained model
    bros_model = BrosForTokenClassification.from_pretrained(
        cfg.model.pretrained_model_name_or_path, config=bros_config
    )

    # model module setting
    model_module = BROSModelPLModule(cfg, tokenizer=tokenizer)
    model_module.model = bros_model
    model_module.class_names = train_dataset.class_names
    model_module.bio_class_names = train_dataset.bio_class_names

    # define trainer logger, callbacks
    loggers = TensorBoardLogger(
        save_dir=cfg.workspace,
        name=cfg.exp_name,
        version=cfg.exp_version,
        default_hp_metric=False,
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.workspace) / cfg.exp_name / cfg.exp_version / "checkpoints",
        filename="bros-sroie-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,  # if you save more than 1 model,
        # then checkpoint and huggingface model are not guaranteed to be matching
        # because we are saving with huggingface model with save_pretrained method
        # in "on_save_checkpoint" method in "BROSModelPLModule"
        mode="min",
    )

    model_summary_callback = ModelSummary(max_depth=5)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min"
    )

    # define Trainer and start training
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        strategy="ddp_find_unused_parameters_true",
        num_nodes=cfg.get("num_nodes", 1),
        precision=16 if cfg.train.use_fp16 else 32,
        logger=loggers,
        callbacks=[
            lr_callback,
            checkpoint_callback,
            model_summary_callback,
            early_stop_callback,
        ],
        max_epochs=cfg.train.max_epochs,
        num_sanity_val_steps=3,
        gradient_clip_val=cfg.train.clip_gradient_value,
        gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    # load training config
    finetune_sroie_ee_bio_config = {
        "workspace": "./finetune_sroie_ee_bio",
        "exp_name": "bros-base-uncased_sroie",
        "tokenizer_path": "naver-clova-ocr/bros-base-uncased",
        "dataset": "jinho8345/sroie-bio",
        "task": "ee",
        "seed": 1,
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
        "model": {
            "pretrained_model_name_or_path": "jinho8345/bros-base-uncased",
            "max_seq_length": 512,
        },
        "train": {
            "batch_size": 16,
            "num_samples_per_epoch": 526,
            "max_epochs": 30,
            "use_fp16": True,
            "accelerator": "gpu",
            "strategy": {"type": "ddp"},
            "clip_gradient_algorithm": "norm",
            "clip_gradient_value": 1.0,
            "num_workers": 8,
            "optimizer": {
                "method": "adamw",
                "params": {"lr": 5e-05},
                "lr_schedule": {"method": "linear", "params": {"warmup_steps": 0}},
            },
            "val_interval": 1,
        },
        "val": {"batch_size": 16, "num_workers": 0, "limit_val_batches": 1.0},
    }

    # convert dictionary to omegaconf and update config
    cfg = OmegaConf.create(finetune_sroie_ee_bio_config)
    train(cfg)
