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
    BrosTokenizer,
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


class FUNSDBIOESDataset(Dataset):
    """FUNSD BIOES tagging Dataset

    FUNSD : Form Understanding in Noisy Scanned Documents
    BIOES tagging : begin, in, out, end, single tagging

    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_seq_length=512,
        split="train",
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id

        self.examples = load_dataset(self.dataset)[split]

        self.class_names = ["header", "question", "answer"]
        self.pad_token = self.tokenizer.pad_token
        self.ignore_label_id = -100

        # self.out_class_name = "other"
        self.out_class_name = "other"
        self.bioes_class_names = [self.out_class_name]
        for class_name in self.class_names:
            self.bioes_class_names.extend(
                [
                    f"B_{class_name}",
                    f"I_{class_name}",
                    f"E_{class_name}",
                    f"S_{class_name}",
                ]
            )
        self.bioes_class_name2idx = {
            name: idx for idx, name in enumerate(self.bioes_class_names)
        }

    def __len__(self):
        return len(self.examples)

    def tokenize_word_and_tag_bioes(self, word, label):
        word = [e for e in word if e["text"].strip() != ""]
        if len(word) == 0:
            return [], [], []

        bboxes = [e["box"] for e in word]
        texts = [e["text"] for e in word]

        word_input_ids = []
        word_bboxes = []
        word_labels = []
        for idx, (bbox, text) in enumerate(zip(bboxes, texts)):
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)

            word_input_ids.append(input_ids)
            word_bboxes.append([bbox for _ in range(len(input_ids))])

            # bioes tagging for known classes (except Other class)
            if label != self.out_class_name:
                if len(word) == 1:  # single text in the word
                    token_labels = ["S_" + label] + [self.pad_token] * (
                        len(input_ids) - 1
                    )
                else:
                    if idx == 0:  # multiple text in the word and first text of that
                        token_labels = ["B_" + label] + [self.pad_token] * (
                            len(input_ids) - 1
                        )
                    elif (
                        idx == len(word) - 1
                    ):  # multiple text in the word and last text of that
                        token_labels = ["E_" + label] + [self.pad_token] * (
                            len(input_ids) - 1
                        )
                    else:  # rest of the text in the word
                        token_labels = ["I_" + label] + [self.pad_token] * (
                            len(input_ids) - 1
                        )
            else:
                token_labels = [label] + [self.pad_token] * (len(input_ids) - 1)
            word_labels.append(token_labels)

        assert len(word_input_ids) > 0
        assert len(word_input_ids) == len(word_bboxes) == len(word_labels)

        return word_input_ids, word_bboxes, word_labels

    def __getitem__(self, idx):
        sample = self.examples[idx]

        word_labels = sample["labels"]
        words = sample["words"]
        assert len(word_labels) == len(words)

        width, height = sample["img"].size
        cls_bbs = [0] * 4  # bbox for first token
        sep_bbs = [width, height] * 2  # bbox for last token

        padded_input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        padded_bboxes = np.zeros((self.max_seq_length, 4), dtype=np.float32)
        padded_labels = np.ones(self.max_seq_length, dtype=int) * -100
        attention_mask = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        are_box_end_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)

        input_ids_list: List[List[int]] = []
        labels_list: List[List[str]] = []
        bboxes_list: List[List[List[int]]] = []
        start_token_indices = []
        end_token_indices = []

        for word_idx, (label, word) in enumerate(zip(word_labels, words)):
            word_input_ids, word_bboxes, word_labels = self.tokenize_word_and_tag_bioes(
                word, label
            )

            if word_input_ids == []:
                continue

            input_ids_list.extend(word_input_ids)
            labels_list.extend(word_labels)
            bboxes_list.extend(word_bboxes)

        tokens_length_list: List[int] = [len(l) for l in input_ids_list]

        # consider [CLS] token that will be added to input_ids, shift "end token indices" 1 to the right
        et_indices = np.array(list(itertools.accumulate(tokens_length_list))) + 1

        # since we subtract original length from shifted indices, "start token indices" are aligned as well
        st_indices = et_indices - np.array(tokens_length_list)

        # last index will be used for [SEP] token
        # to make sure st_indices and end_indices are paired, in case st_indices are cut by max_sequence length,
        st_indices = st_indices[st_indices < self.max_seq_length - 1]
        et_indices = et_indices[et_indices < self.max_seq_length - 1]

        # to make sure st_indices and end_indices are paired, in case st_indices are cut by max_sequence length,
        min_len = min(len(st_indices), len(et_indices))
        st_indices = st_indices[:min_len]
        et_indices = et_indices[:min_len]
        assert len(st_indices) == len(et_indices)

        input_ids: List[int] = list(itertools.chain.from_iterable(input_ids_list))
        bboxes: List[List[int]] = list(itertools.chain.from_iterable(bboxes_list))
        labels: List[str] = list(itertools.chain.from_iterable(labels_list))

        assert len(input_ids) == len(bboxes) == len(labels)

        # CLS, EOS token update for input_ids, labels, bboxes
        input_ids = (
            [self.cls_token_id]
            + input_ids[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(bboxes) == 0:  # When len(json_obj["words"]) == 0 (no OCR result)
            bboxes = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            bboxes = [cls_bbs] + bboxes[: self.max_seq_length - 2] + [sep_bbs]
        bboxes = np.array(bboxes)
        labels = [self.pad_token] + labels[: self.max_seq_length - 2] + [self.pad_token]
        labels = [
            self.bioes_class_name2idx[l]
            if l != self.pad_token
            else self.ignore_label_id
            for l in labels
        ]

        # update ppadded input_ids, labels, bboxes
        len_ori_input_ids = len(input_ids)
        padded_input_ids[:len_ori_input_ids] = input_ids
        padded_labels[:len_ori_input_ids] = np.array(labels)
        attention_mask[:len_ori_input_ids] = 1
        padded_bboxes[:len_ori_input_ids, :] = bboxes

        # expand bbox from [x1, y1, x2, y2] (2points) -> [x1, y1, x2, y1, x2, y2, x1, y2] (4points)
        padded_bboxes = padded_bboxes[:, [0, 1, 2, 1, 2, 3, 0, 3]]

        # Normalize bbox -> 0 ~ 1
        padded_bboxes[:, [0, 2, 4, 6]] = padded_bboxes[:, [0, 2, 4, 6]] / width
        padded_bboxes[:, [1, 3, 5, 7]] = padded_bboxes[:, [1, 3, 5, 7]] / height

        are_box_first_tokens[st_indices] = True
        are_box_end_tokens[et_indices] = True

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
            "labels": padded_labels,
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

        class_id = (label_id - 1) // 4
        is_b_tag = label_id % 4 == 1

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
        self.bioes_class_names = None
        self.tokenizer = tokenizer
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx, *args):
        # unpack batch
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]
        box_first_token_mask = batch["are_box_first_tokens"]
        labels = batch["labels"]

        # inference model
        prediction = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            box_first_token_mask=box_first_token_mask,
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
        gt_labels = batch["labels"]
        labels = batch["labels"]

        # inference model
        prediction = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            box_first_token_mask=are_box_first_tokens,
            labels=labels,
        )

        val_loss = prediction.loss
        pred_labels = torch.argmax(prediction.logits, -1)

        n_batch_gt_classes, n_batch_pred_classes, n_batch_correct_classes = 0, 0, 0
        batch_size = prediction.logits.shape[0]

        for example_idx, (
            pred_label,
            gt_label,
            box_first_token_mask,
            box_end_token_mask,
        ) in enumerate(
            zip(pred_labels, gt_labels, are_box_first_tokens, are_box_end_tokens)
        ):
            # validation loss : # calculate validation loss of "box_first_tokens" only
            valid_gt_label = gt_label[box_first_token_mask]
            valid_pred_label = pred_label[box_first_token_mask]

            gt_parse = parse_from_seq(valid_gt_label, self.class_names)
            pred_parse = parse_from_seq(valid_pred_label, self.class_names)

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

            box_first_token_idx2ori_idx = box_first_token_mask.nonzero(as_tuple=True)[0]
            box2token_span_maps = (
                torch.hstack(
                    (
                        (box_first_token_mask == True).nonzero(),
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

        print("prediction: ...")
        for cls, text in pred_cls2text.items():
            pprint(f"   {cls} : {text}")

        print("gt: ...")
        for cls, text in gt_cls2text.items():
            pprint(f"   {cls} : {text}")

        print("correct: ...")
        for cls, text in correct_cls2text.items():
            pprint(f"   {cls} : {text}")

        step_out = {
            "n_batch_gt_classes": n_batch_gt_classes,
            "n_batch_pr_classes": n_batch_pred_classes,
            "n_batch_correct_classes": n_batch_correct_classes,
        }

        # self.validation_step_outputs.append(step_out)
        self.log_dict({"val_loss": val_loss}, sync_dist=True, prog_bar=True)
        self.log_dict(step_out, sync_dist=True)

        # return
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
    tokenizer = BrosTokenizer.from_pretrained(cfg.tokenizer_path)

    # prepare FUNSD dataset
    train_dataset = FUNSDBIOESDataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        split="train",
    )

    val_dataset = FUNSDBIOESDataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        split="test",
    )

    # make data module & update data_module train and val dataset
    data_module = BROSDataPLModule(cfg)
    data_module.train_dataset = train_dataset
    data_module.val_dataset = val_dataset

    # Load BROS config & pretrained model
    ## update config
    bros_config = BrosConfig.from_pretrained(cfg.model.pretrained_model_name_or_path)
    bioes_class_names = train_dataset.bioes_class_names
    id2label = {idx: name for idx, name in enumerate(bioes_class_names)}
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
    model_module.bioes_class_names = train_dataset.bioes_class_names

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
        filename="bros-funsd-{epoch:02d}-{val_loss:.2f}",
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
        strategy=cfg.train.get("strategy", None),
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
        log_every_n_steps=5,
    )

    trainer.fit(model_module, data_module, ckpt_path=cfg.train.get("ckpt_path", None))


if __name__ == "__main__":
    # load training config
    finetune_funsd_ee_bioes_config = {
        "workspace": "./finetune_funsd_ee_bioes",
        "exp_name": "bros-base-uncased_funsd_bioes_tagging",
        "tokenizer_path": "naver-clova-ocr/bros-base-uncased",
        "dataset": "jinho8345/funsd",
        "task": "ee",
        "seed": 2023,
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
        "model": {
            "pretrained_model_name_or_path": "naver-clova-ocr/bros-base-uncased",
            "max_seq_length": 512,
        },
        "train": {
            "ckpt_path": None,  # or None
            "batch_size": 16,
            "num_samples_per_epoch": 150,
            "max_epochs": 30,
            "use_fp16": True,
            "accelerator": "gpu",
            "strategy": "ddp_find_unused_parameters_true",
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
    cfg = OmegaConf.create(finetune_funsd_ee_bioes_config)
    train(cfg)
