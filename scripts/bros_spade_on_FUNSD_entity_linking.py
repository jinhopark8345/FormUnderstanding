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
    BrosForTokenClassificationWithSpade,
    BrosTokenizer,
)

from datasets import load_dataset, load_from_disk

torch.set_printoptions(threshold=2000000)


"""

Entity Linking (EL) task is a task linking between entities.
For example,  "DATE" text is connected to "12/12/2019" text.

    "DATE" -> tokens : ["DA", "TE"], input_ids : [3312, 5123]
    "12/12/2019" -> tokens : ["12", "/", "12", "/", "20", "19"], input_ids : [12, 13, 12, 13, 1123, 777]

    here you are training a model to connect "DA" token with "12" token

However, in Entity Extraction (EE) task is a task to extract entities
from sequence of text.  Often times, Entities consist of multiple
words and each word consists of multiple tokens.  So in EE task, you
are finding entities by finding corressponding tokens (or connecting
corresponding tokens that belong to each entity)

"""

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

class FUNSDSpadeDataset(Dataset):
    """FUNSD BIOES tagging Dataset

    FUNSD : Form Understanding in Noisy Scanned Documents

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

        self.class_names = ["other", "header", "question", "answer"]
        self.out_class_name = "other"
        self.class_idx_dic = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }
        self.pad_token = self.tokenizer.pad_token
        self.ignore_label_id = -100

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sample = self.examples[idx]

        word_labels = sample["labels"]
        words = sample["words"]
        linkings = sample["linkings"]
        assert len(word_labels) == len(words)

        width, height = sample["img"].size
        cls_bbs = [0] * 4  # bbox for first token
        sep_bbs = [width, height] * 2  # bbox for last token

        # make placeholders
        padded_input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        padded_bboxes = np.zeros((self.max_seq_length, 4), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        are_box_end_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        itc_labels = np.zeros(self.max_seq_length, dtype=int)
        stc_labels = np.ones(self.max_seq_length, dtype=np.int64) * self.max_seq_length
        el_labels = np.ones(self.max_seq_length, dtype=int) * self.max_seq_length

        # convert linkings from "word_idx to word_idx" to "text_box to text_box"
        from_text_box2to_text_box = {}
        for linking in linkings:
            if not linking:
                continue
            from_word_idx, to_word_idx = linking[0]
            from_text_box, to_text_box = words[from_word_idx][0], words[to_word_idx][0]
            from_text_box = tuple([from_text_box["text"], tuple(from_text_box["box"])])
            to_text_box = tuple([to_text_box["text"], tuple(to_text_box["box"])])
            from_text_box2to_text_box[from_text_box] = to_text_box

        """
        in the beginning, words are like below

            [
                [{'box': [147, 148, 213, 168], 'text': 'Attorney'},
                    {'box': [216, 151, 275, 168], 'text': 'General'},
                    {'box': [148, 172, 187, 190], 'text': 'Betty'},
                    {'box': [191, 169, 206, 187], 'text': 'D.'},
                    {'box': [211, 170, 305, 191], 'text': 'Montgomery'}],
                [{'box': [275, 249, 377, 267], 'text': 'CONFIDENTIAL'},
                    {'box': [380, 250, 457, 267], 'text': 'FACSIMILE'},
                    {'box': [264, 267, 369, 281], 'text': 'TRANSMISSION'},
                    {'box': [369, 267, 422, 281], 'text': 'COVER'},
                    {'box': [420, 267, 467, 281], 'text': 'SHEET'}],
                [{'box': [352, 297, 383, 314], 'text': '(614)'},
                    {'box': [384, 296, 405, 313], 'text': '466-'},
                    {'box': [406, 297, 438, 312], 'text': '5087'}]
                ...
            ]

        and "words" and "word_labels" are synchronized based on their indices

        1. filter out "text_box" with emtpy text
        2. convert words into input_ids and bboxes
            2-1. convert word into "list of text_box"
                2-1-1. convert "text_box" into "list of tokens"

        in result we will have,

            - input_ids & bboxes are synchronized based on their indices

            - text_box_idx2token_indices : List[List[int]]:
                    -> text_box_idx to token_indices (of corressponding text)

            - label2text_box_indices_list : Dict[str, List[List[int]]]
                    -> list of text_box_indices belong to each label (class_name) mapping

            - text_box2text_box_idx : Dict[tuple, int]
                    -> tuple value of text_box to text_box_idx mapping,
                       going to use with "from_text_box2to_text_box" (came from converting linking gt)
                       to get linkings between text_box_indices

        """

        # 1. filter out "text_box" with emtpy text
        word_and_label_list = []
        for word, label in zip(words, word_labels):
            cur_word_and_label = []
            for e in word:
                if e["text"].strip() != "":
                    cur_word_and_label.append(e)
            if cur_word_and_label:
                word_and_label_list.append((cur_word_and_label, label))


        # 2. convert words into input_ids and bboxes
        text_box_idx = 0
        cum_token_idx = 0
        input_ids = []
        bboxes = []
        text_box_idx2token_indices = []
        label2text_box_indices_list = {cls_name: [] for cls_name in self.class_names}
        text_box2text_box_idx = {}
        for word_idx, (word, label) in enumerate(word_and_label_list):
            text_box_indices = []

            # 2-1. convert word into "list of text_box"
            for text_and_box in word:
                text_box_indices.append(text_box_idx)

                text, box = text_and_box["text"], text_and_box["box"]
                text_box2text_box_idx[tuple([text, tuple(box)])] = text_box_idx
                this_text_box_token_indices = []

                if text.strip() == "":
                    continue

                # 2-1-1. convert "text_box" into "list of tokens"
                this_input_ids = self.tokenizer.encode(text, add_special_tokens=False)
                input_ids += this_input_ids
                this_bboxes = [box for _ in range(len(this_input_ids))]
                bboxes += this_bboxes

                for _ in this_input_ids:
                    cum_token_idx += 1
                    this_text_box_token_indices.append(cum_token_idx)

                text_box_idx2token_indices.append(this_text_box_token_indices)
                text_box_idx += 1

            label2text_box_indices_list[label].append(text_box_indices)
        tokens_length_list: List[int] = [len(l) for l in label2text_box_indices_list]

        # convert linkings from "text_box to text_box" to "text_box idx to text_box idx"
        from_text_box_idx2to_text_box_idx = {
            text_box2text_box_idx[from_text_box]: text_box2text_box_idx[to_text_box]
            for from_text_box, to_text_box in from_text_box2to_text_box.items()
        }

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

        are_box_first_tokens[st_indices] = True
        are_box_end_tokens[et_indices] = True

        # from_text_box_idx2to_text_box_idx = {k-1: v-1 for k, v in from_text_box_idx2to_text_box_idx.items()}
        for from_idx, to_idx in from_text_box_idx2to_text_box_idx.items():

            if from_idx >= len(text_box_idx2token_indices) or to_idx >= len(text_box_idx2token_indices):
                continue

            if (
                text_box_idx2token_indices[from_idx][0] >= self.max_seq_length
                or text_box_idx2token_indices[to_idx][0] >= self.max_seq_length
            ):
                continue

            word_from = text_box_idx2token_indices[from_idx][0]
            word_to = text_box_idx2token_indices[to_idx][0]
            el_labels[word_to] = word_from


        # For [CLS] and [SEP]
        input_ids = (
            [self.cls_token_id]
            + input_ids[: self.max_seq_length - 2]
            + [self.sep_token_id]
        )
        if len(bboxes) == 0:
            # When len(json_obj["words"]) == 0 (no OCR result)
            bboxes = [cls_bbs] + [sep_bbs]
        else:  # len(list_bbs) > 0
            bboxes = [cls_bbs] + bboxes[: self.max_seq_length - 2] + [sep_bbs]
        bboxes = np.array(bboxes)

        # update ppadded input_ids, labels, bboxes
        len_ori_input_ids = len(input_ids)
        padded_input_ids[:len_ori_input_ids] = input_ids
        # padded_labels[:len_ori_input_ids] = np.array(labels)
        attention_mask[:len_ori_input_ids] = 1
        padded_bboxes[:len_ori_input_ids, :] = bboxes

        # expand bbox from [x1, y1, x2, y2] (2points) -> [x1, y1, x2, y1, x2, y2, x1, y2] (4points)
        padded_bboxes = padded_bboxes[:, [0, 1, 2, 1, 2, 3, 0, 3]]

        # Normalize bbox -> 0 ~ 1
        padded_bboxes[:, [0, 2, 4, 6]] = padded_bboxes[:, [0, 2, 4, 6]] / width
        padded_bboxes[:, [1, 3, 5, 7]] = padded_bboxes[:, [1, 3, 5, 7]] / height

        # convert to tensor
        padded_input_ids = torch.from_numpy(padded_input_ids)
        padded_bboxes = torch.from_numpy(padded_bboxes)
        attention_mask = torch.from_numpy(attention_mask)
        are_box_first_tokens = torch.from_numpy(are_box_first_tokens)
        are_box_end_tokens = torch.from_numpy(are_box_end_tokens)
        itc_labels = torch.from_numpy(itc_labels)
        stc_labels = torch.from_numpy(stc_labels)
        el_labels = torch.from_numpy(el_labels)

        return_dict = {
            "input_ids": padded_input_ids,
            "bbox": padded_bboxes,
            "attention_mask": attention_mask,
            "are_box_first_tokens": are_box_first_tokens,
            "are_box_end_tokens": are_box_end_tokens,
            "el_labels": el_labels,
            "itc_labels": itc_labels,
            "stc_labels": stc_labels,
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
        self.tokenizer = tokenizer
        self.dummy_idx = None

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx, *args):
        # unpack batch
        input_ids = batch["input_ids"]
        bbox = batch["bbox"]
        attention_mask = batch["attention_mask"]
        are_box_first_tokens = batch["are_box_first_tokens"]
        # itc_labels = batch["itc_labels"]
        # stc_labels = batch["stc_labels"]
        el_labels = batch["el_labels"]

        # inference model
        prediction = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            itc_mask=are_box_first_tokens,
            itc_labels=itc_labels,
            stc_labels=stc_labels,
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
        itc_labels = batch["itc_labels"]
        stc_labels = batch["stc_labels"]

        # inference model
        prediction = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            itc_mask=are_box_first_tokens,
            itc_labels=itc_labels,
            stc_labels=stc_labels,
        )

        self.log_dict({"val_loss": prediction.loss}, sync_dist=True, prog_bar=True)

        return prediction.loss
        # pr_itc_labels = torch.argmax(prediction.itc_logits, -1)
        # pr_stc_labels = torch.argmax(prediction.stc_logits, -1)

        # (
        #     n_batch_gt_classes,
        #     n_batch_pr_classes,
        #     n_batch_correct_classes,
        # ) = eval_ee_spade_batch(
        #     pr_itc_labels,
        #     itc_labels,
        #     are_box_first_tokens,
        #     pr_stc_labels,
        #     stc_labels,
        #     attention_mask,
        #     self.class_names,
        #     self.dummy_idx,
        # )

        # step_out = {
        #     "loss": prediction.loss,
        #     "n_batch_gt_classes": n_batch_gt_classes,
        #     "n_batch_pr_classes": n_batch_pr_classes,
        #     "n_batch_correct_classes": n_batch_correct_classes,
        # }
        # self.validation_step_outputs.append(step_out)

        # return step_out

    # def on_validation_epoch_end(self):
    #     all_preds = self.validation_step_outputs

    #     n_total_gt_classes, n_total_pr_classes, n_total_correct_classes = 0, 0, 0

    #     for step_out in all_preds:
    #         n_total_gt_classes += step_out["n_batch_gt_classes"]
    #         n_total_pr_classes += step_out["n_batch_pr_classes"]
    #         n_total_correct_classes += step_out["n_batch_correct_classes"]

    #     precision = (
    #         0.0
    #         if n_total_pr_classes == 0
    #         else n_total_correct_classes / n_total_pr_classes
    #     )
    #     recall = (
    #         0.0
    #         if n_total_gt_classes == 0
    #         else n_total_correct_classes / n_total_gt_classes
    #     )
    #     f1 = (
    #         0.0
    #         if recall * precision == 0
    #         else 2.0 * recall * precision / (recall + precision)
    #     )

    #     self.log_dict(
    #         {
    #             "precision": precision,
    #             "recall": recall,
    #             "f1": f1,
    #         },
    #         sync_dist=True,
    #     )

    #     self.validation_step_outputs.clear()

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
    train_dataset = FUNSDSpadeDataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.max_seq_length,
        split="train",
    )

    val_dataset = FUNSDSpadeDataset(
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

    bros_config.id2label = {
        i: label for i, label in enumerate(train_dataset.class_names)
    }
    bros_config.label2id = {
        label: i for i, label in enumerate(train_dataset.class_names)
    }
    bros_config.num_labels = len(train_dataset.class_names)

    ## load pretrained model
    bros_model = BrosForTokenClassificationWithSpade.from_pretrained(
        cfg.model.pretrained_model_name_or_path, config=bros_config
    )

    # model module setting
    model_module = BROSModelPLModule(cfg, tokenizer=tokenizer)
    model_module.model = bros_model
    model_module.class_names = train_dataset.class_names
    model_module.dummy_idx = cfg.model.max_seq_length

    # model_module.bioes_class_names = train_dataset.bioes_class_names

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
        mode="min",
        save_top_k=1,  # if you save more than 1 model,
        # then checkpoint and huggingface model are not guaranteed to be matching
        # because we are saving with huggingface model with save_pretrained method
        # in "on_save_checkpoint" method in "BROSModelPLModule"
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
        log_every_n_steps=1,
    )

    trainer.fit(model_module, data_module, ckpt_path=cfg.train.get("ckpt_path", None))


if __name__ == "__main__":
    # load training config
    finetune_funsd_ee_bioes_config = {
        "workspace": "./finetune_funsd_el_spade__bros-base-uncased",
        "exp_name": "finetune_funsd_el_spade__bros-base-uncased_1",
        "tokenizer_path": "naver-clova-ocr/bros-base-uncased",
        "dataset": "jinho8345/funsd",
        "task": "el",
        "seed": 1,
        "cudnn_deterministic": False,
        "cudnn_benchmark": True,
        "model": {
            "pretrained_model_name_or_path": "naver-clova-ocr/bros-base-uncased",
            "max_seq_length": 512,
        },
        "train": {
            "ckpt_path": None,  # or None
            "batch_size": 1,
            "num_samples_per_epoch": 149,
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
        "val": {"batch_size": 1, "num_workers": 8, "limit_val_batches": 1.0},
    }

    # convert dictionary to omegaconf and update config
    cfg = OmegaConf.create(finetune_funsd_ee_bioes_config)
    train(cfg)
