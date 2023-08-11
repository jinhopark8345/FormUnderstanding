import itertools
from dataclasses import dataclass
from pprint import pprint

import numpy as np
import torch
from bros import BrosConfig, BrosTokenizer
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


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


@dataclass
class CFG:
    dataset: str
    max_seq_length: dict
    tokenizer_path: str


if __name__ == "__main__":
    cfg = CFG("jinho8345/funsd", 512, "naver-clova-ocr/bros-base-uncased")

    # Load Tokenizer (going to be used in dataset to to convert texts to input_ids)
    tokenizer = BrosTokenizer.from_pretrained(cfg.tokenizer_path)

    # prepare SROIE dataset
    train_dataset = FUNSDBIOESDataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        split="train",
    )

    sample1 = train_dataset[0]
