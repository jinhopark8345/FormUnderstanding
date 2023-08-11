import itertools
from dataclasses import dataclass
from pprint import pprint

import numpy as np
import torch
from bros import BrosConfig, BrosTokenizer
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


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
        assert len(word_labels) == len(words)

        width, height = sample["img"].size
        cls_bbs = [0] * 4  # bbox for first token
        sep_bbs = [width, height] * 2  # bbox for last token

        padded_input_ids = np.ones(self.max_seq_length, dtype=int) * self.pad_token_id
        padded_bboxes = np.zeros((self.max_seq_length, 4), dtype=np.float32)
        attention_mask = np.zeros(self.max_seq_length, dtype=int)
        are_box_first_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)
        are_box_end_tokens = np.zeros(self.max_seq_length, dtype=np.bool_)

        itc_labels = np.zeros(self.max_seq_length, dtype=int)
        stc_labels = np.ones(self.max_seq_length, dtype=np.int64) * self.max_seq_length

        input_ids_list: List[List[int]] = []
        labels_list: List[List[str]] = []
        bboxes_list: List[List[List[int]]] = []
        start_token_indices = []
        end_token_indices = []

        # filter out word with empty text
        word_and_label_list = []
        for word, label in zip(words, word_labels):
            cur_word_and_label = []
            for e in word:
                if e["text"].strip() != "":
                    cur_word_and_label.append(e)
            if cur_word_and_label:
                word_and_label_list.append((cur_word_and_label, label))

        # save classes_dic
        text_box_idx = 0
        input_ids = []
        bboxes = []
        box_to_token_indices = []
        cum_token_idx = 0
        label2text_box_indices_list = {cls_name: [] for cls_name in self.class_names}
        for word, label in word_and_label_list:
            text_box_indices = []
            for text_and_box in word:
                text_box_indices.append(text_box_idx)
                text_box_idx += 1

                text, box = text_and_box["text"], text_and_box["box"]
                this_box_token_indices = []

                if text.strip() == "":
                    continue

                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                input_ids += tokens
                bb = [box for _ in range(len(tokens))]
                bboxes += bb

                for _ in tokens:
                    cum_token_idx += 1
                    this_box_token_indices.append(cum_token_idx)

                box_to_token_indices.append(this_box_token_indices)

            label2text_box_indices_list[label].append(text_box_indices)

        tokens_length_list: List[int] = [len(l) for l in label2text_box_indices_list]

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

        # make itc(initial token), stc (sequence token) labels
        for class_name in self.class_names:
            if class_name == self.out_class_name:
                continue
            if class_name not in label2text_box_indices_list:
                continue

            for word_list in label2text_box_indices_list[class_name]:
                is_first, last_word_idx = True, -1
                for word_idx in word_list:
                    if word_idx >= len(box_to_token_indices):
                        break
                    box2token_list = box_to_token_indices[word_idx]
                    for converted_word_idx in box2token_list:
                        if converted_word_idx >= self.max_seq_length:
                            break  # out of idx

                        if is_first:
                            itc_labels[converted_word_idx] = self.class_idx_dic[
                                class_name
                            ]
                            is_first, last_word_idx = False, converted_word_idx
                        else:
                            stc_labels[converted_word_idx] = last_word_idx
                            last_word_idx = converted_word_idx

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

        return_dict = {
            "input_ids": padded_input_ids,
            "bbox": padded_bboxes,
            "attention_mask": attention_mask,
            "are_box_first_tokens": are_box_first_tokens,
            "are_box_end_tokens": are_box_end_tokens,
            "itc_labels": itc_labels,
            "stc_labels": stc_labels,
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
    train_dataset = FUNSDSpadeDataset(
        dataset=cfg.dataset,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        split="test",
    )

    sample1 = train_dataset[0]
