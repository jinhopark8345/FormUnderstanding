import pandas as pd
import torch
from datasets import load_dataset
from transformers import BrosForTokenClassification, BrosProcessor

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

# BrosProcessor = BrosProcessor.from_pretrained("naver-clova-ocr/bros-base-uncased")
# model = BrosForTokenClassification.from_pretrained("naver-clova-ocr/bros-base-uncased")

data = load_dataset("jinho8345/funsd")["train"]
# (Pdb++) data[0].keys()
# dict_keys(['img', 'filename', 'boxes', 'labels', 'words', 'linkings', 'ids'])


"""
img, filename -> same
ids -> idk


# entity extraction
linkings -> don't use
boxes, labels, words -> convert to 'word' level

    # case 1 : label : BIOES tagging
    # case 2 : label : initial token tagging & subsequent token tagging

# case 3 : entity linking
labels -> don't use
boxes, labels, linkings -> convert to 'word' level

"""


def entity_to_word(s):
    entity_boxes = s["boxes"]
    entity_labels = s["labels"]
    entities = s["words"]

    words, boxes, labels = [], [], []

    for entity, entity_label in zip(entities, entity_labels):
        cur_texts, cur_boxes = [e["text"] for e in entity], [e["box"] for e in entity]
        for text, box in zip(cur_texts, cur_boxes):
            if text.strip() == "":
                continue

            words.append(text)
            boxes.append(box)
            labels.append(entity_label)

    assert len(words) == len(boxes)
    assert len(words) == len(labels)

    return words, boxes, labels


def entity_to_word_bioes_tagging(s):
    entity_boxes = s["boxes"]
    entity_labels = s["labels"]
    entities = s["words"]

    words, boxes, labels = [], [], []

    for entity, entity_label in zip(entities, entity_labels):
        cur_texts, cur_boxes = [], []
        for e in entity:
            if e["text"].strip() == "":
                continue
            cur_texts.append(e["text"])
            cur_boxes.append(e["box"])

        for idx, (text, box) in enumerate(zip(cur_texts, cur_boxes)):
            # handle "other" class
            if entity_label == "other":
                labels.append("O")
            else:
                # handle meaningful classes' single token entity
                if len(cur_texts) == 1:
                    labels.append("S_" + entity_label)

                # handle meaningful classes' mulitple token entity
                else:
                    if idx == 0:
                        labels.append("B_" + entity_label)
                    elif idx == len(cur_texts) - 1:
                        labels.append("E_" + entity_label)
                    else:
                        labels.append("I_" + entity_label)

            words.append(text)
            boxes.append(box)

    assert len(words) == len(boxes)
    assert len(words) == len(labels)

    return words, boxes, labels


def entity_to_word_bio_tagging(s):
    entity_boxes = s["boxes"]
    entity_labels = s["labels"]
    entities = s["words"]

    words, boxes, labels = [], [], []

    for entity, entity_label in zip(entities, entity_labels):
        cur_texts, cur_boxes = [], []
        for e in entity:
            if e["text"].strip() == "":
                continue
            cur_texts.append(e["text"])
            cur_boxes.append(e["box"])

        for idx, (text, box) in enumerate(zip(cur_texts, cur_boxes)):
            # handle "other" class
            if entity_label == "other":
                labels.append("O")
            else:
                # handle meaningful classes
                if idx == 0:
                    labels.append("B_" + entity_label)
                else:
                    labels.append("I_" + entity_label)

            words.append(text)
            boxes.append(box)

    assert len(words) == len(boxes)
    assert len(words) == len(labels)

    return words, boxes, labels


def entity_to_word_entity_intra_relation(s):
    entity_boxes = s["boxes"]
    entity_labels = s["labels"]
    entities = s["words"]

    words, boxes = [], []
    start_word_labels, subsequent_word_labels = [], []

    word_cnt = 0
    for entity, entity_label in zip(entities, entity_labels):
        cur_texts, cur_boxes = [], []

        for e in entity:
            if e["text"].strip() == "":
                continue
            cur_texts.append(e["text"])
            cur_boxes.append(e["box"])

        for cur_idx, (text, box) in enumerate(zip(cur_texts, cur_boxes)):
            words.append(text)
            boxes.append(box)
            start_word_labels.append("other")  # default value
            subsequent_word_labels.append(-1)  # default index

            if entity_label != "other":
                print(entity_label)
                if cur_idx == 0:
                    # for start_word_labels
                    start_word_labels[word_cnt] = entity_label
                else:
                    start_word_labels[word_cnt] = "PAD_" + entity_label
                    subsequent_word_labels[word_cnt - 1] = word_cnt

            word_cnt += 1

    assert len(words) == len(boxes)
    assert len(words) == len(start_word_labels)
    assert len(words) == len(subsequent_word_labels)

    return words, boxes, start_word_labels, subsequent_word_labels


def entity_to_word_entity_entity_relation(s):
    entity_boxes = s["boxes"]
    entity_labels = s["labels"]
    entity_linkings = s["linkings"]
    entities = s["words"]

    """
    entity_idx -> first_word_idx


    from_entity_idx2to_entity_idx -> from_word_idx2to_word_idx but both from_word and to_word label shouldn't be 'other'

    """

    words, boxes, word_labels = [], [], []
    entity_entity_labels = []

    entity_idx2first_word_idx = {}
    for entity_idx, (entity, entity_label) in enumerate(zip(entities, entity_labels)):
        cur_texts, cur_boxes = [], []
        for e in entity:
            if e["text"].strip() == "":
                continue
            cur_texts.append(e["text"])
            cur_boxes.append(e["box"])

        if cur_texts:
            entity_idx2first_word_idx[entity_idx] = len(words)
        else:
            entity_idx2first_word_idx[entity_idx] = -1

        for text, box in zip(cur_texts, cur_boxes):
            words.append(text)
            boxes.append(box)
            word_labels.append(entity_label)

    entity_entity_labels = [-1 for _ in range(len(words))]

    for links in entity_linkings:
        for link in links:
            from_entity_idx, to_entity_idx = link

            from_word_idx = entity_idx2first_word_idx[from_entity_idx]
            to_word_idx = entity_idx2first_word_idx[to_entity_idx]

            if from_word_idx == -1 or to_word_idx == -1:
                continue

            from_word_label = word_labels[from_word_idx]
            to_word_label = word_labels[to_word_idx]

            if from_word_label == "other" or to_word_label == "other":
                continue

            entity_entity_labels[from_word_idx] = to_word_idx

    return words, boxes, entity_entity_labels



if __name__ == '__main__':

    # words, boxes, labels = entity_to_word(data[0]) # for simple token classification
    # words, boxes, labels = entity_to_word_bio_tagging(data[0]) # for bio tagging token classification
    # words, boxes, labels = entity_to_word_bioes_tagging(data[0]) # for bioes tagging token classification

    (
        ee_words,
        ee_boxes,
        ee_start_word_labels,
        ee_subsequent_word_labels,
    ) = entity_to_word_entity_intra_relation(data[0])

    el_words, el_boxes, el_labels = entity_to_word_entity_entity_relation(data[0])

    df = pd.DataFrame(
        {
            "words": ee_words,
            "start_word_labels": ee_start_word_labels,
            "subsequent_word_labels": ee_subsequent_word_labels,
            "ee_labels": el_labels,
        }
    )

    print(df)
