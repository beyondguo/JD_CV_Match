#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/2 13:38
# @Author  : Frances
# @Site    : 
# @File    : util.py.py
# @Software: PyCharm

import os
import json
import shutil
import logging
import pickle
import tensorflow as tf

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def test_ner(results, path):
    """
    Run perl script to evaluate model
    """
    output_file = path + "_predict.utf8"
    with open(output_file, "w", encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir("log"):
        os.makedirs("log")


def clean(params):
    """
    Clean current folder
    remove saved model and training log
    """
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)

    if os.path.isfile(params.map_file):
        os.remove(params.map_file)

    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)

    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)

    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)

    if os.path.isdir("log"):
        shutil.rmtree("log")

    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")

    if os.path.isfile(params.config_file):
        os.remove(params.config_file)

    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)


def save_config(config, config_file):
    """
    Save configuration of the model
    parameters are stored in json format
    """
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    """
    Load configuration of the model
    parameters are stored in json format
    """
    with open(config_file, encoding="utf8") as f:
        return json.load(f)

# 读取dictionary
def load_maps(map_file_path):
    '''
    Load configuration for entity interface
    :param map_file_path: path of map_file
    :return: maps
    '''
    ### TODO:检查文件路径存在与否
    with open(map_file_path, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    f.close()
    return char_to_id, id_to_char, tag_to_id, id_to_tag

def convert_to_text(line):
    """
    Convert conll data to text
    """
    to_print = []
    for item in line:

        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)


def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def create_model(session, Model_class, path, load_vec, config, id_to_char, logger,graph):
    # create model, reuse parameters if exists
    model = Model_class(config,graph)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            emb_weights = session.run(model.char_lookup.read_value())
            emb_weights = load_vec(config["emb_file"],id_to_char, config["char_dim"], emb_weights)
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model

def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

# 返回实体抽取结果
def extract(sentence,tags):
    entities = []
    entity = ""
    chunk_start = False
    chunk_end = False
    for i in range(len(tags)):
        if tags[i][0] == "S":
            entities.append({"value": sentence[i], "start": i, "end": i+1, "type":tags[i][2:]})
        if i==0 and tags[i][0]!='O': chunk_start = True
        if tags[i][0] == 'B':chunk_start = True
        if i>0:
            if tags[i-1] == 'O' and tags[i][0] == 'I': chunk_start = True
            if tags[i - 1][0] == 'B' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'B' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'I' and tags[i][0] == 'S': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'O': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'B': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'I': chunk_end = True
            if tags[i - 1][0] == 'E' and tags[i][0] == 'S': chunk_end = True

        if chunk_end or chunk_start:

            if chunk_end and entity:
                entities[-1]['value'] = entity
                entities[-1]['end'] = i
                chunk_end = False
                entity = ""
            if chunk_start:
                entities.append({'type': tags[i][2:], 'start': i})
                entity = sentence[i]
                chunk_start = False

        elif entity:
            entity+=sentence[i]

        if entity and i+1==len(tags):
            # entity+=sentence[i]
            entities[-1]['value'] = entity
            entities[-1]['end'] = i+1
            chunk_end = False
            entity = ""
    return entities





