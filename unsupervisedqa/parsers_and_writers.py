# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Module to handle reading, (de)seriaizing and dumping data
"""
from io import TextIOWrapper
import json
import math
from typing import List
import attr
from .data_classes import Cloze, Paragraph
import hashlib


def clozes2squadformat(clozes, out_fobj):
    assert all([c.question_text is not None for c in clozes]), 'Translate these clozes firse, some dont have questions'
    data = {cloze.paragraph.paragraph_id: {'context': cloze.paragraph.text, 'qas': []} for cloze in clozes}
    for cloze in clozes:
        qas = data[cloze.paragraph.paragraph_id]
        qas['qas'].append({
            'question': cloze.question_text, 'id': cloze.cloze_id,
            'answers': [{'text': cloze.answer_text, 'answer_start': cloze.answer_start}]
        })
    squad_dataset = {
        'version': 1.1,
        'data': [{'title': para_id, 'paragraphs': [payload]} for para_id, payload in data.items()]
    }
    json.dump(squad_dataset, out_fobj)


def _parse_attr_obj(cls, serialized):
    return cls(**json.loads(serialized))


def dumps_attr_obj(obj):
    return json.dumps(attr.asdict(obj))


def parse_clozes(fobj):
    for serialized in fobj:
        if serialized.strip('\n') != '':
            yield _parse_attr_obj(Cloze, serialized)


def dump_clozes(clozes, fobj):
    for cloze in clozes:
        fobj.write(dumps_attr_obj(cloze))
        fobj.write('\n')


def _get_paragraph_id(text):
    return hashlib.sha1(text.encode()).hexdigest()

def crop_paragraph(paragraph, max_length):
    # recount max-length, avoid a small paragraph to be generated, divide paragraph equally
    factor = math.ceil(len(paragraph) / max_length)
    new_max_length = len(paragraph) / factor

    all_para_sm = []
    sents = paragraph.split('.')
    para_sm = ''
    for sent in sents:
        if (len(para_sm) + len(sent)) > new_max_length:
            all_para_sm.append(para_sm)
            para_sm = sent
        else:
            if len(para_sm) > 0:
                para_sm += f'.{sent}'
            else:
                para_sm += f'{sent}'
    
    if len(para_sm) > 0:
        all_para_sm.append(para_sm)
        
    return all_para_sm


def parse_paragraphs_from_txt(fobj):
    for paragraph_text in fobj:
        para_text = paragraph_text.strip('\n')
        if para_text != '':
            if len(para_text) > 2000:
                small_paragraphs = crop_paragraph(para_text, 2000)
                for para_sm in small_paragraphs:
                    para_sm = para_sm.strip() # remove whitespace on start and end
                    yield Paragraph(
                        paragraph_id=_get_paragraph_id(para_sm),
                        text=para_sm
                    )
            else:        
                yield Paragraph(
                    paragraph_id=_get_paragraph_id(para_text),
                    text=para_text
                )

# def parse_paragraphs_from_txt(fobj: TextIOWrapper):
#     for paragraph_text in fobj:
#         paragraphs = paragraph_text.split('.')
#         for para_text in paragraphs:
#             para_text = para_text.strip()
#             if para_text != '':
#                 yield Paragraph(
#                     paragraph_id=_get_paragraph_id(para_text),
#                     text=para_text
#                 )


def parse_paragraphs_from_jsonl(fobj):
    for serialized in fobj:
        if serialized.strip('\n') != '':
            yield _parse_attr_obj(Paragraph, serialized)
