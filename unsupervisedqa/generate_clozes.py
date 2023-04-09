# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Functionality to extract cloze questions from paragraphs of text
"""
# import spacy
import hashlib
from .configs import MIN_ANSWER_CHAR_LEN, MAX_ANSWER_CHAR_LEN,\
    MIN_ANSWER_WORD_LEN, MAX_ANSWER_WORD_LEN, CLOZE_MASKS, MIN_CLOZE_WORD_LEN, MAX_CLOZE_WORD_LEN,\
    MIN_CLOZE_WORDSIZE, MAX_CLOZE_WORDSIZE, MIN_CLOZE_CHAR_LEN, MAX_CLOZE_CHAR_LEN,  \
    MAX_QUESTION_WORDSIZE_THRESHOLD, MAX_PARAGRAPH_WORDSIZE_THRESHOLD, MAX_PARAGRAPH_CHAR_LEN_THRESHOLD, \
    MAX_PARAGRAPH_WORD_LEN_THRESHOLD, MAX_QUESTION_CHAR_LEN_THRESHOLD, MAX_QUESTION_WORD_LEN_THRESHOLD, \
    NOUNPHRASE_LABEL, SPACY_MODEL
from .data_classes import Cloze, Paragraph, Sentence

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer_roberta_base = AutoTokenizer.from_pretrained("cahya/xlm-roberta-base-indonesian-NER")
model_roberta_base = AutoModelForTokenClassification.from_pretrained("cahya/xlm-roberta-base-indonesian-NER")

# nlp = spacy.load(SPACY_MODEL)

def generate_clean_ne(raw_ner):
    # make sure it sorted by index
    raw_ner = sorted(raw_ner, key=lambda x: x['index'])
    
    ner_results = []
    ner_clean = {
        'entity': None,
        'start': None,
        'end': None,
        'word_parts': [],
    }
    
    prev_entity_flag = None
    prev_entity_cat = None
    prev_start = None
    prev_end = None
    
    for i, ne in enumerate(raw_ner):
        current_entity = ne['entity']
        current_entity_flag = ne['entity'][0]
        current_entity_cat = ne['entity'][2:]
        
        current_word = ne['word']
        current_start = ne['start']
        current_end = ne['end']
        
        if i > 0:
            prev_ne = raw_ner[i-1]
            
            prev_entity_flag = prev_ne['entity'][0]
            prev_entity_cat = prev_ne['entity'][2:]
            prev_start = prev_ne['start']
            prev_end = prev_ne['end']
        
        if (current_start == prev_end and current_entity_cat == prev_entity_cat) \
            or (current_entity_flag == 'I' and current_entity_cat == prev_entity_cat):
            ner_clean['word_parts'].append(current_word)
        else:
            if len(ner_clean['word_parts']) > 0:
                ner_clean['end'] = prev_end
                ner_clean['entity'] = prev_entity_cat
                ner_results.append(ner_clean.copy())
            ner_clean = {'entity': None, 'start': None, 'end': None, 'word_parts': []}
            
            # add current to new
            ner_clean['word_parts'].append(current_word)
            ner_clean['start'] = current_start

    # concat words
    for _, ne in enumerate(ner_results):
        ne['word'] = ''.join(ne['word_parts'])
        ne['word'] = ne['word'].replace('▁',' ')
    
    # replace space on first char
    space_char = ' '
    for i, ne in enumerate(ner_results):
        if ne['word'][0] == space_char:
            ne['word'] = ne['word'][1:]
            ne['word_parts'][0] = ne['word_parts'][0][1:]
            if ne['start'] > 0:
                ne['start'] = ne['start']+1
    
    return ner_results


def run_ner(sentence, tokenizer, model):
    pipe = pipeline("ner", model=model, tokenizer=tokenizer)
    ne_raw = pipe(sentence)
    ne_clean = generate_clean_ne(ne_raw)
    return ne_clean


def get_sentence_id(paragraph_text, sentence_text):
    rep = paragraph_text + sentence_text
    return hashlib.sha1(rep.encode()).hexdigest()


def get_sentences(paragraph: str):
    para_split = paragraph.split('.')
    for i, t in enumerate(para_split):
        t = t.strip()
        sent = Sentence(
            sentence_id=get_sentence_id(paragraph, t),
            text=t,
            start_char=sum([len(s) for s in para_split[:i]])
        )
        yield sent


def mask_answer(text, answer_text, answer_start, answer_type):
    before, after = text[:answer_start], text[answer_start + len(answer_text):]
    return before + CLOZE_MASKS[answer_type] + after


def noun_phrase_answer_generator(sent):
    return [(n_p.text, n_p.start_char - sent.start_char, NOUNPHRASE_LABEL) for n_p in sent.noun_chunks]


# def named_entity_answer_generator(sent):
#     return [(e.text, e.start_char - sent.start_char, e.label_) for e in sent.ents]

def named_entity_answer_generator(sent):
    ne_results = run_ner(sent, tokenizer_roberta_base, model_roberta_base)
    return [(e['word'], e['start'], e['entity']) for e in ne_results]


def is_appropriate_cloze(sentence) -> bool:
    good_char_len = MIN_CLOZE_CHAR_LEN < len(sentence) < MAX_CLOZE_CHAR_LEN
    no_links = not (('https://' in sentence) or ('http://' in sentence))
    tokens = sentence.split()
    good_word_lens = all([MIN_CLOZE_WORDSIZE <= len(tok) <= MAX_CLOZE_WORDSIZE for tok in tokens])
    good_num_tokens = MIN_CLOZE_WORD_LEN <= len(tokens) <= MAX_CLOZE_WORD_LEN
    return good_char_len and no_links and good_word_lens and good_num_tokens


def is_appropriate_answer(answer_text) -> bool:
    correct_char_length = MIN_ANSWER_CHAR_LEN <= len(answer_text) <= MAX_ANSWER_CHAR_LEN
    correct_word_length = MIN_ANSWER_WORD_LEN <= len(answer_text.split()) <= MAX_ANSWER_WORD_LEN
    return correct_char_length and correct_word_length


# def is_appropriate_squad_datapoint(question_text, answer_text, paragraph_text):
#     p_char_len_good = len(paragraph_text) <= MAX_PARAGRAPH_CHAR_LEN_THRESHOLD
#     p_word_len_good = len(paragraph_text.split()) <= MAX_PARAGRAPH_WORD_LEN_THRESHOLD
#     p_wordsize_good = all([len(w) <= MAX_PARAGRAPH_WORDSIZE_THRESHOLD for w in paragraph_text.split()])
#     p_good = p_char_len_good and p_word_len_good and p_wordsize_good

#     q_char_len_good = len(question_text) <= MAX_QUESTION_CHAR_LEN_THRESHOLD
#     q_word_len_good = len(question_text.split()) <= MAX_QUESTION_WORD_LEN_THRESHOLD
#     q_wordsize_good = all([len(w) <= MAX_QUESTION_WORDSIZE_THRESHOLD for w in question_text.split()])
#     q_good = q_char_len_good and q_word_len_good and q_wordsize_good

#     a_char_len_good = MIN_ANSWER_CHAR_LEN <= len(answer_text) <= MAX_ANSWER_CHAR_LEN
#     a_word_len_good = MIN_ANSWER_WORD_LEN <= len(answer_text.split()) <= MAX_ANSWER_WORD_LEN
#     a_good = a_char_len_good and a_word_len_good
#     return p_good and q_good and a_good


def is_appropriate_datapoint(question_text, answer_text, paragraph_text):
    p_char_len_good = len(paragraph_text) <= MAX_PARAGRAPH_CHAR_LEN_THRESHOLD
    p_word_len_good = len(paragraph_text.split()) <= MAX_PARAGRAPH_WORD_LEN_THRESHOLD
    p_wordsize_good = all([len(w) <= MAX_PARAGRAPH_WORDSIZE_THRESHOLD for w in paragraph_text.split()])
    p_good = p_char_len_good and p_word_len_good and p_wordsize_good

    q_char_len_good = len(question_text) <= MAX_QUESTION_CHAR_LEN_THRESHOLD
    q_word_len_good = len(question_text.split()) <= MAX_QUESTION_WORD_LEN_THRESHOLD
    q_wordsize_good = all([len(w) <= MAX_QUESTION_WORDSIZE_THRESHOLD for w in question_text.split()])
    q_good = q_char_len_good and q_word_len_good and q_wordsize_good

    a_char_len_good = MIN_ANSWER_CHAR_LEN <= len(answer_text) <= MAX_ANSWER_CHAR_LEN
    a_word_len_good = MIN_ANSWER_WORD_LEN <= len(answer_text.split()) <= MAX_ANSWER_WORD_LEN
    a_good = a_char_len_good and a_word_len_good
    return p_good and q_good and a_good


def get_cloze_id(paragraph_text, sentence_text, answer_text):
    rep = paragraph_text + sentence_text + answer_text
    return hashlib.sha1(rep.encode()).hexdigest()


def generate_clozes_from_paragraph(paragraph, answer_generator):
    # clozes = []
    para_text = paragraph.text
    sentences = get_sentences(para_text)
    for sentence in sentences:
        is_good = is_appropriate_cloze(sentence.text)
        if is_good:
            answers = answer_generator(sentence.text)
            for answer_text, answer_start, answer_type in answers:
                if is_appropriate_answer(answer_text):
                    c = Cloze(
                        cloze_id=get_cloze_id(paragraph.text, sentence.text, answer_text),
                        paragraph=paragraph,
                        source_text=sentence.text,
                        source_start=sentence.start_char,
                        cloze_text=mask_answer(sentence.text, answer_text, answer_start, answer_type),
                        answer_text=answer_text,
                        answer_start=answer_start,
                        constituency_parse=None,
                        root_label=None,
                        answer_type=answer_type,
                        question_text=None
                    )
                    yield c
    # return clozes
