import json
import re
import pickle
import argparse
import os
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from functools import partial
from itertools import islice


def seek_to_line(f, n):
    if n > 0:
        for _ in islice(f, n - 1):
            pass


PADDING = "<PAD>"
PADDING_INT = 0
SEP_INT = -1
SEP = "<SEP>"

text_to_int_dct = {PADDING: PADDING_INT, SEP: SEP_INT}

total_ent_len = 0
total_doc_len = 0


def decode_json(file=None, max_entity_length=60, filter_file='filter.xlsx', abbr_file='SemanticTypes_2018AB.txt'):
    filter_data = None
    abbr_dct = dict()
    with open(abbr_file, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            abbr_dct[line.split('|')[0]] = line.split('|')[2]

    with open(file, mode='r') as f:
        content = json.load(f)
    # print(len(content))
    output = list()
    concepts = dict()
    semantic_set = set()
    doc_set = set()
    ent_len = 0
    if pubmed_format <= 0:
        content = [content]
    for doc in content:
        if pubmed_format > 0:
            doc_id = doc['docID']
        if pubmed_format > 0 and doc_id not in filtered_doc_dct:
            continue
        # doc_title = doc['docTitle']
        # doc_abstract = doc['docAbstract']
        doc_entities = list()
        if pubmed_format > 0:
            doc_set.add(doc_id)
        entities = doc['entities']
        for entity in entities:
            for e in entity['evaluations']:
                concept_id = e['conceptID']
                if pubmed_format > 0:
                    concept = e['concept']
                else:
                    concept = e['conceptPreferredName']
                semantics = e['semantics']
                if filter_data is not None:
                    hit = False
                    for semantic in semantics:
                        if semantic in filter_data:
                            hit = True
                            break
                    if not hit:
                        continue
                semantic_set.update(semantics)
                concepts[concept_id] = (concept, '&&'.join([abbr_dct[s] for s in semantics]))
                doc_entities.append(concept_id)
                # for semantic in semantics:
                #     # TODO
                #     semantic_entity_dict[semantic].append(concept_id)
        ent_len += len(doc_entities)
        splits = int(np.ceil(len(doc_entities) / max_entity_length))
        for i in range(splits):
            if i == splits - 1 and len(doc_entities) // max_entity_length != splits:
                left_ent = doc_entities[i * max_entity_length:]
                output.append(left_ent + [PADDING] * (max_entity_length - len(left_ent)))
            else:
                output.append(doc_entities[i * max_entity_length: i * max_entity_length + max_entity_length])
        output.append([SEP])
    return output, concepts, semantic_set, doc_set, ent_len


def text_to_int(entities, text_to_int_dct):
    entities_int = list()
    for ent_seq in entities:
        ent_seq_int = list()
        for ent in ent_seq:
            ent_seq_int.append(text_to_int_dct.get(ent, PADDING_INT))
        entities_int.append(ent_seq_int)
    return entities_int


def generate_data(files=['./data/ace_entities.txt', './data/tmprss2_entities.txt'],
                  output_file='./data/input_file.pkl', entity_encode_file='',
                  abbr_file='SemanticTypes_2018AB.txt', max_entity_length=60, n_split=10,
                  n_parallel=1):
    global total_doc_len
    global total_ent_len
    concepts = dict()
    entities = list()
    semantics = set()
    docs = set()

    num_process = n_parallel

    partial_fn = partial(decode_json, max_entity_length=max_entity_length, abbr_file=abbr_file)
    result = Parallel(num_process)(delayed(partial_fn)(file=file) for file in files)

    for ent_seq, cons, semantic_set, doc_set, ent_len in result:
        total_ent_len += ent_len
        concepts.update(cons)
        entities.extend(ent_seq)
        semantics.update(semantic_set)
        docs.update(doc_set)

    total_doc_len += len(docs)
    print("semantics: ", len(semantics))
    print("总共文献数: ", total_doc_len)
    print("总实体数: ", total_ent_len)
    concept_ids = [PADDING]
    concept_texts = [PADDING]
    semantics = [PADDING]
    for k, v in concepts.items():
        concept_ids.append(str(k))
        concept_texts.append(str(v[0]))
        semantics.append(str(v[1]))
    print("实体id数:", len(set(concept_ids)) - 1, "实体数:", len(set(concept_texts)) - 1)

    from collections import Counter
    counter = Counter(concept_texts).most_common(n=(len(set(concept_texts)) - len(set(concept_ids))))
    print("counter: ", counter)
    for k, v in concepts.items():
        if str(v[0]) in dict(counter):
            print("重复k,v: ", k, v)

    ent_dct_length = len(text_to_int_dct)
    # 保证全局统一编码
    ent_encode = list()
    for item in concept_ids:
        if item not in text_to_int_dct:
            # 忽略SEP
            text_to_int_dct[item] = ent_dct_length - 1
            ent_dct_length += 1
        ent_encode.append(text_to_int_dct.get(item))

    np.savetxt(entity_encode_file, list(zip(concept_ids, concept_texts, semantics, ent_encode)),
               fmt="%s", delimiter='\t')

    entities = text_to_int(entities, text_to_int_dct)
    with open(output_file, 'wb') as f:
        pickle.dump(entities, f)

    lengths = [len(item) for item in entities]
    if len(lengths) > 0:
        print("统计:", len(lengths), np.max(lengths), np.min(lengths), np.mean(lengths), np.argmax(np.bincount(lengths)))

    # import matplotlib.pyplot as plt
    #
    # plt.plot(np.bincount(lengths))
    # plt.show()


def filter_year_doc(doc_file, start_year, end_year, n_split, n_parallel):
    with open(doc_file, 'rb') as f:
        count = 0
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
    print("doc json lines: ", count)
    m = count // n_split
    n = count % n_split
    slices = []
    for i in range(n_split):
        slices.append((i * m, (i + 1) * m if i != n_split - 1 else (i + 1) * m + n + 1))

    def func_read(begin, end, doc_file=None):
        res = []
        with open(doc_file, mode='r') as f:
            seek_to_line(f, begin)
            line = f.readline()
            pos = begin + 1
            while line:
                try:
                    line = line.strip(',\n')
                    group = re.match('"(.*)":.*"pubYear":(.*?),', line)
                    if end_year >= int(group[2]) >= start_year:
                        res.append((group[1], group[2]))
                    res.append((group[1], group[2]))
                except Exception as e:
                    print("error: ", e)
                if pos < end:
                    line = f.readline()
                    pos += 1
                else:
                    break
        return res

    func_partial = partial(func_read, doc_file=doc_file)
    data = Parallel(n_parallel)(delayed(func_partial)(b, e) for b, e in slices)
    doc_dct = {}
    for item in data:
        doc_dct.update(dict(item))
    return doc_dct


def split_data(n, length):
    slices = []
    m = length // n
    k = length % n
    e = 0
    partition = []
    for i in range(n):
        if i < k:
            b = i * (m + 1)
            e = (i + 1) * (m + 1)
        else:
            b = e
            e = b + m
            if e > length:
                e = length
        slices.append((b, e))
        partition.extend([i] * (e - b))
    return slices, partition


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='.', type=str)
    parser.add_argument('--output_file', default='model_input_file.pkl', type=str, help='pickle file')
    parser.add_argument('--entity_encode_file', default='entity_encode_file.txt', type=str)
    parser.add_argument('--abbr_file', default='SemanticTypes_2018AB.txt', type=str)
    parser.add_argument('--doc_file', default='../data/doc_infos.json', type=str)
    parser.add_argument('--maxlen_seqs', default=10, type=int, help="max length of input event")
    parser.add_argument('--n_split', default=10, type=int, help="n_split")
    parser.add_argument('--n_parallel', default=cpu_count(), type=int, help="n_parallel")
    parser.add_argument('--start_year', default=2015, type=int, help="start_year")
    parser.add_argument('--end_year', default=2015, type=int, help="end_year")
    parser.add_argument('--pubmed_format', default=0, type=int, help="pubmed_format")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_file = args.output_file
    entity_encode_file = args.entity_encode_file
    max_entity_length = args.maxlen_seqs
    abbr_file = args.abbr_file
    n_split = args.n_split
    n_parallel = args.n_parallel
    doc_file = args.doc_file
    start_year = args.start_year
    end_year = args.end_year
    pubmed_format = args.pubmed_format

    # 获取>=start_year的文档
    if pubmed_format > 0:
        filtered_doc_dct = filter_year_doc(doc_file, start_year, end_year, n_split, n_parallel)
        print("doc_dct size:", len(filtered_doc_dct))

    files = os.listdir(input_dir)
    files_all = [os.path.abspath(os.path.join(input_dir, file)) for file in files if not file.startswith('.')]
    # 分拆处理实体
    length = len(files_all)
    slices, _ = split_data(n_split, length + 1)
    for i, (b, e) in enumerate(slices):
        print("idx:", i, b, e)
        if pubmed_format > 0:
            files = [file for file in files_all if
                     not file.startswith('.') and re.match(".*pubmed20n(\d+)\.xml", file) is not None and b < int(
                         re.match(".*pubmed20n(\d+)\.xml", file).group(1)) <= e]
        else:
            files = files_all

        for file in [output_file, entity_encode_file]:
            directory = os.path.dirname(os.path.abspath(file))
            if not os.path.exists(directory):
                os.mkdir(directory)
        generate_data(files=files, output_file=output_file + '.{}'.format(i),
                      entity_encode_file=entity_encode_file + '.{}'.format(i),
                      max_entity_length=max_entity_length, abbr_file=abbr_file, n_parallel=n_parallel)
    # 合并n_split个结果
    entity_encode = list()
    unique_ids = set()
    for i in range(n_split):
        with open(entity_encode_file + '.{}'.format(i), mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                if line[0] not in unique_ids:
                    # 同一个concept_id的类别会不一样？
                    entity_encode.append(line)
                    unique_ids.add(line[0])
    entity_encode = sorted(entity_encode, key=lambda x: int(x[-1]))
    np.savetxt(entity_encode_file, entity_encode, fmt="%s", delimiter='\t')
    print("去重实体个数:", len(unique_ids))
