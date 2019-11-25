import csv
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math, re, argparse
import json
import functools
import pickle

en_sws = set(stopwords.words())
wn = WordNetLemmatizer()

order_to_number = {
    'first': 1, 'one': 1, 'seco': 2, 'two': 2, 'third': 3, 'three': 3, 'four': 4, 'forth': 4, 'five': 5, 'fifth': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nin': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
}

# load data
with open("/home/qingyang/Desktop/Dialog/TaskOriented/camrest676/data/CamRest676/CamRest676.json") as f:
    raw_data = json.loads(f.read().lower())
    
with open("/home/qingyang/Desktop/Dialog/TaskOriented/camrest676/data/CamRest676/CamRestOTGY.json", "r") as f:
    raw_entities = json.load(f)

# get entities
entity_dict = {}
entities = []

for k in raw_entities['informable']:
    entities.extend(raw_entities['informable'][k])
    for item in raw_entities['informable'][k]:
        entity_dict[item] = k



def clean_sentence(s, entity_dict):
    s = s.replace('<go> ', '').replace(' SLOT', '_SLOT')
    s = '<GO> ' + s + ' </s>'
    for item in entity_dict:
        # s = s.replace(item, 'VALUE_{}'.format(self.entity_dict[item]))
        s = clean_replace(s, item, '{}_SLOT'.format(entity_dict[item]))
    return s

def clean_replace(s, r, t, forward=True, backward=False):
    def clean_replace_single(s, r, t, forward, backward, sidx=0):
        idx = s[sidx:].find(r)
        if idx == -1:
            return s, -1
        idx += sidx
        idx_r = idx + len(r)
        if backward:
            while idx > 0 and s[idx - 1]:
                idx -= 1
        elif idx > 0 and s[idx - 1] != ' ':
            return s, -1

        if forward:
            while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                idx_r += 1
        elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
            return s, -1
        return s[:idx] + t + s[idx_r:], idx_r

    sidx = 0
    while sidx != -1:
        s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
    return s


def similar(a,b):
    return a == b or a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]
    #return a == b or b.endswith(a) or a.endswith(b)    

def setsub(a,b):
    junks_a = []
    useless_constraint = ['temperature','week','est ','quick','reminder','near']
    for i in a:
        flg = False
        for j in b:
            if similar(i,j):
                flg = True
        if not flg:
            junks_a.append(i)
    for junk in junks_a:
        flg = False
        for item in useless_constraint:
            if item in junk:
                flg = True
        if not flg:
            return False
    return True

def setsim(a,b):
    a,b = set(a),set(b)
    return setsub(a,b) and setsub(b,a)



class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


def bleu_metric(data):
    gen, truth = [],[]
    for row in data:
        gen.append(row['generated_response'])
        truth.append(row['response'])
    wrap_generated = [[_] for _ in gen]
    wrap_truth = [[_] for _ in truth]
    sc = BLEUScorer().score(zip(wrap_generated, wrap_truth))
    return sc


def pack_dial(data):
    dials = {}
    for turn in data:
        dial_id = int(turn['dial_id'])
        if dial_id not in dials:
            dials[dial_id] = []
        dials[dial_id].append(turn)
    return dials


def extract_constraint(z, entiteis):
    z = z.split()
    
    if 'EOS_Z1' not in z:
        s = set(z)
    else:
        idx = z.index('EOS_Z1')
        s = set(z[:idx])
        
    # post-process
    if 'moderately' in s:
        s.discard('moderately')
        s.add('moderate')

    return s.intersection(entities)


def success_f1_metric(data):
    dials = pack_dial(data)
    
    tp, fp, fn = 0, 0, 0
    
    for dial_id in dials:
        truth_req, gen_req = set(), set()
        dial = dials[dial_id]
        
        # retrieve
        for turn_num, turn in enumerate(dial):
            gen_response_token = turn['generated_response'].split()
            response_token = turn['response'].split()
            
            for idx, w in enumerate(gen_response_token):
                if w.endswith('SLOT') and w != 'SLOT':
                    gen_req.add(w.split('_')[0])
                    
            for idx, w in enumerate(response_token):
                if w.endswith('SLOT') and w != 'SLOT':
                    truth_req.add(w.split('_')[0])

        # ignore name slots
        gen_req.discard('name')
        truth_req.discard('name')
        
        # true postive and false positive
        for req in gen_req:
            if req in truth_req:
                tp += 1
            else:
                fp += 1
                
        for req in truth_req:
            if req not in gen_req:
                fn += 1
    precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def match_metric(data, entities):
    dials = pack_dial(data)
    match, total = 0, 1e-8
    # no point of using success here!
    # success = 0
    
    for dial_id in dials:
        truth_req, gen_req = [], []
        dial = dials[dial_id]
        gen_bspan, truth_cons, gen_cons = None, None, set()
        truth_response_req = []
        
        for turn_num, turn in enumerate(dial):
            # check generated response Belief Span 
            if 'SLOT' in turn['generated_response']:
                gen_bspan = turn['generated_bspan']
                gen_cons = extract_constraint(gen_bspan, entities)
    
            # check response Belief Span 
            if 'SLOT' in turn['response']:
                truth_cons = extract_constraint(turn['bspan'], entities)
                
            gen_response_token = turn['generated_response'].split()
            response_token = turn['response'].split()
            
            # Useless code
#             for idx, w in enumerate(gen_response_token):
#                 if w.endswith('SLOT') and w != 'SLOT':
#                     gen_req.append(w.split('_')[0])
# #                 if w == 'SLOT' and idx != 0:
# #                     gen_req.append(gen_response_token[idx - 1])
                    
#             for idx, w in enumerate(response_token):
#                 if w.endswith('SLOT') and w != 'SLOT':
#                     truth_response_req.append(w.split('_')[0])
        
        # if no slot from all generated responses
        if not gen_cons:
            gen_bspan = dial[-1]['generated_bspan']
            gen_cons = extract_constraint(gen_bspan, entities)
            
        if truth_cons:
            if gen_cons == truth_cons:
                match += 1
            total += 1

    return match / total