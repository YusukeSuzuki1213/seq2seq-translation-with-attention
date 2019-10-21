from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
    )

# 読み込んだデータを保持しておくクラス
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {} # word -> index e.g. {"SOS": 0, "EOS": 1}
        self.word2count = {} # how many word appear e.g. {"SOS": 15, "EOS": 12}
        self.index2word = {0: "SOS", 1: "EOS"} # index -> word
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def filterPair(p):
    # “I am” or “He is” etc. に”変換される”文だけでフィルタ
    do_filter = True if len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes) else False
    return do_filter

# 最大長10語（末尾の句読点を含む）にフィルタ
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s) 
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) 
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # データを1行ずつリストとして読み込む
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Type: List[List[str]], e.g. [['i am a hero', '私はヒーロです'], ['i am a man', '私は男です'], ...]
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines] 
    
    # lang2 -> lang1
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    # lang1 -> lang2
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs