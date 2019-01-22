from __future__ import division
import csv
import io
import pytz
import glob

import nltk
from nltk import sent_tokenize, word_tokenize, Text
from nltk.probability import FreqDist
import numpy as np
import random

DEFAULT_AUTHOR = "Unknown"

def type_token_ratio(text):
    return (len(set(text)) / len(text)) * 100

def mean_word_len(tokens):
    words = set(tokens)
    word_chars = [len(word) for word in words]
    return sum(word_chars) /  float(len(word_chars))

def mean_sentence_len(sentence_word_length):
    return np.mean(sentence_word_length)

def std_sentence_len(sentence_word_length):
    return np.std(sentence_word_length)

def term_per_thousand(term, fdist):
    """
    term       X
    -----  = ------
      N       1000
    """
    return (fdist[term] * 1000) / fdist.N()


def read_from(fileobj_or_filename):
    try:
        with open(fileobj_or_filename, "rb") as f:
            return fileobj_or_filename, f.read().decode(encoding='utf-8', errors='ignore')
    except TypeError:
        # Assuming that we're opened in text mode
        return getattr(fileobj_or_filename, "name", "unknown"), fileobj_or_filename.read()


class StyloDocument(object):

    def __init__(self, fileobj_or_filename, author=DEFAULT_AUTHOR):
        file_name, doc = read_from(fileobj_or_filename)
        author = author

        tokens = word_tokenize(doc)
        text = Text(tokens)
        fdist = FreqDist(text)
        sentences = sent_tokenize(doc)
        sentence_chars = [len(s) for s in sentences]
        sentence_word_length = [len(s.split()) for s in sentences]
        paragraphs = [p for p in doc.split("\n\n") if len(p) > 0 and not p.isspace()]
        paragraph_word_length = [len(p.split()) for p in paragraphs]

        self.props = {
            "author": author,
            "title": file_name,
            "lexical_diversity": type_token_ratio(text),
            "mean_word_len": mean_word_len(tokens),
            "mean_sentence_len": np.mean(sentence_word_length),
            "std_sentence_len": np.mean(sentence_word_length),
            "mean_paragraph_len": np.mean(paragraph_word_length),
            "document_len": sum(sentence_chars),
            "commas": term_per_thousand(',', fdist),
            "semicolons": term_per_thousand(';', fdist),
            "quotes": term_per_thousand('"', fdist),
            "exclamations": term_per_thousand('!', fdist),
            "colons": term_per_thousand(':', fdist),
            "dashes": term_per_thousand('-', fdist),
            "mdashes": term_per_thousand('--', fdist),
            "ands": term_per_thousand('and', fdist),
            "buts": term_per_thousand('but', fdist),
            "howevers": term_per_thousand('however', fdist),
            "ifs": term_per_thousand('if', fdist),
            "thats": term_per_thousand('that', fdist),
            "mores": term_per_thousand('more', fdist),
            "musts": term_per_thousand('must', fdist),
            "mights": term_per_thousand('might', fdist),
            "this": term_per_thousand('this', fdist),
            "verys": term_per_thousand('very', fdist)
        }

    def csv_output(self):
        output = io.StringIO()
        csv.writer(output).writerow([
            self.props[k] for k in sorted(self.props.keys())
        ])
        return output.getvalue().strip('\r\n')

    def csv_header(self):
        return (
            ','.join(sorted(self.props.keys()))
        )


    def text_output(self):
        print("##############################################")
        print("")
        print("Name: ", self.props["title"])
        print("")
        print(">>> Phraseology Analysis <<<")
        print("")
        print("Lexical diversity        :", self.props["lexical_diversity"])
        print("Mean Word Length         :", self.props["mean_word_len"])
        print("Mean Sentence Length     :", self.props["mean_sentence_len"])
        print("STDEV Sentence Length    :", self.props["std_sentence_len"])
        print("Mean paragraph Length    :", self.props["mean_paragraph_len"])
        print("Document Length          :", self.props["document_len"])
        print("")
        print(">>> Punctuation Analysis (per 1000 tokens) <<<")
        print("")
        print('Commas                   :', self.props["commas"])
        print('Semicolons               :', self.props["semicolons"])
        print('Quotations               :', self.props["quotes"])
        print('Exclamations             :', self.props["exclamations"])
        print('Colons                   :', self.props["colons"])
        print('Hyphens                  :', self.props["dashes"]) # m-dash or n-dash?
        print('Double Hyphens           :', self.props["mdashes"]) # m-dash or n-dash?
        print("")
        print(">>> Lexical Usage Analysis (per 1000 tokens) <<<")
        print("")
        print('and                      :', self.props['ands'])
        print('but                      :', self.props['buts'])
        print('however                  :', self.props['howevers'])
        print('if                       :', self.props['ifs'])
        print('that                     :', self.props['thats'])
        print('more                     :', self.props['mores'])
        print('must                     :', self.props['musts'])
        print('might                    :', self.props['mights'])
        print('this                     :', self.props['this'])
        print('very                     :', self.props['verys'])
        print('')



class StyloCorpus(object):

    
    def __init__(self,documents_by_author):
        self.documents_by_author = documents_by_author

    @classmethod
    def from_path_list(cls, path_list, author=DEFAULT_AUTHOR):
        stylodoc_list = cls.convert_paths_to_stylodocs(path_list)
        documents_by_author = {author:stylodoc_list}
        return cls(documents_by_author)

    @classmethod
    def from_stylodoc_list(cls, stylodoc_list, author=DEFAULT_AUTHOR):
        author = DEFAULT_AUTHOR
        documents_by_author = {author:stylodoc_list}
        return cls(documents_by_author)

    @classmethod
    def from_documents_by_author(cls, documents_by_author):
        return cls(documents_by_author)

    @classmethod
    def from_paths_by_author(cls, paths_by_author):
        documents_by_author = {}
        for author, path_list in paths_by_author.iteritems():
            documents_by_author[author] = cls.convert_paths_to_stylodocs(path_list,author)
        return cls(documents_by_author)

    @classmethod
    def from_glob_pattern(cls, pattern):
        documents_by_author = {}
        if isinstance(pattern,list):
            for p in pattern:
                documents_by_author.update(cls.get_dictionary_from_glob(p))
        else:
            documents_by_author = cls.get_dictionary_from_glob(pattern)
        return cls(documents_by_author)

    @classmethod
    def convert_paths_to_stylodocs(cls, path_list, author=DEFAULT_AUTHOR):
        stylodoc_list = []
        for path in path_list:
            sd = StyloDocument(path, author)
            stylodoc_list.append(sd)
        return stylodoc_list

    @classmethod
    def get_dictionary_from_glob(cls, pattern):
        documents_by_author = {}
        for path in glob.glob(pattern):
            author = path.split('/')[-2]
            document = StyloDocument(path, author)
            if author not in documents_by_author:
                documents_by_author[author] = [document]
            else:
                documents_by_author[author].append(document)
        return documents_by_author

    def csv_data(self, author=None):
        csv_data = self.csv_header() + '\n'
        if not author:
            for a in self.documents_by_author.keys():
                for doc in self.documents_by_author[a]:
                    csv_data += doc.csv_output() + '\n'
        else:
            for doc in self.documents_by_author[author]:
                csv_data += doc.csv_output() + '\n'

        return csv_data

    def output_csv(self, out_file, author=None):
        csv_data = self.csv_data()
        if out_file:
            with open(out_file,'w') as f:
                f.write(csv_data)
        return csv_data

