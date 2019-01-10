import random
import unittest
import os
from extract import StyloDocument

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test/oliver_twist_fixture.txt')
        self.doc = StyloDocument(self.path)

    def test_is_stylo_doc(self):
        self.assertIsInstance(self.doc, StyloDocument)

    def test_doc_has_author(self):
        unknown_doc = StyloDocument(self.path)
        self.assertEqual(unknown_doc.props["author"],"Unknown")

    def test_doc_with_author(self):
        dickens_doc = StyloDocument(self.path, author="Charles Dickens")
        self.assertEqual(dickens_doc.props["author"],"Charles Dickens")        

    def test_mean_sentence_len(self):
        self.assertAlmostEqual(self.doc.props["mean_sentence_len"], 20.401408450704224)

    def test_mean_word_len(self):
        self.assertAlmostEqual(self.doc.props["mean_word_len"], 6.563677130044843)

    def test_type_token_ration(self):
        self.assertAlmostEqual(self.doc.props["lexical_diversity"], 21.296915289848155)

    def test_csv_output(self):
        out = self.doc.csv_data()
        self.assertTrue("Unknown" in out)
        self.assertTrue("oliver_twist_fixture.txt" in out)
        self.assertTrue("20.4014,19.5202,46.4706" in out)

    def test_csv_header(self):
        out = self.doc.csv_header()
        self.assertTrue('Author,Title,LexicalDiversity' in out)
        self.assertTrue('Mights,This,Verys' in out)

if __name__ == '__main__':
    unittest.main()
