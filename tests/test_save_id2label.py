import unittest
from tempfile import TemporaryDirectory
from typing import Dict

from transformers import AutoConfig, AutoModelForTokenClassification, AutoModelWithHeads, BertForSequenceClassification


PATH = "./tmp/"


def get_default(num_label):
    labels = ["LABEL_" + str(i) for i in range(num_label)]
    label_dict = {id: label for id, label in enumerate(labels)}
    return labels, label_dict


class TestSaveLabel(unittest.TestCase):
    def setUp(self):
        self.labels = [
            "ADJ",
            "ADP",
            "ADV",
            "AUX",
            "CCONJ",
            "DET",
            "INTJ",
            "NOUN",
            "NUM",
            "PART",
            "PRON",
            "PROPN",
            "PUNCT",
            "SCONJ",
            "SYM",
            "VERB",
            "X",
        ]
        self.label_map: Dict[int, str] = {i: label for i, label in enumerate(self.labels)}
        self.model_name = "bert-base-uncased"
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.label_map,
            label2id={label: i for i, label in enumerate(self.labels)},
        )

    def test_classification_model_head_labels(self):
        model = AutoModelForTokenClassification.from_pretrained(self.model_name, config=self.config)
        with TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir)
            model.load_head(temp_dir)

        self.assertEqual(self.labels, model.get_labels())
        self.assertDictEqual(self.label_map, model.get_labels_dict())

    def test_sequ_classification_model_head_labels(self):
        model = BertForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        with TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir)
            model.load_head(temp_dir)

        self.assertEqual(self.labels, model.get_labels())
        self.assertDictEqual(self.label_map, model.get_labels_dict())

    def test_model_with_heads_tagging_head_labels(self):
        model = AutoModelWithHeads.from_pretrained(self.model_name, config=self.config)
        model.add_tagging_head("test_head", num_labels=len(self.labels), id2label=self.label_map)
        with TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir, "test_head")
            model.load_head(temp_dir)
        model.load_adapter("pos/ldc2012t13@vblagoje", "text_task")

        self.assertEqual(self.labels, model.get_labels())
        self.assertDictEqual(self.label_map, model.get_labels_dict())

    def test_multiple_heads_label(self):
        model = AutoModelWithHeads.from_pretrained(self.model_name, config=self.config)
        model.add_tagging_head("test_head", num_labels=len(self.labels), id2label=self.label_map)
        with TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir, "test_head")
            model.load_head(temp_dir )
        model.load_adapter("pos/ldc2012t13@vblagoje", "text_task")
        model.add_classification_head("classification_head")
        default_label, default_label_dict = get_default(2)

        self.assertEqual(model.get_labels("classification_head"), default_label)
        self.assertEqual(model.get_labels_dict("classification_head"), default_label_dict)

    def test_model_with_heads_multiple_heads(self):

        model = AutoModelWithHeads.from_pretrained(self.model_name, config=self.config)
        model.add_tagging_head("test_head", num_labels=len(self.labels), id2label=self.label_map)
        model.add_classification_head("second_head", num_labels=5)
        with TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir + "/test_head", "test_head")
            model.load_head(temp_dir + "/test_head")
            model.save_head(temp_dir + "/second_head", "second_head")
            model.load_head(temp_dir + "/second_head")
        model.load_adapter("pos/ldc2012t13@vblagoje", "text_task")

        self.assertEqual(model.get_labels("test_head"), self.labels)
        self.assertEqual(model.get_labels_dict("test_head"), self.label_map)



if __name__ == "__main__":
    unittest.main()
