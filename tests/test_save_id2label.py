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
        self.dir = TemporaryDirectory()

    def test_classification_model_head_labels(self):
        model_name = "bert-base-uncased"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.label_map,
            label2id={label: i for i, label in enumerate(self.labels)},
        )
        model_saved = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
        model_saved.save_head(self.dir.name + "/classification")

        config = AutoConfig.from_pretrained(model_name, num_labels=len(self.labels))
        model_loaded = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
        # model_loaded.load_adapter("pos/ldc2012t13@vblagoje", "text_task")
        model_loaded.load_head(self.dir.name + "/classification")
        self.assertEqual(model_loaded.get_labels(), self.labels)
        self.assertEqual(model_loaded.get_labels_dict(), self.label_map)

    def test_sequ_classification_model_head_labels(self):
        model_name = "bert-base-uncased"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.label_map,
            label2id={label: i for i, label in enumerate(self.labels)},
        )
        model = BertForSequenceClassification.from_pretrained(model_name, config=config)
        with TemporaryDirectory() as temp_dir:
            model.save_head(temp_dir)
            model.load_head(temp_dir)

        self.assertEqual(self.labels, model.get_labels())
        self.assertDictEqual(self.label_map, model.get_labels_dict())


    def test_model_with_heads_tagging_head_labels(self):
        model_name = "bert-base-uncased"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.label_map,
            label2id={label: i for i, label in enumerate(self.labels)},
        )
        model_saved = AutoModelWithHeads.from_pretrained(model_name, config=config)
        model_saved.add_tagging_head("test_head", num_labels=len(self.labels), id2label=self.label_map)
        model_saved.save_head(self.dir.name + "/model_with_heads", "test_head")

        config = AutoConfig.from_pretrained(model_name)
        model_loaded = AutoModelWithHeads.from_pretrained(model_name, config=config)
        model_loaded.load_head(self.dir.name + "/model_with_heads")
        model_loaded.load_adapter("pos/ldc2012t13@vblagoje", "text_task")

        model_loaded.add_classification_head("second_head")

        self.assertEqual(model_loaded.get_labels("test_head"), self.labels)
        self.assertEqual(model_loaded.get_labels_dict("test_head"), self.label_map)

        default_label, default_label_dict = get_default(2)

        self.assertEqual(model_loaded.get_labels("second_head"), default_label)
        self.assertEqual(model_loaded.get_labels_dict("second_head"), default_label_dict)

    def test_model_with_heads_multiple_heads(self):
        model_name = "bert-base-uncased"
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.label_map,
            label2id={label: i for i, label in enumerate(self.labels)},
        )
        model_saved = AutoModelWithHeads.from_pretrained(model_name, config=config)
        model_saved.add_tagging_head("test_head", num_labels=len(self.labels), id2label=self.label_map)
        model_saved.save_head(self.dir.name + "/model_with_heads", "test_head")

        config = AutoConfig.from_pretrained(model_name)
        model_loaded = AutoModelWithHeads.from_pretrained(model_name, config=config)
        model_loaded.load_head(self.dir.name + "/model_with_heads")
        model_loaded.load_adapter("pos/ldc2012t13@vblagoje", "text_task")

        self.assertEqual(model_loaded.get_labels("test_head"), self.labels)
        self.assertEqual(model_loaded.get_labels_dict("test_head"), self.label_map)

    def tearDown(self):
        self.dir.cleanup()


if __name__ == "__main__":
    unittest.main()
