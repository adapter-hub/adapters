# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" XNLI utils (dataset loading and evaluation) """


import logging
import os
import pandas as pd
from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available
from typing import List, Optional, Union
if is_tf_available():
    import tensorflow as tf
from ...tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

def conceptnet_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    return _conceptnet_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )
def _conceptnet_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = conceptnet_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = conceptnet_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

class ConceptnetProcessor(DataProcessor):
    """Processor for the Conceptnet dataset."""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("train", i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(InputExample(guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("test", i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(InputExample(guid, text_a=text_a, text_b=text_b, label=label))
        return examples


    def get_labels(self):
        """See base class."""
        return ['Antonym', 'AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'CreatedBy', 'DefinedAs',
         'DerivedFrom', 'Desires', 'DistinctFrom', 'Entails', 'EtymologicallyRelatedTo', 'FormOf', 'HasA',
          'HasContext', 'HasFirstSubevent', 'HasLastSubevent', 'HasPrerequisite', 'HasProperty', 'HasSubevent',
           'InstanceOf', 'IsA', 'LocatedNear', 'MadeOf', 'MannerOf', 'MotivatedByGoal', 'NotCapableOf', 'NotDesires',
            'NotHasProperty', 'PartOf', 'ReceivesAction', 'RelatedTo', 'SimilarTo', 'SymbolOf', 'Synonym', 'UsedFor',
             'dbpedia/capital', 'dbpedia/field', 'dbpedia/genre', 'dbpedia/genus', 'dbpedia/influencedBy', 'dbpedia/knownFor',
              'dbpedia/language', 'dbpedia/leader', 'dbpedia/occupation', 'dbpedia/product']


conceptnet_processors = {
    "conceptnet": ConceptnetProcessor,
}

conceptnet_output_modes = {
    "conceptnet": "classification",
}

conceptnet_tasks_num_labels = {
    "conceptnet": 46,
}
