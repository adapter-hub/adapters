import logging

import click

from transformers.modeling_xlm_roberta import XLMRobertaModel


#
# BUILD A FRANKENSTEIN-BERT by stitching together parts (adapters) of different berts
#

def copy_adapter_weights(source, target):
    # source_bert = source[0].bert
    # target_bert = target[0].bert

    if hasattr(source.config, 'adapters'):
        for task in source.config.adapters:
            logging.root.info('adding {}'.format(task))
            target.add_adapter(task)
            target.config.adapters.append(task)

            source_params = [(k, v) for (k, v) in source.named_parameters() if 'adapters.{}'.format(task) in k]
            target_params = [(k, v) for (k, v) in target.named_parameters() if 'adapters.{}'.format(task) in k]

            for (source_k, source_v), (target_k, target_v) in zip(source_params, target_params):
                assert source_k == target_k
                target_v.data.copy_(source_v.data)
    if hasattr(source.config, 'language_adapters'):
        for language in source.config.language_adapters:
            logging.root.info('adding {}'.format(language))
            target.add_language_adapter(language)
            target.config.language_adapters.append(language)

            source_params = [(k, v) for (k, v) in source.named_parameters() if 'language_adapters.{}'.format(language) in k]
            target_params = [(k, v) for (k, v) in target.named_parameters() if 'language_adapters.{}'.format(language) in k]

            for (source_k, source_v), (target_k, target_v) in zip(source_params, target_params):
                assert source_k == target_k
                target_v.data.copy_(source_v.data)


@click.command()
@click.option('--input_paths', '-i', multiple=True, help='Name of the training task (sts, nli, ...)')
@click.option('--model_save_path', '-o', help='Path of the folder where the trained model should be saved')
def main(input_paths, model_save_path):
    # we start by copying the first model

    # input_paths = ['data/models_final/csqa/', 'data/models_final/sst_glue/']
    # model_save_path = 'data/models_final/16_test/'

    print('Base model: {}'.format(input_paths[0]))
    target = XLMRobertaModel.from_pretrained(input_paths[0])

    # then we copy over the adapters from the other models
    for model_path in input_paths[1:]:
        print('Copying weights: {}'.format(model_path))
        source = XLMRobertaModel.from_pretrained(model_path)
        copy_adapter_weights(source=source, target=target)

    print('DONE. Saving...')
    target.save_pretrained(model_save_path)


if __name__ == '__main__':


    main()
