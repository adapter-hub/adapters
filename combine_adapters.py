import logging

import click

from convert_model import load_model_from_old_format


#
# BUILD A FRANKENSTEIN-BERT by stitching together parts (adapters) of different berts
#

def copy_adapter_weights(source, target):
    # source_bert = source[0].bert
    # target_bert = target[0].bert

    if hasattr(source.config, 'text_task_adapters'):
        for task in source.config.text_task_adapters:
            logging.root.info('adding {}'.format(task))
            target.add_task_adapter(task)
            target.config.text_task_adapters.append(task)

            source_params = [(k, v) for (k, v) in source.named_parameters() if 'adapters.{}'.format(task) in k]
            target_params = [(k, v) for (k, v) in target.named_parameters() if 'adapters.{}'.format(task) in k]

            for (source_k, source_v), (target_k, target_v) in zip(source_params, target_params):
                assert source_k == target_k
                target_v.data.copy_(source_v.data)
    if hasattr(source.config, 'text_lang_adapters'):
        for language in source.config.text_lang_adapters:
            logging.root.info('adding {}'.format(language))
            target.add_language_adapter(language)
            target.config.text_lang_adapters.append(language)

            source_params = [(k, v) for (k, v) in source.named_parameters() if 'text_lang_adapters.{}'.format(language) in k]
            target_params = [(k, v) for (k, v) in target.named_parameters() if 'text_lang_adapters.{}'.format(language) in k]

            for (source_k, source_v), (target_k, target_v) in zip(source_params, target_params):
                assert source_k == target_k
                target_v.data.copy_(source_v.data)


@click.command()
@click.option('--input_paths', '-i', multiple=True, help='Name of the training task (sts, nli, ...)')
@click.option('--model_save_path', '-o', help='Path of the folder where the trained model should be saved')
def main(input_paths, model_save_path):
    # we start by copying the first model

    # input_paths = ['../data/adapters_16_bert_base/csqa/', '../data/adapters_16_bert_base/sst/', '../data/adapters_16_bert_base/multinli/']
    # model_save_path = '../data/adapters_16_bert_base/csqa-multinli-sst/'

    print('Base model: {}'.format(input_paths[0]))
    target = load_model_from_old_format(input_paths[0])

    # then we copy over the adapters from the other models
    for model_path in input_paths[1:]:
        print('Copying weights: {}'.format(model_path))
        source = load_model_from_old_format(model_path)
        copy_adapter_weights(source=source, target=target)

    print('DONE. Saving...')
    target.save_pretrained(model_save_path)


if __name__ == '__main__':


    main()
