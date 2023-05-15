'''
Lists of models which have tokenizers with wrong max_sequence_lenght in huggingface
and list of models with encoder-decoder architecture from which we need to use only encoder part
and models from sentence-transformers  
'''
model_lists = {
    '512': [
        'ai-forever/ruBert-base',
        'ai-forever/ruBert-large',
        'ai-forever/ruRoberta-large',
        'DeepPavlov/rubert-base-cased-sentence',
        'ai-forever/sbert_large_mt_nlu_ru',
        'ai-forever/sbert_large_nlu_ru'
    ],
    '1024': [
        'ai-forever/ruT5-base',
        'ai-forever/ruT5-large',
        'IlyaGusev/rut5_base_sum_gazeta',
        'ai-forever/FRED-T5-large'
    ],
    'enc_dec': [
        'ai-forever/ruT5-base',
        'ai-forever/ruT5-large',
        'IlyaGusev/rut5_base_sum_gazeta',
        'facebook/mbart-large-50',
        'IlyaGusev/mbart_ru_sum_gazeta',
        'Kirili4ik/mbart_ruDialogSum'
    ],
    'sent': [
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'sentence-t5-large',
        'sentence-transformers/stsb-xlm-r-multilingual',
        'sentence-transformers/distiluse-base-multilingual-cased-v2',
        'sentence-transformers/all-mpnet-base-v2'
    ]
}