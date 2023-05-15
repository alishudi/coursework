from torch.cuda import is_available
from torch import no_grad
from string2string.misc import ModelEmbeddings
from string2string.similarity import CosineSimilarity
from sentence_transformers import SentenceTransformer
from model_lists import model_lists

class Cosine_similiarity:
    def __init__(self, model_name):
        self.is_sent = model_name in model_lists['sent']
        if self.is_sent:
            self.emb_model = SentenceTransformer(
                model_name_or_path=model_name,
                device='cuda' if is_available() else 'cpu'
            )
        else:
            self.emb_model = ModelEmbeddings(
                model_name_or_path=model_name,
                device='cuda' if is_available() else 'cpu'
            )
        self.emb_type = 'mean_pooling' #'last_hidden_state'
        self.cosine_similarity = CosineSimilarity()

        #those models has wrong max sequence length assigned on HF
        if model_name in model_lists['512']:
            self.emb_model.tokenizer.model_max_length = 512

        if model_name in model_lists['1024']:
            self.emb_model.tokenizer.model_max_length = 1024

        self.is_encdec = model_name in model_lists['enc_dec']

    def set_emb_type(self, emb_type):
        self.emb_type = emb_type

    def calc_matrix(self, group, url2record):
        n = len(group)

        headlines = [url2record[url]['patched_title'] for url in group]
        texts = [url2record[url]['patched_text'] for url in group]
        
        if self.is_sent:
            headline_embs = self.emb_model.encode(headlines, show_progress_bar=False, convert_to_tensor=True)
            text_embs = self.emb_model.encode(texts, show_progress_bar=False, convert_to_tensor=True)
        elif self.is_encdec:
            headline_embs = self.s_get_embeddings(headlines, embedding_type=self.emb_type)
            text_embs = self.s_get_embeddings(texts, embedding_type=self.emb_type)
        else:
            headline_embs = self.emb_model.get_embeddings(headlines, embedding_type=self.emb_type)
            text_embs = self.emb_model.get_embeddings(texts, embedding_type=self.emb_type)
        headline_embs = headline_embs.repeat_interleave(n, dim=0)
        text_embs = text_embs.repeat(n, 1)

        matrix = self.cosine_similarity.compute(headline_embs, text_embs, dim=1).reshape(n, n)
        return matrix.cpu().numpy()


    def s_get_embeddings(self, text, embedding_type='last_hidden_state'):
            
            """
            Rewritten method from string2string.misc.model_embeddings to work with encoder-decoder models
            """

            if embedding_type not in ['last_hidden_state', 'mean_pooling']:
                raise ValueError(f'Invalid embedding type: {embedding_type}. Only "last_hidden_state" and "mean_pooling" are supported.')

            encoded_text = self.emb_model.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors='pt',
            )

            encoded_text = encoded_text.to(self.emb_model.device)

            with no_grad():
                embeddings = self.emb_model.model(decoder_input_ids=encoded_text['input_ids'][:,:1],
                                                  decoder_attention_mask=encoded_text['attention_mask'][:,:1],
                                                  **encoded_text)

            if embedding_type == 'last_hidden_state':
                embeddings = embeddings.encoder_last_hidden_state[:, 0, :]
            elif embedding_type == 'mean_pooling':
                embeddings = embeddings.encoder_last_hidden_state.mean(dim=1)

            return embeddings