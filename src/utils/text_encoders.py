from pytorch_pretrained_bert import BertTokenizer
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_EOS_INDEX, DEFAULT_UNKNOWN_INDEX
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#Doc torchnlp
#https://pytorchnlp.readthedocs.io/_/downloads/en/latest/pdf/
import torch
# BertTokenizer reserved tokens: "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"


class MyBertTokenizer(BertTokenizer):
    """ Patch of pytorch_pretrained_bert.BertTokenizer to fit torchnlp TextEncoder() interface. """

    def __init__(self, vocab_file, do_lower_case=True, append_eos=False):
        # eos : end of sentence : add token for end of sentence at the end of the sentence
        super().__init__(vocab_file, do_lower_case=do_lower_case)
        self.append_eos = append_eos

        self.itos = list(self.vocab.keys()) #itos probablement index to string
        self.stoi = {token: index for index, token in enumerate(self.itos)}

        self.vocab = self.itos
        self.vocab_size = len(self.vocab)

    def encode(self, text, eos_index=DEFAULT_EOS_INDEX, unknown_index=DEFAULT_UNKNOWN_INDEX):
        """ Returns a :class:`torch.LongTensor` encoding of the `text`. """
        text = self.tokenize(text)
        unknown_index = self.stoi['[UNK]']  # overwrite unknown_index according to BertTokenizer vocab
        vector = [self.stoi.get(token, unknown_index) for token in text]
        if self.append_eos:
            vector.append(eos_index)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        """ Given a :class:`torch.Tensor`, returns a :class:`str` representing the decoded text.
        Note that, depending on the tokenization method, the decoded version is not guaranteed to be
        the original text.
        """
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
