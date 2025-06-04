from tokenizers import ByteLevelBPETokenizer


class Nep_Tok:
    def __init__(self):
        self.tokenizer = ByteLevelBPETokenizer.from_file("model/vocab.json", "model/merges.txt")

    
    def encode(self,text,return_tokens=False,return_encode_obj=False):
        encode_obj = self.tokenizer.encode(text)
        ids=encode_obj.ids
        

        if return_encode_obj:
            return encode_obj    #if encode object is returned no need to return ids and tokens

        if return_tokens:
            tokens=[]
            for id in ids:
                tokens.append(self.tokenizer.decode([id]))
            return ids,tokens
        
        return ids
        
    def decode(self,ids,skip_special_tokens=False,strict=False):
        return self.tokenizer.decode(ids,skip_special_tokens=skip_special_tokens)
    
    def add_tokens(self,tokens_list):
        self.tokenizer.add_special_tokens(tokens_list)
        

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def return_da_tokenizer_object(self):
        return self.tokenizer.copy()
    
tok = Nep_Tok()
a=tok.return_da_tokenizer_object()
tok.add_tokens(['<s>', '</s>'])
print(tok.get_vocab_size())
print(tok.decode(tok.encode('<s> यो एउटा नेपाली टोकनाइजर हो। </s>', return_tokens=False)))
print(a.get_vocab_size())

