from tokenizers import ByteLevelBPETokenizer


class Nep_Tok:
    def __init__(self):
        tokenizer=ByteLevelBPETokenizer()
        self.tokenizer = tokenizer.from_file("model/vocab.json", "model/merges.txt")

    
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
        
    def decode(self,ids):
        return self.tokenizer.decode(ids)
    
    def add_tokens(self,tokens_list):
        self.tokenizer.add_special_tokens(tokens_list)
        



tok=Nep_Tok()
tok.add_tokens(['<s>','</s>'])
print(tok.encode('<s> यो एउटा नेपाली टोकनाइजर हो। </s>',return_tokens=True))
