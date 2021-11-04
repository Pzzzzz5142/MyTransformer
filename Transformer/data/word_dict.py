class WordDict(object):
    def __init__(self, word_dict_path) -> None:
        super().__init__()

        with open(word_dict_path, "r") as f:
            self.vocab_size = int(f.readline().strip())
            self.word_dict = {}
            for line in f.readlines():
                word, ind = line.strip().split()
                ind = int(ind)
                self.word_dict[word] = ind
                if word[0] == "<":
                    sp_tokens = word[1:-1]
                    if sp_tokens == "pad":
                        self.padding_idx = ind
                    elif sp_tokens == "unk":
                        self.unknown_idx = ind
                    elif sp_tokens == "bos":
                        self.bos_idx = ind
                    elif sp_tokens == "eos":
                        self.eos_idx = ind

