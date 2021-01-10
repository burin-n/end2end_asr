import codecs


class Alphabet(object):
    # config file is a path to alphabets.txt
    def __init__(self, config_file):
        self._config_file = config_file
        self._label_to_str = ["ϵ"]
        self._str_to_label = {"ϵ": 0}
        self._size = 1
        with codecs.open(config_file, 'r', 'utf-8') as fin:
            for line in fin:
                if line[0:2] == '\\#':
                    line = '#\n'
                elif line[0] == '#':
                    continue
                self._label_to_str += line[:-1]  # remove the line ending
                self._str_to_label[line[:-1]] = self._size
                self._size += 1


    def string_from_label(self, label):
        return self._label_to_str[label]


    def label_from_string(self, string):
        return self._str_to_label[string]


    def decode(self, labels):
        res = ''
        for label in labels:
            res += self.string_from_label(label)
        return res


    def size(self):
        return self._size


    def config_file(self):
        return self._config_file
   
   
    def encode(self, strings):
        res = []
        for ch in strings:
            res.append(self._str_to_label[ch])
        return res

            
if __name__ == '__main__':
    alphabet = Alphabet('/root/th007_tl_no_cv/thai_eng_alphabet.txt')
    for ch in alphabet._label_to_str:
        print('ก' + ch) 
        if(ch == " "): print("SPACEE")
        if(ch == '\n'): print("NEWLINE")