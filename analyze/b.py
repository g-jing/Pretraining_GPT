import os, glob
import json, string, json_lines

from tqdm import tqdm
import pdb

class Analyze():
    def __init__(self, datafolder, output_folder):
        self._input_file = input_file
        self._output_folder = output_folder

        self.res = {}
        self.res['num_dialogs'] = 0             #
        self.res['length_each_dialog'] = {}     #     
        self.res['num_turns_each_dialog'] = {}  # 
        self.res['num_words_each_dialog'] = {}  #     
        self.res['num_words_each_turn'] = {}    #
        self.res['tokens'] = {}

        self.translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

        self._process()
    
    def _process(self):
        with open(self._input_file, 'rb') as f: # opening file in binary(rb) mode    
            count = 0
            for dialog_text in tqdm(json_lines.reader(f)):
                #print(item)
                #if len(item)>1:
                #   print(item)
                #   pdb.set_trace()
                count += 1
                self.res['num_turns_each_dialog'][len(dialog_text)] = \
                    self.res['num_turns_each_dialog'].get(len(dialog_text), 0) + 1 

                length_cur_dialog = 0
                num_word_cur_dialog = 0
                for cur_sentence in dialog_text:
                    length_cur_dialog += len(cur_sentence)

                    # replace punctation with space
                    cur_sentence = cur_sentence.translate(self.translator)
                    cur_sentence_word = cur_sentence.split(' ')     # ["word1", "word2", "word3", ......]

                    num_words_cur_turn = len(cur_sentence_word)
                    num_word_cur_dialog += num_words_cur_turn
                    self.res['num_words_each_turn'][num_words_cur_turn] =  \
                        self.res['num_words_each_turn'].get(num_words_cur_turn, 0) + 1

                    for each_word in cur_sentence_word:
                        self.res['tokens'][each_word] = self.res['tokens'].get(each_word, 0) + 1

            self.res['num_dialogs'] = count
            self.res['num_words_each_dialog'][num_word_cur_dialog] = \
                    self.res['num_words_each_dialog'].get(num_word_cur_dialog, 0) + 1
            self.res['length_each_dialog'][length_cur_dialog] = \
                    self.res['length_each_dialog'].get(length_cur_dialog, 0) + 1
            

        pdb.set_trace()
        self._save()
        

    def _save(self):
        with open(os.path.join(self._output_folder, 'large.json'), 'w') as f:
            json.dump(self.res, f)



if __name__=="__main__":

    ############################################################
    input_file = '/home/wuqy1203/Desktop/Dataset/Reddit/train.jsonl'
    #datafolder = '/home/ubuntu/cr_data/kinetics400'
    output_folder = '/home/chongruo/analyze/Pretraining_GPT/analyze'
    ############################################################

    create_data = Analyze(input_file, output_folder)
