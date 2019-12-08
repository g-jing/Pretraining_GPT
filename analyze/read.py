import json
import pdb

with open('small.json') as json_file:
    data = json.load(json_file)

    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> # dialogues')
    print(data['num_dialogs'])


    # length
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> # length')
    length_dict = data['length_each_dialog'] 

    length_sum = 0
    temp = []
    for each_length, how_many in length_dict.items():
        #print(each_length, how_many)
        temp.append( [each_length, how_many] )
        length_sum += int(str(each_length)) * how_many
    temp = sorted(temp, key=lambda x: -x[1])
    print( length_sum / data['num_dialogs'] )
        

    # num of turns
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> # turns')

    turns_dict = data['num_turns_each_dialog']
    #print(data['num_turns_each_dialog'])
    turns_sum = 0
    temp = []
    for each_turn, how_many in turns_dict.items():
        temp.append( [each_turn, how_many] )
        turns_sum += (int(str(each_turn))/2.0) * how_many
    temp = sorted(temp, key=lambda x: -x[1])
    print( turns_sum / data['num_dialogs'] )


    # num of words in each dialog
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> # num of words in each dialog')
    words_dict = data['num_words_each_dialog']
    print(words_dict)
    temp_sum = 0
    temp = []
    for each_turn, how_many in words_dict.items():
        temp.append( [each_turn, how_many] )
        temp_sum += int(str(each_turn)) * how_many
    temp = sorted(temp, key=lambda x: -x[1])
    print( temp_sum / data['num_dialogs'] )
    pdb.set_trace()

    # num of words in each dialog
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> # num of words')
    print( len(data['tokens'].keys()) )

    


