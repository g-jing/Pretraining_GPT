# Pretraining_GPT

input format should be jsonlines file:

["hello", "hi", "how are you?", "I am good", "do you like movies?", "No, I don't"]

Then it will be processed into following format in python:

"A:hello\n\n\nB:hi\n\n\nA:how are you?\n\n\nB:I am good\n\n\nA:do you like movies?\n\n\nB:No, I don't\n\n\n"


Download 30GB data:

https://drive.google.com/open?id=1TC4uliw5QqcL8zeDyfKxwV6E5V0LLTCF


# install requirement 
install jsonlines

```pip install jsonlines```

install apex version that modified the error due to cuda version

`git clone https://github.com/qywu/apex`

`cd apex`

```pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./```


from Qingyang Wu github, install torchfly:

```git clone https://github.com/qywu/TorchFly.git```

```cd TorchFly```

`pip install -e .`

`pip install -r requirements.txt`

install tnesorboard

`pip install tensorboard`

install transformers from huggingface

`pip install transformers`

install allennlp

`pip install allennlp`

# test the model during training
Run bot.py and chat with the bot. You could use a different model by changing the model location. Default location is Checkpoint/best.pth

# Retrain
If training is interupted, please find the closest checkpoint and reload model and optimizer. Also, you need to find the rough position of Dataset. Besides, you need to reset the schedule.


