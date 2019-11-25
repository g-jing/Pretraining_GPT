# Pretraining_GPT

input format should be jsonlines file:

["hello", "hi", "how are you?", "I am good", "do you like movies?", "No, I don't"]

Then it will be processed into following format in python:

"A:hello\n\n\nB:hi\n\n\nA:how are you?\n\n\nB:I am good\n\n\nA:do you like movies?\n\n\nB:No, I don't\n\n\n"


Download 30GB data:

https://drive.google.com/open?id=1TC4uliw5QqcL8zeDyfKxwV6E5V0LLTCF


# install requirement 
install jsonlines

```shell
pip install jsonlines
```
Install apex version that modified the error due to cuda version

```shell
git clone https://github.com/qywu/apex

cd apex

pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```


from Qingyang Wu github, install torchfly:

```python
git clone https://github.com/qywu/TorchFly.git

cd TorchFly

pip install -e .

pip install -r requirements.txt
```

install tensorboard

```
pip install tensorboard
```

install transformers from huggingface

```
pip install transformers
```

install allennlp

```
pip install allennlp
```

# test the model during training
Run bot.py and chat with the bot. You could use a different model by changing the model location. Default location is Checkpoint/best.pth

# storage
Model is 500MB, and optimizer is 1000MB.
Total model: ten nearest model_optimiza plus 1 model/4 hours, making it total round 50 GB.

# Retrain
If training is interupted, please find the closest checkpoint and reload model and optimizer. Also, you need to find the rough position of Dataset. Besides, you need to reset the schedule.
### Note: one gpu memory overlead will not lead to training crush(but still not acceptable, for lacking a gpu will leads to a need for a different hyperparameter set). Make sure data all GPU are working correctly at all time.
# Next
1. using hdf5 or other methods to make the model train on limited cpu memory condition
https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5

2. multiwoz, persuasion, camerest

3. double roles training or single roles training

4. ablation study on randomlize start positions

# pretraining problem
1. hyperparameter setting maybe not good. It maybe better to use the hyperparameter from microsoft https://github.com/microsoft/DialoGPT/blob/master/LSP_train.py
they mention lr depends on model size, meaning they tried different lr?

2. data processing more? like deleting n-gram dialogue?

# Experiemnt
We will run on three experiments, persuasionforgood, multiwoz and cameres




