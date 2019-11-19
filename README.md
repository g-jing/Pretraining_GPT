# Pretraining_GPT
test
1. Preprocess ["hello", "hi", "how are you?", "I am good", "do you like movies?", "No, I don't"].

The result becomes: "A:hello\n\n\nB:hi\n\n\nA:how are you?\n\n\nB:I am good\n\n\nA:do you like movies?\n\n\nB:No, I don't\n\n\n"


Download 30GB data:

https://drive.google.com/open?id=1TC4uliw5QqcL8zeDyfKxwV6E5V0LLTCF


# install apex
# modified the error due to cuda version
git clone https://github.com/qywu/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# install requirement 
1. install jsonlines: pip install jsonlines
2. from Qingyang Wu github, install torchfly:
git clone https://github.com/qywu/TorchFly.git
cd TorchFly
pip install -e .
pip install -r requirements.txt

install tnesorboardX
pip install tensorboardX

install transformers from huggingface
pip install transformers



# using tokenizer.decode to check back


