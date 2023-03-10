# Multi-robot Visual-and-Language Navigation

Codes are publicly available for IROS2023 manuscript -> 

### Extending Visual-and-Language Navigation to Multi-Robot Scenarios

## Requirements
datasets==1.18.3
huggingface-hub==0.12.1
igibson==2.0.3
line-profiler==4.0.2
networkx==2.7.1
tensorboardX==2.6
sentencepiece==0.1.95
tokenizers==0.10.3
tomli==2.0.1
torch==1.12.0
torchaudio==0.12.0
torchvision==0.13.0
transformers==4.12.3

## Usage
The default hyperparameters for the algorithm can be found in ''param.py''.
You can modify the robot num in file ''train.py'' and 'train_joint.py'

# Pre-train Predictor
If you want to pre-train the VLNBERT-based predictor model, use the command:
```shell
mkdir -p snap/VLN-BERT
python train.py --model Vanilla-VLNBERT --optim RMSProp
```

If you want to pre-train the DUET-based predictor model, use the command:
```shell
mkdir -p snap/DUET
python train.py --model DUET --optim AdamW
```
# Joint Training
If you want to train the VLNBERT-based predictor model and controller model, use the command:
```shell
mkdir -p snap/VLN-BERT
python train_joint.py --model VLN-BERT --optim RMSProp
```

If you want to train the DUET-based predictor model and controller model, use the command:
```shell
mkdir -p snap/DUET
python train_joint.py --model DUET --optim AdamW
```
