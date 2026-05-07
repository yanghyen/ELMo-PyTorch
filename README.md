# bilm-pytorch
PyTorch implementation of the pretrained biLM used to compute ELMo
representations from ["Deep contextualized word representations"](http://arxiv.org/abs/1802.05365).

This repository supports both training biLMs and using pre-trained models for prediction.

**Note**: This is a PyTorch conversion of the original TensorFlow implementation. The original TensorFlow version is available at [bilm-tf](https://github.com/allenai/bilm-tf).

Citation:

```
@inproceedings{Peters:2018,
  author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  title={Deep contextualized word representations},
  booktitle={Proc. of NAACL},
  year={2018}
}
```

## Installing

Install Python version 3.8 or later, PyTorch version 2.0 or later, and other dependencies:

```bash
pip install -r requirements.txt
python setup.py install
```

Or install directly with pip:
```bash
pip install torch>=2.0.0 numpy>=1.21.0 h5py>=3.0.0 tqdm>=4.60.0
python setup.py install
```

Ensure the tests pass in your environment by running:
```bash
python test_pytorch_conversion.py
```

## Using pre-trained models

We have several different English language pre-trained biLMs available for use.
Each model is specified with two separate files, a JSON formatted "options"
file with hyperparameters and a hdf5 formatted file with the model
weights. Links to the pre-trained models are available [here](https://allennlp.org/elmo).

**Note**: Pre-trained weights from the original TensorFlow version need to be converted for use with this PyTorch implementation. The model architecture is compatible, but weight loading may require adaptation.

### Example usage

```python
import torch
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

# Location of pretrained LM
vocab_file = 'vocab.txt'
options_file = 'options.json'
weight_file = 'weights.hdf5'

# Create a Batcher to map text to character ids
batcher = Batcher(vocab_file, 50)

# Build the biLM model
bilm = BidirectionalLanguageModel(options_file, weight_file)
bilm.eval()  # Set to evaluation mode

# Input sentences
sentences = [['First', 'sentence', '.'], ['Second', 'sentence', '.']]

# Create batch
character_ids = batcher.batch_sentences(sentences)
character_ids = torch.from_numpy(character_ids).long()

# Get ELMo representations
with torch.no_grad():
    bilm_output = bilm(character_ids)
    elmo_embeddings = weight_layers('elmo', bilm_output, l2_coef=0.0)
    
    # Extract the weighted representations
    elmo_representations = elmo_embeddings['weighted_op']
    print("ELMo representations shape:", elmo_representations.shape)
```

## Training your own ELMo model

### Preparing training data

Training data should be a plain text file with one sentence per line. For best results:
- Tokenize the data (e.g., using spaCy or NLTK)
- One sentence per line
- Tokens separated by spaces

### Creating vocabulary

Create a vocabulary file with one token per line. Include the special tokens:
```
<S>
</S>
<UNK>
token1
token2
...
```

### Training

Use the training script:

```bash
python bin/train_elmo.py \
    --save_dir /path/to/save/dir \
    --vocab_file /path/to/vocab.txt \
    --train_prefix /path/to/training/data \
    --use_character_inputs \
    --batch_size 32 \
    --n_epochs 10 \
    --learning_rate 0.001
```

### Training options

Key training parameters:

- `--use_character_inputs`: Use character-level inputs (recommended)
- `--batch_size`: Batch size for training
- `--n_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--test_prefix`: Optional test data prefix for evaluation

## Model Architecture

The PyTorch implementation maintains the same architecture as the original:

1. **Character-level CNN**: Converts character sequences to word representations
2. **Highway Networks**: Applies highway connections for better gradient flow
3. **Bidirectional LSTM**: Captures forward and backward context
4. **ELMo Embedder**: Learns weighted combinations of layer representations

### Key Components

- `BidirectionalLanguageModel`: Main model class
- `ElmoEmbedder`: Computes weighted layer combinations
- `LanguageModelTrainer`: Training utilities
- `Batcher`: Converts text to character/token IDs

## Differences from TensorFlow Version

1. **Framework**: Uses PyTorch instead of TensorFlow
2. **API**: More Pythonic API with standard PyTorch conventions
3. **Training**: Simplified training loop with modern PyTorch practices
4. **Device Support**: Automatic GPU/CPU detection and usage
5. **Checkpointing**: PyTorch-style model checkpointing

## GPU Support

The model automatically detects and uses CUDA if available:

```python
# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Model will automatically use GPU if available
bilm = BidirectionalLanguageModel(options_file, weight_file)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project maintains the same license as the original bilm-tf implementation.

## Acknowledgments

This PyTorch implementation is based on the original TensorFlow implementation by AllenAI. Special thanks to the original authors for their groundbreaking work on ELMo.