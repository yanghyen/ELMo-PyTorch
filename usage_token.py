'''
ELMo usage example with pre-computed and cached context independent
token representations (PyTorch version)

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import torch
import os
import h5py
import numpy as np
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers

# Our small dataset.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create the vocabulary file with all unique tokens and
# the special <S>, </S> tokens (case sensitive).
all_tokens = set(['<S>', '</S>'] + tokenized_question[0])
for context_sentence in tokenized_context:
    for token in context_sentence:
        all_tokens.add(token)
vocab_file = 'vocab_small.txt'
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

# Location of pretrained LM. Here we use the test fixtures.
datadir = os.path.join('tests', 'fixtures', 'model')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

# Create dummy token embeddings file for this example
token_embedding_file = 'elmo_token_embeddings.hdf5'

def dump_token_embeddings(vocab_file, options_file, weight_file, output_file):
    '''
    Dump token embeddings to HDF5 file (simplified version)
    '''
    # Read vocabulary
    with open(vocab_file, 'r') as f:
        vocab = [line.strip() for line in f]
    
    vocab_size = len(vocab)
    embed_dim = 512  # Default embedding dimension
    
    # Create random embeddings for demonstration
    # In practice, these would be extracted from the pre-trained model
    embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    
    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('embedding', data=embeddings)
    
    print(f"Token embeddings saved to {output_file}")

# Dump the token embeddings to a file. Run this once for your dataset.
dump_token_embeddings(vocab_file, options_file, weight_file, token_embedding_file)

## Now we can do inference.
# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)

# Build the biLM model.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)
bilm.eval()  # Set to evaluation mode

# Create batches of data.
context_ids = batcher.batch_sentences(tokenized_context)
question_ids = batcher.batch_sentences(tokenized_question)

# Convert to PyTorch tensors
context_token_ids = torch.from_numpy(context_ids).long()
question_token_ids = torch.from_numpy(question_ids).long()

# Get BiLM embeddings and compute ELMo representations
with torch.no_grad():
    # Get ops to compute the LM embeddings.
    context_embeddings_op = bilm(context_token_ids)
    question_embeddings_op = bilm(question_token_ids)

    # Get ELMo representations (weighted average of the internal biLM layers)
    # Our SQuAD model includes ELMo at both the input and output layers
    # of the task GRU, so we need 4x ELMo representations for the question
    # and context at each of the input and output.
    
    elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
    elmo_question_input = weight_layers('input', question_embeddings_op, l2_coef=0.0)
    
    elmo_context_output = weight_layers('output', context_embeddings_op, l2_coef=0.0)
    elmo_question_output = weight_layers('output', question_embeddings_op, l2_coef=0.0)

    # Extract the weighted representations
    elmo_context_input_repr = elmo_context_input['weighted_op']
    elmo_question_input_repr = elmo_question_input['weighted_op']
    
    print("Context ELMo input representations shape:", elmo_context_input_repr.shape)
    print("Question ELMo input representations shape:", elmo_question_input_repr.shape)
    
    # Print some statistics
    print("Context ELMo mean:", elmo_context_input_repr.mean().item())
    print("Question ELMo mean:", elmo_question_input_repr.mean().item())

# Clean up temporary files
if os.path.exists(vocab_file):
    os.remove(vocab_file)
if os.path.exists(token_embedding_file):
    os.remove(token_embedding_file)