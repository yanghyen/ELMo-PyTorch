'''
ELMo usage example with character inputs (PyTorch version).

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import torch
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

# Location of pretrained LM. Here we use the test fixtures.
datadir = os.path.join('tests', 'fixtures', 'model')
vocab_file = os.path.join(datadir, 'vocab_test.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Build the biLM model.
bilm = BidirectionalLanguageModel(options_file, weight_file)
bilm.eval()  # Set to evaluation mode

# Input data
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create batches of data.
context_ids = batcher.batch_sentences(tokenized_context)
question_ids = batcher.batch_sentences(tokenized_question)

# Convert to PyTorch tensors
context_character_ids = torch.from_numpy(context_ids).long()
question_character_ids = torch.from_numpy(question_ids).long()

# Get BiLM embeddings
with torch.no_grad():
    context_embeddings_op = bilm(context_character_ids)
    question_embeddings_op = bilm(question_character_ids)

    # Get ELMo representations (weighted average of the internal biLM layers)
    # Our SQuAD model includes ELMo at both the input and output layers
    # of the task GRU, so we need 4x ELMo representations for the question
    # and context at each of the input and output.
    
    # Context input ELMo
    elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
    
    # Question input ELMo (reusing weights from context)
    elmo_question_input = weight_layers('input', question_embeddings_op, l2_coef=0.0)
    
    # Context output ELMo
    elmo_context_output = weight_layers('output', context_embeddings_op, l2_coef=0.0)
    
    # Question output ELMo (reusing weights from context)
    elmo_question_output = weight_layers('output', question_embeddings_op, l2_coef=0.0)

    # Extract the weighted representations
    elmo_context_input_repr = elmo_context_input['weighted_op']
    elmo_question_input_repr = elmo_question_input['weighted_op']
    
    print("Context ELMo input representations shape:", elmo_context_input_repr.shape)
    print("Question ELMo input representations shape:", elmo_question_input_repr.shape)
    
    # Print some statistics
    print("Context ELMo mean:", elmo_context_input_repr.mean().item())
    print("Question ELMo mean:", elmo_question_input_repr.mean().item())