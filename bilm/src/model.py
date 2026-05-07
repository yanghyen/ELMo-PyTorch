import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import json
import re

from .data import UnicodeCharsVocabulary, Batcher

DTYPE = torch.float32
DTYPE_INT = torch.int64


class BidirectionalLanguageModel(nn.Module):
    def __init__(
            self,
            options_file: str,
            weight_file: str,
            use_character_inputs=True,
            embedding_weight_file=None,
            max_batch_size=128,
        ):
        '''
        Creates the language model computational graph and loads weights

        Two options for input type:
            (1) To use character inputs (paired with Batcher)
                pass use_character_inputs=True, and ids_placeholder
                of shape (None, None, max_characters_per_token)
                to forward
            (2) To use token ids as input (paired with TokenBatcher),
                pass use_character_inputs=False and ids_placeholder
                of shape (None, None) to forward.
                In this case, embedding_weight_file is also required input

        options_file: location of the json formatted file with
                      LM hyperparameters
        weight_file: location of the hdf5 file with LM weights
        use_character_inputs: if True, then use character ids as input,
            otherwise use token ids
        max_batch_size: the maximum allowable batch size 
        '''
        super(BidirectionalLanguageModel, self).__init__()
        
        with open(options_file, 'r') as fin:
            options = json.load(fin)

        if not use_character_inputs:
            if embedding_weight_file is None:
                raise ValueError(
                    "embedding_weight_file is required input with "
                    "not use_character_inputs"
                )

        self._options = options
        self._weight_file = weight_file
        self._embedding_weight_file = embedding_weight_file
        self._use_character_inputs = use_character_inputs
        self._max_batch_size = max_batch_size

        self._build()
        self._load_weights()

    def _build(self):
        '''
        Build the computational graph
        '''
        # character embedding layer
        if self._use_character_inputs:
            self._build_character_embedding()
        else:
            self._build_token_embedding()

        # LSTM layers
        self._build_lstm()

        # output projection layer
        self._build_output_projection()

    def _build_character_embedding(self):
        '''
        Build character embedding layer
        '''
        options = self._options
        
        # character vocabulary size
        char_vocab_size = options['char_cnn']['n_characters']
        
        # character embedding dimension
        char_embed_dim = options['char_cnn']['embedding']['dim']
        
        # character embedding layer
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim)
        
        # CNN layers for character-level word representations
        filters = options['char_cnn']['filters']
        n_filters = sum(f[1] for f in filters)
        
        self.char_cnn_layers = nn.ModuleList()
        for i, (width, num) in enumerate(filters):
            conv = nn.Conv1d(char_embed_dim, num, width, padding=width//2)
            self.char_cnn_layers.append(conv)
        
        # highway layers
        self.highway_layers = nn.ModuleList()
        n_highway = options['char_cnn']['n_highway']
        highway_dim = n_filters
        
        for i in range(n_highway):
            self.highway_layers.append(Highway(highway_dim))
        
        # projection layer
        projection_dim = options['char_cnn']['projection']['dim']
        self.projection = nn.Linear(highway_dim, projection_dim)

    def _build_token_embedding(self):
        '''
        Build token embedding layer
        '''
        # Load pre-trained token embeddings
        with h5py.File(self._embedding_weight_file, 'r') as f:
            embeddings = f['embedding'][...]
        
        vocab_size, embed_dim = embeddings.shape
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.token_embedding.weight.data = torch.from_numpy(embeddings)

    def _build_lstm(self):
        '''
        Build LSTM layers
        '''
        options = self._options
        lstm_options = options['lstm']
        
        # LSTM dimensions
        if self._use_character_inputs:
            input_dim = options['char_cnn']['projection']['dim']
        else:
            input_dim = self.token_embedding.embedding_dim
            
        hidden_dim = lstm_options['projection_dim']
        cell_dim = lstm_options['dim']
        n_layers = lstm_options['n_layers']
        
        # Forward LSTM
        self.forward_lstm = nn.ModuleList()
        for i in range(n_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            lstm_layer = LSTMLayer(layer_input_dim, cell_dim, hidden_dim)
            self.forward_lstm.append(lstm_layer)
        
        # Backward LSTM
        self.backward_lstm = nn.ModuleList()
        for i in range(n_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            lstm_layer = LSTMLayer(layer_input_dim, cell_dim, hidden_dim)
            self.backward_lstm.append(lstm_layer)

    def _build_output_projection(self):
        '''
        Build output projection layer for language modeling
        '''
        options = self._options
        lstm_options = options['lstm']
        
        hidden_dim = lstm_options['projection_dim']
        
        if self._use_character_inputs:
            # Word-level LM softmax (pretraining); fallback for legacy options
            vocab_size = options.get(
                'n_tokens_vocab', options['char_cnn']['n_characters']
            )
        else:
            vocab_size = self.token_embedding.num_embeddings

        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def _load_weights(self):
        '''
        Load pre-trained weights from HDF5 file
        '''
        with h5py.File(self._weight_file, 'r') as f:
            # Load character embedding weights
            if self._use_character_inputs:
                char_embed_weights = f['char_embed'][...]
                self.char_embedding.weight.data = torch.from_numpy(char_embed_weights)
                
                # Load CNN weights
                for i, conv_layer in enumerate(self.char_cnn_layers):
                    conv_w = f[f'CNN/W_{i}'][...]
                    conv_b = f[f'CNN/b_{i}'][...]
                    conv_layer.weight.data = torch.from_numpy(conv_w).transpose(1, 2).transpose(0, 1)
                    conv_layer.bias.data = torch.from_numpy(conv_b)
                
                # Load highway weights
                for i, highway_layer in enumerate(self.highway_layers):
                    highway_w = f[f'CNN_high_{i}/W_transform'][...]
                    highway_b = f[f'CNN_high_{i}/b_transform'][...]
                    highway_layer.transform_gate.weight.data = torch.from_numpy(highway_w.T)
                    highway_layer.transform_gate.bias.data = torch.from_numpy(highway_b)
                    
                    carry_w = f[f'CNN_high_{i}/W_carry'][...]
                    carry_b = f[f'CNN_high_{i}/b_carry'][...]
                    highway_layer.carry_gate.weight.data = torch.from_numpy(carry_w.T)
                    highway_layer.carry_gate.bias.data = torch.from_numpy(carry_b)
                
                # Load projection weights
                proj_w = f['CNN_proj/W_proj'][...]
                proj_b = f['CNN_proj/b_proj'][...]
                self.projection.weight.data = torch.from_numpy(proj_w.T)
                self.projection.bias.data = torch.from_numpy(proj_b)
            
            # Load LSTM weights
            for i, (fw_layer, bw_layer) in enumerate(zip(self.forward_lstm, self.backward_lstm)):
                # Forward LSTM
                fw_w = f[f'RNN_{i}/RNN/MultiRNNCell/Cell{i}/LSTMCell/W_0'][...]
                fw_b = f[f'RNN_{i}/RNN/MultiRNNCell/Cell{i}/LSTMCell/B'][...]
                fw_layer.load_weights(fw_w, fw_b)
                
                # Backward LSTM
                bw_w = f[f'RNN_{i}/RNN_1/MultiRNNCell/Cell{i}/LSTMCell/W_0'][...]
                bw_b = f[f'RNN_{i}/RNN_1/MultiRNNCell/Cell{i}/LSTMCell/B'][...]
                bw_layer.load_weights(bw_w, bw_b)

    def forward(self, inputs):
        '''
        Forward pass through the bidirectional language model
        
        Args:
            inputs: character ids of shape (batch_size, sequence_length, max_chars_per_token)
                   or token ids of shape (batch_size, sequence_length)
        
        Returns:
            Dictionary containing:
                'lm_embeddings': internal representations from each layer
                'mask': mask for valid tokens
        '''
        if self._use_character_inputs:
            return self._forward_character_inputs(inputs)
        else:
            return self._forward_token_inputs(inputs)

    def _forward_character_inputs(self, char_ids):
        '''
        Forward pass with character inputs
        '''
        batch_size, seq_len, max_chars = char_ids.shape
        
        # Create mask for valid tokens (non-zero character sequences)
        mask = (char_ids.sum(dim=-1) > 0).float()
        
        # Character embedding
        char_embeds = self.char_embedding(char_ids)  # (batch, seq, max_chars, embed_dim)
        char_embeds = char_embeds.transpose(2, 3)  # (batch, seq, embed_dim, max_chars)
        
        # CNN over characters
        cnn_outputs = []
        for i, conv_layer in enumerate(self.char_cnn_layers):
            # Reshape for conv1d: (batch * seq, embed_dim, max_chars)
            x = char_embeds.view(-1, char_embeds.size(2), char_embeds.size(3))
            conv_out = conv_layer(x)  # (batch * seq, n_filters, max_chars)
            conv_out = F.relu(conv_out)
            conv_out = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch * seq, n_filters, 1)
            conv_out = conv_out.squeeze(2)  # (batch * seq, n_filters)
            cnn_outputs.append(conv_out)
        
        # Concatenate CNN outputs
        cnn_output = torch.cat(cnn_outputs, dim=1)  # (batch * seq, total_filters)
        cnn_output = cnn_output.view(batch_size, seq_len, -1)  # (batch, seq, total_filters)
        
        # Highway layers
        highway_output = cnn_output
        for highway_layer in self.highway_layers:
            highway_output = highway_layer(highway_output)
        
        # Projection
        token_embeddings = self.projection(highway_output)  # (batch, seq, proj_dim)
        
        return self._run_lstm(token_embeddings, mask)

    def _forward_token_inputs(self, token_ids):
        '''
        Forward pass with token inputs
        '''
        # Create mask for valid tokens (non-padding)
        mask = (token_ids > 0).float()
        
        # Token embedding
        token_embeddings = self.token_embedding(token_ids)
        
        return self._run_lstm(token_embeddings, mask)

    def _run_lstm(self, token_embeddings, mask):
        '''
        Run bidirectional LSTM
        '''
        batch_size, seq_len, embed_dim = token_embeddings.shape
        
        # Forward LSTM
        forward_states = []
        forward_input = token_embeddings
        
        for lstm_layer in self.forward_lstm:
            forward_output, _ = lstm_layer(forward_input, mask)
            forward_states.append(forward_output)
            forward_input = forward_output
        
        # Backward LSTM (reverse sequence)
        backward_states = []
        backward_input = torch.flip(token_embeddings, dims=[1])
        backward_mask = torch.flip(mask, dims=[1])
        
        for lstm_layer in self.backward_lstm:
            backward_output, _ = lstm_layer(backward_input, backward_mask)
            backward_states.append(torch.flip(backward_output, dims=[1]))  # Flip back
            backward_input = backward_output
        
        # Combine forward and backward states
        lm_embeddings = []
        
        # Add token embeddings as layer 0
        lm_embeddings.append(token_embeddings.unsqueeze(1))  # (batch, 1, seq, embed)
        
        # Add LSTM layer outputs
        for fw_state, bw_state in zip(forward_states, backward_states):
            combined = torch.cat([fw_state, bw_state], dim=-1)  # (batch, seq, 2*hidden)
            lm_embeddings.append(combined.unsqueeze(1))  # (batch, 1, seq, 2*hidden)
        
        # Stack all layers
        lm_embeddings = torch.cat(lm_embeddings, dim=1)  # (batch, n_layers+1, seq, dim)

        return {
            'lm_embeddings': lm_embeddings,
            'mask': mask,
            'forward_output': forward_states[-1],
            'backward_output': backward_states[-1],
        }


class Highway(nn.Module):
    '''
    Highway network layer
    '''
    def __init__(self, input_dim):
        super(Highway, self).__init__()
        self.transform_gate = nn.Linear(input_dim, input_dim)
        self.carry_gate = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        transform = torch.sigmoid(self.transform_gate(x))
        carry = torch.sigmoid(self.carry_gate(x))
        transformed = F.relu(self.transform_gate(x))
        return transform * transformed + carry * x


class LSTMLayer(nn.Module):
    '''
    LSTM layer with projection
    '''
    def __init__(self, input_dim, cell_dim, projection_dim):
        super(LSTMLayer, self).__init__()
        self.input_dim = input_dim
        self.cell_dim = cell_dim
        self.projection_dim = projection_dim
        
        # LSTM cell
        self.lstm_cell = nn.LSTMCell(input_dim, cell_dim)
        
        # Projection layer
        self.projection = nn.Linear(cell_dim, projection_dim)
        
    def load_weights(self, weight_matrix, bias_vector):
        '''
        Load weights from TensorFlow format
        '''
        # Split weight matrix into input and hidden weights
        input_weights = weight_matrix[:self.input_dim, :]
        hidden_weights = weight_matrix[self.input_dim:, :]
        
        # Combine input and hidden weights for PyTorch LSTM
        combined_weights = torch.cat([input_weights, hidden_weights], dim=0)
        
        self.lstm_cell.weight_ih.data = combined_weights[:self.input_dim, :].T
        self.lstm_cell.weight_hh.data = combined_weights[self.input_dim:, :].T
        self.lstm_cell.bias_ih.data = bias_vector
        self.lstm_cell.bias_hh.data = torch.zeros_like(bias_vector)
        
    def forward(self, inputs, mask):
        '''
        Forward pass through LSTM layer
        '''
        batch_size, seq_len, input_dim = inputs.shape
        
        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.cell_dim, device=inputs.device)
        c = torch.zeros(batch_size, self.cell_dim, device=inputs.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Get input at time t
            x_t = inputs[:, t, :]  # (batch, input_dim)
            mask_t = mask[:, t].unsqueeze(1)  # (batch, 1)
            
            # LSTM cell forward
            h_new, c_new = self.lstm_cell(x_t, (h, c))
            
            # Apply mask
            h = mask_t * h_new + (1 - mask_t) * h
            c = mask_t * c_new + (1 - mask_t) * c
            
            # Project hidden state
            output = self.projection(h)  # (batch, projection_dim)
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (batch, seq, projection_dim)
        
        return outputs, (h, c)