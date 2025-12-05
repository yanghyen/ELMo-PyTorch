import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


class CharCNN(nn.Module):
    """
    Character-level CNN for ELMo
    GPU 최적화를 위해 효율적인 convolution과 max pooling 사용
    """
    def __init__(self, vocab_size, char_embed_dim=16, filters=None, max_word_length=50):
        super(CharCNN, self).__init__()
        
        if filters is None:
            # filters = [(1, 32), (2, 32), (3, 64), (4, 128), (5, 256), (6, 512), (7, 1024)]
            filters = [(1, 32), (2, 32), (3, 64), (4, 128), (5, 256), (6, 256), (7, 256)]
        
        self.char_embed_dim = char_embed_dim
        self.max_word_length = max_word_length
        self.filters = filters
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim, padding_idx=0)
        
        # Convolution layers - GPU에서 병렬 처리 최적화
        self.convolutions = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, kernel_size=width, padding=0)
            for width, num_filters in filters
        ])
        
        # Output dimension
        self.output_dim = sum(num_filters for _, num_filters in filters)
        
        # Highway networks will be applied after this
        
    def forward(self, char_ids):
        """
        Args:
            char_ids: (batch_size, num_words, max_word_length)
        Returns:
            (batch_size, num_words, output_dim)
        """
        batch_size, num_words, max_word_length = char_ids.size()
        
        # Reshape for character-level processing
        # (batch_size * num_words, max_word_length)
        char_ids_flat = char_ids.view(-1, max_word_length)
        
        # Character embedding: (batch_size * num_words, max_word_length, char_embed_dim)
        char_embeds = self.char_embedding(char_ids_flat)
        
        # Transpose for conv1d: (batch_size * num_words, char_embed_dim, max_word_length)
        char_embeds = char_embeds.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convolutions:
            # Convolution: (batch_size * num_words, num_filters, conv_length)
            conv_out = F.relu(conv(char_embeds))
            # Max pooling over time: (batch_size * num_words, num_filters)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatenate all filter outputs: (batch_size * num_words, total_filters)
        word_embeds = torch.cat(conv_outputs, dim=1)
        
        # Reshape back: (batch_size, num_words, output_dim)
        word_embeds = word_embeds.view(batch_size, num_words, self.output_dim)
        
        return word_embeds


class Highway(nn.Module):
    """
    Highway Network for ELMo
    효율적인 residual connection과 gating mechanism
    """
    def __init__(self, input_dim, num_layers=2):
        super(Highway, self).__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Highway layers
        self.linear_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        
        self.gate_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_layers)
        ])
        
        # Initialize gate bias to negative value for better training start
        for gate_layer in self.gate_layers:
            gate_layer.bias.data.fill_(-2.0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            (batch_size, seq_len, input_dim)
        """
        for linear_layer, gate_layer in zip(self.linear_layers, self.gate_layers):
            # Transform gate
            gate = torch.sigmoid(gate_layer(x))
            
            # Non-linear transformation
            transform = F.relu(linear_layer(x))
            
            # Highway connection: gate * transform + (1 - gate) * x
            x = gate * transform + (1 - gate) * x
        
        return x


class UniLSTM(nn.Module):
    """
    Unidirectional LSTM for ELMo (Forward or Backward)
    GPU 최적화를 위한 효율적인 LSTM 구현
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(UniLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Unidirectional LSTM (완전히 분리된 Forward/Backward)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # 단방향으로 변경
            batch_first=True
        )
        
        # Projection layer (hidden_dim -> input_dim)
        self.projection = nn.Linear(hidden_dim, input_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: actual sequence lengths for packing (optional)
        Returns:
            (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Pack sequences for efficiency if lengths provided
        if lengths is not None:
            # Sort by length for packing
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]
            
            # Pack padded sequences
            packed_x = pack_padded_sequence(x_sorted, lengths_sorted.cpu(), batch_first=True)
            
            # LSTM forward pass
            packed_output, (hidden, cell) = self.lstm(packed_x)
            
            # Unpack sequences with total_length to ensure consistent shape
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)
            
            # Restore original order
            _, unsort_idx = sort_idx.sort()
            lstm_out = lstm_out[unsort_idx]
        else:
            # Standard LSTM forward pass
            lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project back to input dimension
        output = self.projection(lstm_out)
        
        return output


class ELMoBiLM(nn.Module):
    """
    Complete ELMo Bidirectional Language Model
    CharCNN + Highway + BiLSTM + Softmax
    GPU 최적화 및 효율성 극대화
    """
    def __init__(self, config):
        super(ELMoBiLM, self).__init__()
        
        # Configuration
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.projection_dim = config.get('projection_dim', 256)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # Character-level CNN
        # CharCNN을 사용하여 character-level에서 word representation 생성
        char_vocab_size = config.get('char_vocab_size', 262)  # 기본 ASCII + 특수문자
        max_word_length = config.get('max_word_length', 50)
        
        self.char_cnn = CharCNN(
            vocab_size=char_vocab_size,
            char_embed_dim=16,
            max_word_length=max_word_length
        )
        
        # CharCNN output을 embedding_dim으로 projection
        self.char_projection = nn.Linear(self.char_cnn.output_dim, self.embedding_dim)
        
        # 만약 character 정보가 없을 경우를 위한 fallback word embedding
        self.word_embedding = nn.Embedding(
            self.vocab_size, 
            self.embedding_dim, 
            padding_idx=0
        )
        
        # Highway networks
        self.highway = Highway(self.embedding_dim, num_layers=2)
        
        # Forward LSTM (완전히 분리된 단방향)
        self.forward_lstm = UniLSTM(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Backward LSTM (완전히 분리된 단방향)
        self.backward_lstm = UniLSTM(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Projection layers for language modeling
        self.forward_projection = nn.Linear(self.embedding_dim, self.projection_dim)
        self.backward_projection = nn.Linear(self.embedding_dim, self.projection_dim)
        
        # Output layers (softmax)
        self.forward_output = nn.Linear(self.projection_dim, self.vocab_size)
        self.backward_output = nn.Linear(self.projection_dim, self.vocab_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better training"""
        # Word embedding initialization
        nn.init.uniform_(self.word_embedding.weight, -0.1, 0.1)
        self.word_embedding.weight.data[0].fill_(0)  # padding token
        
        # Linear layer initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, forward_input, backward_input, forward_mask=None, backward_mask=None, 
                forward_char_ids=None, backward_char_ids=None):
        """
        Forward pass for both forward and backward language models
        
        Args:
            forward_input: (batch_size, seq_len) - forward sequence (word ids)
            backward_input: (batch_size, seq_len) - backward sequence (word ids)
            forward_mask: (batch_size, seq_len) - mask for forward sequence
            backward_mask: (batch_size, seq_len) - mask for backward sequence
            forward_char_ids: (batch_size, seq_len, max_word_length) - character ids for forward
            backward_char_ids: (batch_size, seq_len, max_word_length) - character ids for backward
            
        Returns:
            forward_logits: (batch_size, seq_len, vocab_size)
            backward_logits: (batch_size, seq_len, vocab_size)
        """
        # Character-level embeddings (if available)
        if forward_char_ids is not None and backward_char_ids is not None:
            # Use CharCNN for character-level word representations
            forward_embeds = self.char_cnn(forward_char_ids)
            backward_embeds = self.char_cnn(backward_char_ids)
            
            # Project to embedding dimension
            forward_embeds = self.char_projection(forward_embeds)
            backward_embeds = self.char_projection(backward_embeds)
        else:
            # Fallback to word-level embeddings
            forward_embeds = self.word_embedding(forward_input)
            backward_embeds = self.word_embedding(backward_input)
        
        # Apply dropout to embeddings before highway
        forward_embeds = self.dropout_layer(forward_embeds)
        backward_embeds = self.dropout_layer(backward_embeds)
        
        # Highway networks
        forward_embeds = self.highway(forward_embeds)
        backward_embeds = self.highway(backward_embeds)
        
        # Calculate sequence lengths for packing (if masks provided)
        forward_lengths = None
        backward_lengths = None
        if forward_mask is not None:
            forward_lengths = forward_mask.sum(dim=1).long()
        if backward_mask is not None:
            backward_lengths = backward_mask.sum(dim=1).long()
        
        # BiLSTM processing
        forward_lstm_out = self.forward_lstm(forward_embeds, forward_lengths)
        backward_lstm_out = self.backward_lstm(backward_embeds, backward_lengths)
        
        # Apply dropout
        forward_lstm_out = self.dropout_layer(forward_lstm_out)
        backward_lstm_out = self.dropout_layer(backward_lstm_out)
        
        # Projection layers
        forward_projected = F.relu(self.forward_projection(forward_lstm_out))
        backward_projected = F.relu(self.backward_projection(backward_lstm_out))
        
        # Apply dropout to projections
        forward_projected = self.dropout_layer(forward_projected)
        backward_projected = self.dropout_layer(backward_projected)
        
        # Output layers (logits for softmax)
        forward_logits = self.forward_output(forward_projected)
        backward_logits = self.backward_output(backward_projected)
        
        return forward_logits, backward_logits
    
    def compute_loss(self, forward_logits, backward_logits, 
                    forward_targets, backward_targets,
                    forward_mask=None, backward_mask=None):
        """
        Compute language modeling loss
        
        Args:
            forward_logits: (batch_size, seq_len, vocab_size)
            backward_logits: (batch_size, seq_len, vocab_size)
            forward_targets: (batch_size, seq_len)
            backward_targets: (batch_size, seq_len)
            forward_mask: (batch_size, seq_len)
            backward_mask: (batch_size, seq_len)
            
        Returns:
            total_loss: scalar tensor
        """
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = forward_logits.size()
        
        forward_logits_flat = forward_logits.view(-1, vocab_size)
        backward_logits_flat = backward_logits.view(-1, vocab_size)
        forward_targets_flat = forward_targets.view(-1)
        backward_targets_flat = backward_targets.view(-1)
        
        # Compute cross entropy loss
        forward_loss = F.cross_entropy(
            forward_logits_flat, 
            forward_targets_flat, 
            reduction='none'
        )
        backward_loss = F.cross_entropy(
            backward_logits_flat, 
            backward_targets_flat, 
            reduction='none'
        )
        
        # Apply masks if provided
        if forward_mask is not None:
            forward_mask_flat = forward_mask.view(-1)
            forward_loss = forward_loss * forward_mask_flat
            forward_loss = forward_loss.sum() / forward_mask_flat.sum()
        else:
            forward_loss = forward_loss.mean()
            
        if backward_mask is not None:
            backward_mask_flat = backward_mask.view(-1)
            backward_loss = backward_loss * backward_mask_flat
            backward_loss = backward_loss.sum() / backward_mask_flat.sum()
        else:
            backward_loss = backward_loss.mean()
        
        # Total loss is average of forward and backward losses
        total_loss = (forward_loss + backward_loss) / 2.0
        
        return total_loss, forward_loss, backward_loss
    
    def get_representations(self, forward_input, backward_input, 
                          forward_mask=None, backward_mask=None,
                          forward_char_ids=None, backward_char_ids=None):
        """
        Extract ELMo representations at different layers
        
        Args:
            forward_input: (batch_size, seq_len)
            backward_input: (batch_size, seq_len)
            forward_mask: (batch_size, seq_len)
            backward_mask: (batch_size, seq_len)
            forward_char_ids: (batch_size, seq_len, max_word_length) - optional
            backward_char_ids: (batch_size, seq_len, max_word_length) - optional
            
        Returns:
            representations: list of (batch_size, seq_len, representation_dim)
        """
        with torch.no_grad():
            # Character-level embeddings (layer 0)
            if forward_char_ids is not None and backward_char_ids is not None:
                forward_embeds = self.char_cnn(forward_char_ids)
                backward_embeds = self.char_cnn(backward_char_ids)
                forward_embeds = self.char_projection(forward_embeds)
                backward_embeds = self.char_projection(backward_embeds)
            else:
                forward_embeds = self.word_embedding(forward_input)
                backward_embeds = self.word_embedding(backward_input)
            
            # Apply dropout before highway networks
            forward_embeds = self.dropout_layer(forward_embeds)
            backward_embeds = self.dropout_layer(backward_embeds)
            
            # Highway networks
            forward_embeds = self.highway(forward_embeds)
            backward_embeds = self.highway(backward_embeds)
            
            # Calculate sequence lengths
            forward_lengths = None
            backward_lengths = None
            if forward_mask is not None:
                forward_lengths = forward_mask.sum(dim=1).long()
            if backward_mask is not None:
                backward_lengths = backward_mask.sum(dim=1).long()
            
            # BiLSTM processing
            forward_lstm_out = self.forward_lstm(forward_embeds, forward_lengths)
            backward_lstm_out = self.backward_lstm(backward_embeds, backward_lengths)
            
            # Concatenate forward and backward representations
            # Layer 0: word embeddings
            layer_0 = torch.cat([forward_embeds, backward_embeds], dim=-1)
            
            # Layer 1: LSTM outputs
            layer_1 = torch.cat([forward_lstm_out, backward_lstm_out], dim=-1)
            
            return [layer_0, layer_1]
    
    def get_elmo_embeddings(self, forward_input, backward_input, 
                           forward_mask=None, backward_mask=None,
                           forward_char_ids=None, backward_char_ids=None,
                           elmo_weights=None):
        """
        Extract task-specific ELMo embeddings using weighted sum
        
        Args:
            forward_input: (batch_size, seq_len)
            backward_input: (batch_size, seq_len)
            forward_mask: (batch_size, seq_len)
            backward_mask: (batch_size, seq_len)
            forward_char_ids: (batch_size, seq_len, max_word_length) - optional
            backward_char_ids: (batch_size, seq_len, max_word_length) - optional
            elmo_weights: ELMoWeightedSum module for task-specific weighting
            
        Returns:
            elmo_embeddings: (batch_size, seq_len, representation_dim)
        """
        # Get all layer representations
        representations = self.get_representations(
            forward_input, backward_input, 
            forward_mask, backward_mask,
            forward_char_ids, backward_char_ids
        )
        
        # Apply task-specific weighted sum if provided
        if elmo_weights is not None:
            return elmo_weights(representations)
        else:
            # Default: return concatenated representations
            return torch.cat(representations, dim=-1)


class ELMoWeightedSum(nn.Module):
    """
    ELMo 가중합: γ * Σ sᵢ * hᵢ
    Task-specific weighted combination of ELMo representations
    """
    def __init__(self, num_layers=2, dropout=0.0):
        super(ELMoWeightedSum, self).__init__()
        
        self.num_layers = num_layers
        
        # Learnable scalar weights (sᵢ)
        self.scalar_weights = nn.Parameter(torch.zeros(num_layers))
        
        # Learnable scaling factor (γ)
        self.gamma = nn.Parameter(torch.ones(1))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, representations):
        """
        Args:
            representations: list of tensors [(batch_size, seq_len, dim), ...]
        Returns:
            weighted_sum: (batch_size, seq_len, dim)
        """
        # Softmax normalize the scalar weights
        normalized_weights = F.softmax(self.scalar_weights, dim=0)
        
        # Weighted sum: Σ sᵢ * hᵢ
        weighted_representations = []
        for i, rep in enumerate(representations):
            if self.dropout is not None:
                rep = self.dropout(rep)
            weighted_representations.append(normalized_weights[i] * rep)
        
        # Sum all weighted representations
        weighted_sum = torch.stack(weighted_representations, dim=0).sum(dim=0)
        
        # Apply scaling factor: γ * Σ sᵢ * hᵢ
        elmo_representation = self.gamma * weighted_sum
        
        return elmo_representation


def create_elmo_model(config):
    """
    Create ELMo BiLM model with given configuration
    
    Args:
        config: dictionary with model configuration
        
    Returns:
        ELMoBiLM model
    """
    model = ELMoBiLM(config)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def create_elmo_weights(num_layers=2, dropout=0.0):
    """
    Create ELMo weighted sum module for task-specific usage
    
    Args:
        num_layers: number of ELMo layers to combine
        dropout: dropout rate for representations
        
    Returns:
        ELMoWeightedSum module
    """
    weights = ELMoWeightedSum(num_layers=num_layers, dropout=dropout)
    
    if torch.cuda.is_available():
        weights = weights.cuda()
    
    return weights