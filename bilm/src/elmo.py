import torch
import torch.nn as nn
import torch.nn.functional as F


class ElmoEmbedder(nn.Module):
    '''
    ELMo embedder that computes weighted combinations of biLM layers
    '''
    def __init__(self, n_layers, layer_dim, l2_coef=None, use_top_only=False, do_layer_norm=False):
        super(ElmoEmbedder, self).__init__()
        
        self.n_layers = n_layers
        self.layer_dim = layer_dim
        self.l2_coef = l2_coef
        self.use_top_only = use_top_only
        self.do_layer_norm = do_layer_norm
        
        if not use_top_only:
            # Learnable layer weights
            self.layer_weights = nn.Parameter(torch.zeros(n_layers))
            
        # Learnable scalar parameter
        self.gamma = nn.Parameter(torch.ones(1))
        
        if do_layer_norm:
            self.layer_norm = nn.LayerNorm(layer_dim)
    
    def forward(self, bilm_outputs):
        '''
        Compute ELMo representations
        
        Args:
            bilm_outputs: dict containing 'lm_embeddings' and 'mask'
                lm_embeddings: (batch, n_layers, seq_len, dim)
                mask: (batch, seq_len)
        
        Returns:
            dict containing:
                'elmo_representations': weighted ELMo representations
                'regularization_loss': L2 regularization loss
        '''
        lm_embeddings = bilm_outputs['lm_embeddings']  # (batch, n_layers, seq_len, dim)
        mask = bilm_outputs['mask']  # (batch, seq_len)
        
        batch_size, n_layers, seq_len, dim = lm_embeddings.shape
        
        # Apply layer normalization if requested
        if self.do_layer_norm:
            lm_embeddings = self.layer_norm(lm_embeddings)
        
        if self.use_top_only:
            # Use only the top layer
            weighted_lm_layers = lm_embeddings[:, -1, :, :]  # (batch, seq_len, dim)
        else:
            # Compute softmax weights over layers
            layer_weights = F.softmax(self.layer_weights, dim=0)  # (n_layers,)
            
            # Weight the layers
            weighted_lm_layers = torch.sum(
                layer_weights.view(1, -1, 1, 1) * lm_embeddings, 
                dim=1
            )  # (batch, seq_len, dim)
        
        # Apply scalar parameter
        elmo_representations = self.gamma * weighted_lm_layers
        
        # Apply mask
        mask_expanded = mask.unsqueeze(-1).expand_as(elmo_representations)
        elmo_representations = elmo_representations * mask_expanded
        
        # Compute regularization loss
        regularization_loss = self._compute_regularization_loss()
        
        return {
            'elmo_representations': elmo_representations,
            'regularization_loss': regularization_loss
        }
    
    def _compute_regularization_loss(self):
        '''
        Compute L2 regularization loss for layer weights
        '''
        if self.l2_coef is None or self.l2_coef == 0.0 or self.use_top_only:
            return torch.tensor(0.0, device=self.gamma.device)
        
        return self.l2_coef * torch.sum(self.layer_weights ** 2)


def weight_layers(name, bilm_ops, l2_coef=None, use_top_only=False, do_layer_norm=False):
    '''
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.
    
    This function creates an ElmoEmbedder module and applies it to the biLM outputs.
    
    Args:
        name: a string prefix used for the trainable variable names (for compatibility)
        bilm_ops: the outputs from BidirectionalLanguageModel forward pass
        l2_coef: the l2 regularization coefficient. Pass None or 0.0 for no regularization.
        use_top_only: if True, then only use the top layer.
        do_layer_norm: if True, then apply layer normalization to each biLM layer
    
    Returns:
        dict containing:
            'weighted_op': ELMo representations
            'regularization_op': regularization loss term
    '''
    lm_embeddings = bilm_ops['lm_embeddings']
    n_layers = lm_embeddings.shape[1]
    layer_dim = lm_embeddings.shape[-1]
    
    # Create ELMo embedder
    elmo_embedder = ElmoEmbedder(
        n_layers=n_layers,
        layer_dim=layer_dim,
        l2_coef=l2_coef,
        use_top_only=use_top_only,
        do_layer_norm=do_layer_norm
    )
    
    # Apply embedder
    elmo_outputs = elmo_embedder(bilm_ops)
    
    return {
        'weighted_op': elmo_outputs['elmo_representations'],
        'regularization_op': elmo_outputs['regularization_loss']
    }


class ElmoModel(nn.Module):
    '''
    Complete ELMo model that combines BiLM and ELMo embedder
    '''
    def __init__(self, bilm_model, l2_coef=None, use_top_only=False, do_layer_norm=False):
        super(ElmoModel, self).__init__()
        
        self.bilm_model = bilm_model
        
        # Get dimensions from BiLM model
        if hasattr(bilm_model, 'forward_lstm') and len(bilm_model.forward_lstm) > 0:
            # Character-based model
            n_layers = len(bilm_model.forward_lstm) + 1  # +1 for token embeddings
            layer_dim = bilm_model.forward_lstm[0].projection_dim * 2  # bidirectional
        else:
            # Token-based model
            n_layers = 2  # Simplified for token-based
            layer_dim = bilm_model.token_embedding.embedding_dim
        
        self.elmo_embedder = ElmoEmbedder(
            n_layers=n_layers,
            layer_dim=layer_dim,
            l2_coef=l2_coef,
            use_top_only=use_top_only,
            do_layer_norm=do_layer_norm
        )
    
    def forward(self, inputs):
        '''
        Forward pass through complete ELMo model
        '''
        # Get BiLM representations
        bilm_outputs = self.bilm_model(inputs)
        
        # Get ELMo representations
        elmo_outputs = self.elmo_embedder(bilm_outputs)
        
        return elmo_outputs