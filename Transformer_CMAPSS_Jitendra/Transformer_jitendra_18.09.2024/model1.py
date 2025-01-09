import torch
import torch.nn as nn

class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, seq_length, num_classes, dropout_rate, batch_first=True, norm_first=False, bias=True):
        super(TransformerTimeSeriesModel, self).__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.seq_length = seq_length
        self.num_classes = num_classes
        
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        
        self.transformer = nn.Transformer(
                                    d_model=model_dim,
                                    nhead=num_heads, 
                                    num_encoder_layers=num_layers, 
                                    num_decoder_layers=num_layers,
                                    dim_feedforward=64, 
                                    dropout=dropout_rate,
                                    batch_first=batch_first,
                                    norm_first=norm_first,
                                    bias=bias)
        
        self.classification_output = nn.Linear(model_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size, seq_length, _ = x.size()
        
        # Apply embedding and positional encoding to input features
        x = self.input_embedding(x)
        x = x + self.positional_encoding[:, :seq_length, :]
        
        # Transformer expects input of shape (seq_length, batch_size, model_dim)
        x = x.permute(1, 0, 2)
        
        # Apply transformer
        memory = self.transformer(x, x)
        
        # Permute back to (batch_size, seq_length, model_dim)
        memory = memory.permute(1, 0, 2)
        
        # Apply dropout
        memory = self.dropout(memory)
        
        # Classification output
        class_output = self.classification_output(memory)
        
        return class_output
