class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data, dummy_var, seq_length):
        self.data = data
        self.dummy_var = dummy_var
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Repeat dummy_var along the sequence length axis
        dummy_repeated = np.tile(self.dummy_var[index], (self.seq_length, 1))
        return self.data[index], 
        


class TransformerTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, dummy_dim, model_dim, num_heads, num_layers, seq_length):
        super(TransformerTimeSeriesModel, self).__init__()
        self.seq_length = seq_length
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.dummy_embedding = nn.Linear(dummy_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, input_dim)
        self.classification_layer = nn.Linear(model_dim, dummy_dim)
        #num_classes and dummy_dim are same thing

    def forward(self, x, dummy):
        x = self.input_embedding(x)
        dummy = self.dummy_embedding(dummy.float())  # Ensure dummy is of type float
        x = x + dummy + self.positional_encoding
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, model_dim)
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, model_dim)
        output_recon = self.output_layer(output)
        output_class = self.classification_layer(output)
        return output_recon, output_class