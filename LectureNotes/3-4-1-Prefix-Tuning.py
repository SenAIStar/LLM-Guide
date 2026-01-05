class PrefixEncoder(nn.Module):
    '''
    The model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.prefix_length, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.encoder_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.encoder_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.prefix_length, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    

if __name__ == "__main__":
    configs = {"prefix_length":20,
               "hidden_size":768,
               "encoder_hidden_size":768,
               "num_hidden_layers":12,
               "prefix_projection":False
               }
    prefix_encoder = PrefixEncoder(config=PretrainedConfig.from_dict(configs))
    print(prefix_encoder)
    batch_size = 8
    prefix = torch.arange(20).long().expand(batch_size, -1)
    print(prefix.shape)
    output = prefix_encoder(prefix)
    print(output.shape)