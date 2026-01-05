from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


def gather_from_model_parallel_region(tensor):  
    """  
    Gathers tensors from all model parallel GPUs.     
    Args:  
        tensor: tensor to gather. Should be located on the current GPU.   
    Returns:  
        A tensor located on the GPU that called the function. If called from GPU 0,  
        it will contain the concatenated tensors from all the model parallel GPUs.  
    """  
    # Check if distributed is initialized and if we are using more than 1 GPU.  
    if not dist.is_initialized() or dist.get_world_size() == 1:  
        return tensor  
  
    # Get the rank of the current GPU and the total number of GPUs.  
    rank = dist.get_rank()  
    world_size = dist.get_world_size()  
  
    # List to store tensors from all GPUs.  
    gathered_tensors = [torch.empty_like(tensor) for _ in range(world_size)]  
  
    # All GPUs perform the all_gather operation.  
    dist.all_gather(gathered_tensors, tensor)  
  
    # Concatenate the gathered tensors.  
    gathered_tensor = torch.cat(gathered_tensors, dim=0)  
  
    return gathered_tensor  

class GPT3ParallelSelfAttention(torch.nn.Module):  
    
    def __init__(self, hidden_size, num_attention_heads,  
                 attention_dropout_prob, output_dropout_prob):  
        super(GPT3ParallelSelfAttention, self).__init__()  
        # Per attention head and per partition values.  
        world_size = torch.distributed.get_world_size()  
        self.hidden_size_per_partition = hidden_size // world_size  
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads  
        self.num_attention_heads_per_partition = num_attention_heads // world_size  
  
        # Strided linear layer using ColumnParallelLinear for query, key, value.  
        self.query_key_value = ColumnParallelLinear(  
            hidden_size, 3 * hidden_size,  
            gather_output=False,  
            bias=True  
        )  
  
        # Dropout for attention scores.  
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)  
  
        # Output linear layer using RowParallelLinear.  
        self.dense = RowParallelLinear(  
            self.hidden_size_per_partition,  
            hidden_size,  
            input_is_parallel=True,  
            bias=True  
        )  
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)  
  
    def _transpose_for_scores(self, tensor):  
        new_tensor_shape = tensor.size()[:-1] + \
                            (self.num_attention_heads_per_partition,  
                            self.hidden_size_per_attention_head)  
        tensor = tensor.view(*new_tensor_shape)  
        return tensor.permute(0, 2, 1, 3)  
        
    def _split_tensor_along_last_dim(self, tensor, num_splits):  
        """将最后一个维度分割成几个块"""  
        last_dim = tensor.size(-1)  
        split_size = last_dim // num_splits  
        return torch.split(tensor, split_size, dim=-1)  
  
    def forward(self, hidden_states, ltor_mask):  
        # Attention heads. [b, s, h]  
        mixed_x_layer = self.query_key_value(hidden_states)  
        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = \
            self._split_tensor_along_last_dim(mixed_x_layer, 3)  
  
        # Reshape and transpose [b, np, s, hn]  
        query_layer = self._transpose_for_scores(mixed_query_layer)  
        key_layer = self._transpose_for_scores(mixed_key_layer)  
        value_layer = self._transpose_for_scores(mixed_value_layer)  
  
        # Raw attention scores. [b, np, s, s]  
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  
        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_attention_head)  
        # Apply the left to right attention mask.  
        attention_scores = torch.mul(attention_scores, ltor_mask) - 10000.0 * (1.0 - ltor_mask)  
  
        # Attention probabilities. [b, np, s, s]  
        attention_probs = F.softmax(attention_scores, dim=-1)  
        attention_probs = self.attention_dropout(attention_probs)  
  
        # Context layer. [b, np, s, hn]  
        context_layer = torch.matmul(attention_probs, value_layer)  
  
        # [b, s, np, hn]  
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)  
        # [b, s, hp]  
        context_layer = context_layer.view(*new_context_layer_shape)  
  
        # Output. [b, s, h]  
        output = self.dense(context_layer)  
        output = self.output_dropout(output)  
  
        return output  
    
class GPT3ParallelMLP(torch.nn.Module):   
  
    def __init__(self, hidden_size, output_dropout_prob):  
        super(GPT3ParallelMLP, self).__init__()   
        # Project to 4h.  
        self.dense_h_to_4h = ColumnParallelLinear(  
            hidden_size, 4 * hidden_size,  
            gather_output=False  
        )  
        # Apply GELU activation.  
        self.gelu_activation = torch.nn.GELU()  
        # Project back to h.  
        self.dense_4h_to_h = RowParallelLinear(  
            4 * hidden_size,  
            hidden_size,  
            input_is_parallel=True  
        )  
        self.dropout = torch.nn.Dropout(output_dropout_prob)  
  
    def forward(self, hidden_states):  
        # [b, s, 4hp]  
        intermediate_parallel = self.dense_h_to_4h(hidden_states)  
        intermediate_parallel = self.gelu_activation(intermediate_parallel)  
  
        # [b, s, h]  
        output = self.dense_4h_to_h(intermediate_parallel)  
        output = self.dropout(output)  
        return output  
    
class GPT3ParallelTransformerLayer(nn.Module): 
  
    def __init__(self,  
                 hidden_size,  
                 num_attention_heads,  
                 attention_dropout_prob,  
                 output_dropout_prob,  
                 layernorm_epsilon):  
        super(GPT3ParallelTransformerLayer, self).__init__()  
  
        # Layernorm on the input data.  
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)  
  
        # Self attention.  
        self.attention = GPT3ParallelSelfAttention(  
            hidden_size,  
            num_attention_heads,  
            attention_dropout_prob,  
            output_dropout_prob  
        )  
  
        # Layernorm on the input data.  
        self.post_attention_layernorm = nn.LayerNorm(hidden_size,  
                                                     eps=layernorm_epsilon)  
  
        # MLP  
        self.mlp = GPT3ParallelMLP(  
            hidden_size,  
            output_dropout_prob  
        )  
  
    def forward(self, hidden_states, ltor_mask):  
        # Layer norm at the beginning of the transformer layer.  
        layernorm_output = self.input_layernorm(hidden_states)  
          
        # Self attention.  
        attention_output = self.attention(layernorm_output, ltor_mask)  
          
        # Residual connection.  
        layernorm_input = hidden_states + attention_output  
          
        # Layer norm post the self attention.  
        layernorm_output = self.post_attention_layernorm(layernorm_input)  
          
        # MLP.  
        mlp_output = self.mlp(layernorm_output)  
          
        # Second residual connection.  
        output = layernorm_input + mlp_output  
  
        return output  
    
class GPT3ParallelTransformer(nn.Module):  

    def __init__(self,  
                 num_layers,  
                 hidden_size,  
                 num_attention_heads,  
                 attention_dropout_prob,  
                 output_dropout_prob,  
                 layernorm_epsilon=1.0e-5):  
        super(GPT3ParallelTransformer, self).__init__()  
  
        # Transformer layers.  
        self.layers = nn.ModuleList([  
            GPT3ParallelTransformerLayer(  
                hidden_size=hidden_size,  
                num_attention_heads=num_attention_heads,  
                attention_dropout_prob=attention_dropout_prob,  
                output_dropout_prob=output_dropout_prob,  
                layernorm_epsilon=layernorm_epsilon  
            )  
            for _ in range(num_layers)  
        ])  
  
        # Final layer norm before output.  
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)  
  
    def forward(self, hidden_states, attention_mask):  
        # Forward pass through each layer.  
        for layer in self.layers:  
            hidden_states = layer(hidden_states, attention_mask)  
  
        # Final layer norm.  
        output = self.final_layernorm(hidden_states)  
  
        return output  

class GPT3Model(nn.Module):  
    """GPT-3 Language model with fairscale VocabParallelEmbedding.     
    The output of the forward method are the logits.  
    """  
  
    def __init__(self,  
                 num_layers,  
                 vocab_size,  
                 hidden_size,  
                 num_attention_heads,  
                 embedding_dropout_prob,  
                 attention_dropout_prob,  
                 output_dropout_prob,  
                 max_sequence_length,  
                 parallel_output=True):  
  
        super(GPT3Model, self).__init__()  
  
        self.parallel_output = parallel_output  
  
        # Word embeddings using fairscale's VocabParallelEmbedding.  
        self.word_embeddings = VocabParallelEmbedding(vocab_size, hidden_size)  
          
        # Initialize the word embeddings.  
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)  
  
        # Position embedding.  
        self.position_embeddings = nn.Embedding(max_sequence_length, hidden_size)  
          
        # Initialize the position embeddings.  
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)  
  
        # Embeddings dropout  
        self.embedding_dropout = nn.Dropout(embedding_dropout_prob)  
  
        # Transformer  
        self.transformer = GPT3ParallelTransformer(  
            num_layers=num_layers,  
            hidden_size=hidden_size,  
            num_attention_heads=num_attention_heads,  
            attention_dropout_prob=attention_dropout_prob,  
            output_dropout_prob=output_dropout_prob,  
            layernorm_epsilon=1e-5  
        )  
  
    def forward(self, input_ids, position_ids, attention_mask):  
        # Embeddings.  
        words_embeddings = self.word_embeddings(input_ids)  
        position_embeddings = self.position_embeddings(position_ids)  
        embeddings = words_embeddings + position_embeddings  
  
        # Dropout.  
        embeddings = self.embedding_dropout(embeddings)  
  
        # Transformer.  
        transformer_output = self.transformer(embeddings, attention_mask)  
  
        # Logits.  
        # If using model parallelism, gather outputs before linear layer.  
        if self.parallel_output:  
            transformer_output = gather_from_model_parallel_region(transformer_output)  
        logits = F.linear(transformer_output, self.word_embeddings.weight)  
  
        return logits  