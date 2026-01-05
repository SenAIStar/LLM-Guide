def get_prompt(self, batch_size):
    prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
    past_key_values = self.prefix_encoder(prefix_tokens)
    # bsz, seqlen, _ = past_key_values.shape
    past_key_values = past_key_values.view(
        batch_size,
        self.pre_seq_len,
        self.n_layer * 2, 
        self.n_head,
        self.n_embd
    )
    past_key_values = self.dropout(past_key_values)
    past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
    return past_key_values

def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_size = input_ids.shape[0]
    past_key_values = self.get_prompt(batch_size=batch_size)
    prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        past_key_values=past_key_values,
    )
    ...