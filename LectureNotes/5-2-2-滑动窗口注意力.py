import torch

# built sliding window attention
def get_window_mask(seq_len, window):
    mask = torch.ones(seq_len, seq_len)
    mask = torch.tril(mask)
    win_mask = torch.ones(seq_len - window, seq_len - window)
    win_mask = 1.0 - torch.tril(win_mask)
    mask[window:, :seq_len - window] = win_mask
    return mask
print(get_window_mask(7, 3)) # test
window_mask = get_window_mask(t, 8)

# simplify multihead attention
S = Q @ K.transpose(1,2) / math.sqrt(dim)
S = F.softmax(S, dim = -1)
S = S * window_mask #
print(S)
o_win = S @ V
print(o_win.shape)