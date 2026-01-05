import torch

def _get_interleave(n):
    '''
    生成每一个对应head的权重m
    '''
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return _get_interleave_power_of_2(closest_power_of_2) + \
               _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

def _gen_alibi_mask(n_head, max_pos):
    slopes = torch.Tensor(_get_interleave(n_head)) # n_head
    # alibi: [n_head, 1, max_pos]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_pos).unsqueeze(0).unsqueeze(0).expand(
        n_head, -1, -1)
    alibi = alibi.view(n_head, 1, max_pos)
    # _fill_with_neg_inf(torch.zeros([max_pos, max_pos]))：
    # 首先，创建一个形状为[max_pos, max_pos]的零矩阵，然后将矩阵中的所有元素填充为负无穷。
    # torch.triu(): 使用torch.triu函数生成上三角矩阵，将传入矩阵的下半部分（对角线以下）置为0，
    # 只保留上半部分（对角线以上）的负无穷值。1参数表示从主对角线的下一行开始置为0。
    # alibi_mask: [max_pos, max_pos]的上三角矩阵，对角线以上的元素为负无穷，其他位置为0。
    alibi_mask = torch.triu(
        _fill_with_neg_inf(torch.zeros([max_pos, max_pos])), 1
    )
    # alibi_mask起到将上三角矩阵mask的作用
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask