import torch, math

def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    将相对位置映射到一个桶编号，用于相对注意力机制。
    对较大的绝对相对位置使用较大的桶。所有大于等于max_distance的相对位置都映射到同一个桶，
    所有小于等于负最大距离的相对位置也映射到同一个桶。这应该允许更平滑地进行更长序列的泛化。
    :param relative_position: 一个int32类型的张量，表示相对位置
    :param bidirectional: 布尔值，表示注意力是否是双向的
    :param num_buckets: 整数，桶的数量
    :param max_distance: 整数，最大距离，用于定义桶的边界
    :return 一个与relative_position形状相同的张量，包含[0, num_buckets)范围内的int32值
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2  # 如果是双向的，则将桶数的一半用于正相对位置，另一半用于负相对位置
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets  # 对于正的位置，映射到后半部分桶
        relative_position = torch.abs(relative_position)  # 取相对位置的绝对值
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))  # 对于单向位置，限制负位置

    # 现在的relative_position在范围[0, inf)内

    # 一半的桶用于小的相对位置增量
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact  # 判断哪些位置在小的桶范围内

    # 另一半桶用于更大位置增量的对数桶，位置上限为max_distance
    relative_postion_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)  # 限制最大值为num_buckets-1
    )

    # 根据位置是否较小来选择对应的桶
    relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
    return relative_buckets  # 返回相对位置对应的桶编号
