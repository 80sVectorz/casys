KV_TIMESTAMP = 'kval_timestamp'
KV_POS_AX = 'kval_p_ax'
KV_SIZE_AX = 'kval_s_ax'
KV_WR_IDX = 'kval_wr_idx'

BASE_RESERVED_NAMES = {
    KV_TIMESTAMP,
    KV_POS_AX, KV_SIZE_AX,
    KV_WR_IDX,
}

def f_kv_pos_ax(ax: int) -> str:
    return f'{KV_POS_AX}{ax}'

def f_kv_size_ax(ax: int) -> str:
    return f'{KV_SIZE_AX}{ax}'

def f_kv_wr_idx(buffer: str) -> str:
    return f'{KV_WR_IDX}_{buffer}'