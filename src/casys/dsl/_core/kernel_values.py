KV_TIMESTAMP = 'kval_timestamp'

KV_POS_AX = 'kval_p_ax'
KV_SIZE_AX = 'kval_s_ax'

KV_RD_IDX = 'kval_rd_idx'
KV_WR_IDX = 'kval_wr_idx'

KV_N_SIM_STEP_REPEATS = 'kval_n_sim_step_repeats'
KV_I_SIM_STEP_INTERNAL = 'kval_i_sim_step_internal'

BASE_RESERVED_NAMES = {
    KV_TIMESTAMP,
    KV_POS_AX, KV_SIZE_AX,
    KV_RD_IDX,KV_WR_IDX,

    KV_N_SIM_STEP_REPEATS,
    KV_I_SIM_STEP_INTERNAL,
}

def f_kv_pos_ax(ax: int) -> str:
    return f'{KV_POS_AX}{ax}'

def f_kv_size_ax(ax: int) -> str:
    return f'{KV_SIZE_AX}{ax}'

def f_kv_rd_idx(buffer: str) -> str:
    return f'{KV_RD_IDX}_{buffer}'

def f_kv_wr_idx(buffer: str) -> str:
    return f'{KV_WR_IDX}_{buffer}'
