
import numpy as np

# 1D DWT
def _dwt1d(arr, H_0, H_1):
    M, N = arr.shape
    N_half = N // 2

    # 转换为 numpy 数组
    H_0 = np.array(H_0, dtype=np.float64)
    H_1 = np.array(H_1, dtype=np.float64)

    # 1D DWT 边界延拓
    padded = np.pad(arr, ((0, 0), (2, 2)), mode='reflect')

    # 初始化输出的低频 L 和高频 H 分量
    L = np.zeros((M, N_half), dtype=np.float64)
    H = np.zeros((M, N_half), dtype=np.float64)

    #滤波和下采样
    for k in range(5):
        L += H_0[k] * padded[:, k: 2 * N_half + k: 2]
    for k in range(3):
        H += H_1[k] * padded[:, k + 2: 2 * N_half + k + 2: 2]

    # 水平拼接低频与高频部分
    return np.hstack((L, H))

# 单层2D DWT函数
def DWTOneLevel(data, region, H_0, H_1):
    r0, c0, r1, c1 = region
    sub_data = data[r0:r1, c0:c1].astype(np.float64).copy()

    # 1. 对行进行 1D DWT
    sub_data = _dwt1d(sub_data, H_0, H_1)

    # 2. 对列进行 1D DWT (转置后处理行，再转置回来)
    sub_data = _dwt1d(sub_data.T, H_0, H_1).T

    # 将处理结果写回原区域
    out_data = data.astype(np.float64).copy()
    out_data[r0:r1, c0:c1] = sub_data
    return out_data

# DWT函数
def DWT(image, num, H_0, H_1):
    current_data = image.astype(np.float64).copy()
    h, w = current_data.shape

    r0, c0 = 0, 0
    r1, c1 = h, w

    # 逐步对左上角的 LL 子带进行迭代分解
    for level in range(num):
        region = [r0, c0, r1, c1]
        current_data = DWTOneLevel(current_data, region, H_0, H_1)

        r1 = r0 + (r1 - r0) // 2
        c1 = c0 + (c1 - c0) // 2

    return current_data

# 1D IDWT
def _idwt1d(arr, G_0, G_1):
    M, N = arr.shape
    N_half = N // 2

    G_0 = np.array(G_0, dtype=np.float64)
    G_1 = np.array(G_1, dtype=np.float64)

    # 1. 分离低频 L 和高频 H
    L = arr[:, :N_half]
    H = arr[:, N_half:]

    # 2. 上采样（偶数位置插 L，奇数位置插 H）
    L_up = np.zeros((M, N), dtype=np.float64)
    H_up = np.zeros((M, N), dtype=np.float64)
    L_up[:, 0::2] = L
    H_up[:, 1::2] = H

    # 3. 边界对称延拓（双侧各延拓 2 个采样点，适应长度为 5 的滤波器）
    L_up_pad = np.pad(L_up, ((0, 0), (2, 2)), mode='reflect')
    H_up_pad = np.pad(H_up, ((0, 0), (2, 2)), mode='reflect')


    X_L = np.zeros((M, N), dtype=np.float64)
    X_H = np.zeros((M, N), dtype=np.float64)

    # 4. 滤波重构
    # G_0 长度为 3，中心对齐
    for k in range(3):
        X_L += G_0[k] * L_up_pad[:, k + 1: k + 1 + N]

    # G_1 长度为 5，中心对齐
    for k in range(5):
        X_H += G_1[k] * H_up_pad[:, k: k + N]

    return X_L + X_H

# 单层2D逆DWT函数
def IDWTOneLevel(data, region, G_0, G_1):
    r0, c0, r1, c1 = region
    sub_data = data[r0:r1, c0:c1].copy().astype(np.float64)

    # 1. 对列进行 1D IDWT (通过转置矩阵实现)
    sub_data = _idwt1d(sub_data.T, G_0, G_1).T

    # 2. 对行进行 1D IDWT
    sub_data = _idwt1d(sub_data, G_0, G_1)

    # 将重构后的数据写回原区域
    out_data = data.astype(np.float64).copy()
    out_data[r0:r1, c0:c1] = sub_data
    return out_data

# 逆DWT函数
def IDWT(image, num, G_0, G_1):
    current_data = image.astype(np.float64).copy()
    h, w = current_data.shape

    # 逆变换的迭代顺序从最深层开始（自底向上）
    for level in reversed(range(num)):
        # 计算当前层重构的目标大小
        size_h = h // (2 ** level)
        size_w = w // (2 ** level)

        region = [0, 0, size_h, size_w]
        current_data = IDWTOneLevel(current_data, region, G_0, G_1)

    return current_data


# 量化函数
def quantize(coeffs, q):

    q_coeffs = np.sign(coeffs) * np.floor(np.abs(coeffs) / q)
    return q_coeffs.astype(np.int32)

# 逆量化函数
def inverse_quantize(q_coeffs, q):
    # 重建+区间中值减少误差
    recon_coeffs = q_coeffs.astype(np.float64) * q + np.sign(q_coeffs) * (q / 2.0)
    return recon_coeffs

# 对最小子带LL5进行预测
def low_subband_predict(LL5):
    pred_error = np.zeros_like(LL5, dtype=np.int32)

    # 左上角顶点：保持不变
    pred_error[0, 0] = LL5[0, 0]

    # 第一行只有左边邻居，做一维水平差分
    pred_error[0, 1:] = LL5[0, 1:] - LL5[0, :-1]

    # 第一列只有上边邻居，做一维垂直差分
    pred_error[1:, 0] = LL5[1:, 0] - LL5[:-1, 0]

    # 内部区域：使用 2D 预测公式 (A + B - C)
    A = LL5[1:, :-1]  # 左边
    B = LL5[:-1, 1:]  # 上边
    C = LL5[:-1, :-1]  # 左上

    pred_val = A + B - C  # 预测值
    pred_error[1:, 1:] = LL5[1:, 1:] - pred_val

    return pred_error

# 重构最小子带LL5
def low_subband_reconstruct(pred_error):
    h, w = pred_error.shape
    recon = np.zeros_like(pred_error, dtype=np.float64)

    # 恢复左上角顶点
    recon[0, 0] = pred_error[0, 0]

    # 恢复第一行 (利用左边累加)
    for j in range(1, w):
        recon[0, j] = recon[0, j - 1] + pred_error[0, j]

    # 恢复第一列 (利用上边累加)
    for i in range(1, h):
        recon[i, 0] = recon[i - 1, 0] + pred_error[i, 0]

    # 内部区域自左向右、自上而下逐个递推恢复
    for i in range(1, h):
        for j in range(1, w):
            A = recon[i, j - 1]  # 已恢复的左边
            B = recon[i - 1, j]  # 已恢复的上边
            C = recon[i - 1, j - 1]  # 已恢复的左上

            pred_val = A + B - C
            recon[i, j] = pred_error[i, j] + pred_val

    return recon

# 光栅扫描函数
def raster_scan(ll_band):
    # 展平LL5
    flat_coeffs = ll_band.flatten()
    scan_sequence = []

    # 遍历每一个系数
    for val in flat_coeffs:
        val_int = int(val)
        if val_int == 0:
            scan_sequence.append(0)  # 0 保持不变
        else:
            # 计算二进制长度 (size)
            size = abs(val_int).bit_length()
            scan_sequence.append((size, val_int))  #(size, amplitude)

    return scan_sequence

# 逆光删扫描函数
def inverse_raster_scan(scan_sequence, shape=(16, 16)):
    flat_coeffs = []

    # 将 (size, amplitude) 还原为原本的整数数值
    for item in scan_sequence:
        if isinstance(item, tuple):
            # 如果是二元组，取第二项
            flat_coeffs.append(item[1])
        else:
            # 如果是 0，保持不变
            flat_coeffs.append(item)

    # 将 1D 列表转换回原形状的 2D 矩阵
    ll_band = np.array(flat_coeffs).reshape(shape)
    return ll_band

# 零树扫描
def get_subbands_info(img_size=512, num_levels=5):
    subbands = []
    for lvl in range(num_levels, 0, -1):
        sz = img_size // (2 ** lvl)
        # HL (右上), LH (左下), HH (右下)
        subbands.append({'lvl': lvl, 'type': 'HL', 'r0': 0, 'r1': sz, 'c0': sz, 'c1': 2 * sz})
        subbands.append({'lvl': lvl, 'type': 'LH', 'r0': sz, 'r1': 2 * sz, 'c0': 0, 'c1': sz})
        subbands.append({'lvl': lvl, 'type': 'HH', 'r0': sz, 'r1': 2 * sz, 'c0': sz, 'c1': 2 * sz})
    return subbands

# =========================================================================
# 2. 计算 DWT 矩阵对应的初始阈值 T0
# =========================================================================
def get_initial_threshold(coeffs, img_size=512):
    """ 计算所有高频量化系数最大绝对值的 2^floor(log2(max)) """
    freq_mask = np.ones((img_size, img_size), dtype=bool)
    freq_mask[0:16, 0:16] = False

    high_max = np.max(np.abs(coeffs[freq_mask]))
    if high_max == 0:
        return 1
    return 2 ** int(np.floor(np.log2(high_max)))


# =========================================================================
# 3. 递归检查当前位置在阈值 T 下是否有显著后代
# =========================================================================
def has_significant_descendant(r, c, sub, T, coeffs, subbands):
    lvl = sub['lvl']
    if lvl == 1:
        return False  # 第一层无后代

    r_rel = r - sub['r0']
    c_rel = c - sub['c0']

    # 获取下一层(lvl-1)相同方向的子带
    child_sub = next(s for s in subbands if s['lvl'] == lvl - 1 and s['type'] == sub['type'])

    # 下一层对应的 2x2 区域左上角坐标
    cr0 = child_sub['r0'] + 2 * r_rel
    cc0 = child_sub['c0'] + 2 * c_rel

    for i in (0, 1):
        for j in (0, 1):
            cr, cc = cr0 + i, cc0 + j
            if abs(coeffs[cr, cc]) >= T:
                return True
            # 递归检查孙子及更后代节点
            if has_significant_descendant(cr, cc, child_sub, T, coeffs, subbands):
                return True
    return False


# =========================================================================
# 4. 当判定为 ZTR 时，将子树在 "本轮" 标记为跳过 (利用栈实现非递归标记)
# =========================================================================
def mark_subtree_skip(r, c, sub, skip_curr, subbands):
    lvl = sub['lvl']
    if lvl == 1:
        return

    r_rel = r - sub['r0']
    c_rel = c - sub['c0']
    child_sub = next(s for s in subbands if s['lvl'] == lvl - 1 and s['type'] == sub['type'])

    cr0 = child_sub['r0'] + 2 * r_rel
    cc0 = child_sub['c0'] + 2 * c_rel

    # 初始化栈，存入 4 个子女
    stack = []
    for i in (0, 1):
        for j in (0, 1):
            stack.append((cr0 + i, cc0 + j, child_sub))

    while stack:
        curr_r, curr_c, curr_sub = stack.pop()
        if skip_curr[curr_r, curr_c]:
            continue
        skip_curr[curr_r, curr_c] = True

        curr_lvl = curr_sub['lvl']
        if curr_lvl > 1:
            child_sub_next = next(s for s in subbands if s['lvl'] == curr_lvl - 1 and s['type'] == curr_sub['type'])
            curr_r_rel = curr_r - curr_sub['r0']
            curr_c_rel = curr_c - curr_sub['c0']
            ccr0 = child_sub_next['r0'] + 2 * curr_r_rel
            ccc0 = child_sub_next['c0'] + 2 * curr_c_rel
            for i in (0, 1):
                for j in (0, 1):
                    stack.append((ccr0 + i, ccc0 + j, child_sub_next))


# =========================================================================
# 5. 前向 2D 矩阵多遍 EZW 扫描 (Encoder 端)
# =========================================================================
def ezw_scan_2d(coeffs, initial_threshold, img_size=512, num_levels=5):
    subbands = get_subbands_info(img_size, num_levels)

    # 建立 2D 索引方便对应
    for idx, sb in enumerate(subbands):
        sb['idx'] = idx

    # 永久显著图 (Permanent Significance Map)
    sig = np.zeros((img_size, img_size), dtype=bool)
    symbols = []

    T = initial_threshold

    while T >= 1:
        # 本轮跳过图 (每轮重置)
        skip_curr = np.zeros((img_size, img_size), dtype=bool)

        # 按照 Fig. 5 规定的子带扫描顺序
        for sb in subbands:
            for r in range(sb['r0'], sb['r1']):
                for c in range(sb['c0'], sb['c1']):
                    # 1. 如果已经被标记为永久显著，跳过
                    if sig[r, c]:
                        continue
                    # 2. 如果在本轮属于零树的一部分被跳过，跳过
                    if skip_curr[r, c]:
                        continue

                    val = int(coeffs[r, c])
                    if abs(val) >= T:
                        # 显著节点：输出 'P' 或 'N'
                        symbols.append('P' if val > 0 else 'N')
                        sig[r, c] = True
                    else:
                        # 不显著节点：判断本轮是否有显著后代
                        if has_significant_descendant(r, c, sb, T, coeffs, subbands):
                            symbols.append('I')  # 隔离零
                        else:
                            symbols.append('Z')  # 零树根
                            # 本轮标记整棵子树跳过
                            mark_subtree_skip(r, c, sb, skip_curr, subbands)
        T //= 2  # 阈值减半

    return symbols


# 逆零树扫描
# =========================================================================
# 6. 逆向多遍 EZW 扫描重构 (Decoder 端)
# =========================================================================
def ezw_inverse_scan_2d(symbols, initial_threshold, img_size=512, num_levels=5):
    subbands = get_subbands_info(img_size, num_levels)

    recon_coeffs = np.zeros((img_size, img_size), dtype=np.float64)
    sig = np.zeros((img_size, img_size), dtype=bool)

    sym_iter = iter(symbols)
    T = initial_threshold

    while T >= 1:
        # 本轮跳过图 (每轮重置)
        skip_curr = np.zeros((img_size, img_size), dtype=bool)

        for sb in subbands:
            for r in range(sb['r0'], sb['r1']):
                for c in range(sb['c0'], sb['c1']):
                    if sig[r, c]:
                        continue
                    if skip_curr[r, c]:
                        continue

                    try:
                        symbol = next(sym_iter)
                    except StopIteration:
                        break

                    if symbol == 'P':
                        recon_coeffs[r, c] = float(T)  # 初始化重构值
                        sig[r, c] = True
                    elif symbol == 'N':
                        recon_coeffs[r, c] = -float(T)
                        sig[r, c] = True
                    elif symbol == 'I':
                        recon_coeffs[r, c] = 0.0
                    elif symbol == 'Z':
                        recon_coeffs[r, c] = 0.0
                        # 标记当前阈值轮次下，该子树在解码中同样跳过
                        mark_subtree_skip(r, c, sb, skip_curr, subbands)
        T //= 2

    return recon_coeffs
