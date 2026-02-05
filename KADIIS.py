from collections import defaultdict
import numpy as np
from scipy.spatial.distance import pdist, squareform


def find_missing_data(data, missing_data):
    n, m = data.shape
    missing_data_dict = defaultdict(list)

    for i in range(n):
        for j in range(m):
            if data[i, j] == missing_data:
                missing_data_dict[j].append(i)
                data[i, j] = 0.00

    return dict(missing_data_dict)

def gaussian_matrix(Data, r, attr_list, missing_data_dict):
    attr_list = [attr_list] if not isinstance(attr_list, (list, tuple)) else attr_list
    missing_attr_list = list(missing_data_dict.keys())
    mis_attr_index = list(set(attr_list) & set(missing_attr_list))
    mis_data_index = []
    for i in mis_attr_index:
        mis_data_index.extend(missing_data_dict[i])
    mis_data_index = list(dict.fromkeys(mis_data_index))

    temp = pdist(Data, 'euclidean')
    temp = squareform(temp)
    temp = np.exp(-(temp ** 2) / r)

    temp[:, mis_data_index] = 1
    temp[mis_data_index, :] = 1
    np.fill_diagonal(temp, 1)
    return temp


def IKFE(matrix):
    n = matrix.shape[0]
    result = 0
    for i in range(n):
        row_sum = np.sum(matrix[i, :])
        result += row_sum / n * np.log(1 / row_sum)
    result = -result
    return result


def calRP(matrix, m):
    n = matrix.shape[0]
    row_sum = matrix.sum(axis=1)
    row_sum = row_sum.tolist()  # 保持列表形式
    min_sum = min(row_sum)
    max_sum = max(row_sum)
    RP_list = []
    for i in range(n):
        rp = np.clip(row_sum[i] * m / n * (row_sum[i] - min_sum + 1) / (max_sum - min_sum + 2), 0, 1)
        RP_list.append(rp)
    return RP_list


def calculate_rho(R, X_values):
    # 4.1. Compute the fuzzy lower approximation
    a = 1 - R
    b = X_values
    lower = sum((np.minimum(a + b - np.multiply(a, b) + np.multiply(np.sqrt(2 * a - np.multiply(a, a)),
                                                                    np.sqrt(2 * b - np.multiply(b, b))),
                            1)).min(-1))

    # 4.2. Compute the fuzzy upper approximation
    a = R
    upper = sum((np.maximum(
        np.multiply(a, b) - np.multiply(np.sqrt(1 - np.multiply(a, a)), np.sqrt(1 - np.multiply(b, b))),
        0)).max(-1))
    # lower = np.min(np.maximum(1 - R, X_values), axis=1)
    # upper = np.max(np.minimum(R, X_values), axis=1)
    rho = 1 - np.sum(lower) / np.sum(upper)
    return rho


def KADIIS(data, delta, lamda, missing_attr_list):
    data = data.astype(float)
    n, m = data.shape
    LA = np.arange(0, m)

    # 预计算全局矩阵和IKFE，用于排序
    global_matrix = gaussian_matrix(data, delta, np.arange(0, m).tolist(), missing_attr_list)
    IKFE_global = IKFE(global_matrix)

    # 预计算所有单属性和多属性矩阵
    SingleAttr_list = []
    IKFEa_list = []

    for l in range(m):
        SingleAttr = gaussian_matrix((np.matrix(data[:, l])).T, delta, l, missing_attr_list)
        SingleAttr_list.append(SingleAttr)
        IKFEa_list.append(IKFE(SingleAttr))

    MultiAttr_list = []
    weight_list = []

    for l in range(m):
        # 移除l
        lA_d = np.setdiff1d(LA, l)
        MultiAttr = gaussian_matrix(data[:, lA_d], delta, lA_d.tolist(), missing_attr_list)
        MultiAttr_list.append(MultiAttr)
        weight_list.append(IKFE_global - IKFE(MultiAttr))

    # 排序属性
    sorted_attr = np.argsort(weight_list)

    # 预计算AS矩阵和IKFE
    AS_matrix_list = []
    IKFEAS_list = []
    current_AS = sorted_attr.copy()

    for i in range(m):
        AS_matrix = gaussian_matrix(data[:, current_AS], delta, current_AS.tolist(), missing_attr_list)
        AS_matrix_list.append(AS_matrix)
        IKFEAS_list.append(IKFE(AS_matrix))
        if len(current_AS) > 1:
            current_AS = current_AS[1:]

    # 预计算array1和array2
    array1 = np.ones(n, dtype=int)

    # 计算SIG
    SIG_aj_xi_list = []
    SIG_Aj_xi_list = []

    for i in range(n):
        array2 = np.ones(n, dtype=int)
        array2[i] = 0

        SIG_aj_xi = []
        SIG_Aj_xi = []

        for j in range(m):
            index = sorted_attr[j]

            # 保持原始矩阵删除方式
            mat1 = SingleAttr_list[index]
            mat1_del_xi = np.delete(np.delete(mat1, i, axis=0), i, axis=1)
            IKFE_X_aj = IKFE(mat1_del_xi)

            mat2 = AS_matrix_list[j]
            mat2_del_xi = np.delete(np.delete(mat2, i, axis=0), i, axis=1)
            IKFE_X_Aj = IKFE(mat2_del_xi)

            # 计算rho
            Umat1 = SingleAttr_list[index]
            Umat2 = MultiAttr_list[index]

            rho_aj_U = calculate_rho(Umat1, array1)
            rho_aj_X = calculate_rho(Umat1, array2)
            rho_Aj_U = calculate_rho(Umat2, array1)
            rho_Aj_X = calculate_rho(Umat2, array2)

            # 保持原始SIG计算方式
            IKUI_aj_U = IKFEa_list[index] + rho_aj_U
            IKUI_aj_X = IKFE_X_aj + rho_aj_X
            IKUI_Aj_U = IKFEAS_list[j] + rho_Aj_U
            IKUI_Aj_X = IKFE_X_Aj + rho_Aj_X

            sig_aj = max(0, 1 - IKUI_aj_U / (IKUI_aj_X + 1e-10)) if IKUI_aj_X > IKUI_aj_U else 0
            sig_Aj = max(0, 1 - IKUI_Aj_U / (IKUI_Aj_X + 1e-10)) if IKUI_Aj_X > IKUI_Aj_U else 0

            SIG_aj_xi.append(sig_aj)
            SIG_Aj_xi.append(sig_Aj)

        SIG_aj_xi_list.append(SIG_aj_xi)
        SIG_Aj_xi_list.append(SIG_Aj_xi)

    # 计算RP - 保持原始方式
    RP_aj_list = []
    RP_Aj_list = []

    for j in range(m):
        index = sorted_attr[j]
        RP = calRP(SingleAttr_list[index], 1)
        RP_aj_list.append(RP)

        RP = calRP(AS_matrix_list[j], m - j)
        RP_Aj_list.append(RP)

    # 计算AF - 保持原始方式
    AF_list = []

    for i in range(n):
        sum1 = 0
        sum2 = 0
        for j in range(m):
            sum1 += (1 - (SIG_aj_xi_list[i][j]) ** lamda) * (1 - (RP_aj_list[j][i]) ** lamda)
            sum2 += (1 - (SIG_Aj_xi_list[i][j]) ** lamda) * (1 - (RP_Aj_list[j][i]) ** lamda)
        AF = (sum1 + sum2) / (2 * m)
        AF_list.append(AF)

    return AF_list


if __name__ == '__main__':
    delta = 0.1
    trandata = np.array([[2.0000, 0.33, 1.000],
                         [0.0000, "*", 0.25],
                         [2.0000, 0.0000, 0.75],
                         ["*", 0.11, 0],
                         [0.0000, 0.78, 0.5000],
                         [1.0000, 1.0000, "*"]])
    missing_data_dict = find_missing_data(trandata, "*")
    print(KADIIS(trandata, delta, 0.5, missing_data_dict))
