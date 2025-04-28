import torch
from scipy import stats
import numpy as np
from openpyxl import load_workbook
import torch.nn.functional as F

def safe_log(x, EPS = 1e-8):
    return (x + EPS).log()

def get_p_value(args, cdf, is_dead, device='cpu'):
    EPS = 1e-8
    order = torch.argsort(cdf)
    cdf = cdf[order]
    F_sorted = cdf.unsqueeze(1)
    is_dead = is_dead[order].unsqueeze(1)
    N = cdf.shape[0]

    is_alive = (1 - is_dead).float()

    denom = 1 - F_sorted + EPS
    weight = is_alive / denom
    F_weight = F_sorted * weight

    cum_weight = torch.cumsum(weight, dim=0)
    cum_F_weight = torch.cumsum(F_weight, dim=0)

    cum_weight_shifted = F.pad(cum_weight[:-1], (0, 0, 1, 0), value=0.0)
    cum_F_weight_shifted = F.pad(cum_F_weight[:-1], (0, 0, 1, 0), value=0.0)

    ecdf_cens = F_sorted * cum_weight_shifted - cum_F_weight_shifted

    ecdf_dead = torch.cumsum(is_dead, dim=0)
    ecdf_upper = (ecdf_dead + ecdf_cens) / N
    ecdf_lower = ecdf_upper - is_dead / N

    KS_upper1 = torch.abs(ecdf_upper - F_sorted)
    KS_lower1 = torch.abs(ecdf_lower - F_sorted)
    KS_error1 = torch.max(torch.cat([KS_upper1, KS_lower1], dim=1), dim=1).values

    KS = torch.max(KS_error1)

    return KS

