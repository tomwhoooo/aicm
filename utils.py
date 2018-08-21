import numpy as np
import cvxpy as cvx
import copy as copy
import random as rand
import scipy
import scipy.stats


# Here we are required to perform the first iteration, i.e. no dropping, just impute the one
# with more n/a (y, df2) with the help of the one with less n/a (x, df1).


def simple_lr(x, y):
    a = cvx.Variable(1)
    b = cvx.Variable(1)
    obj = cvx.norm(y - (a * x + b), 2)
    prob = cvx.Problem(cvx.Minimize(obj))
    result = prob.solve(solver = 'SCS')
    a_star = a.value
    b_star = b.value
    return a_star, b_star, result


def slr_imp(x, y):
    na_loc_x = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(x.T)[0]))[0])
    na_loc_y = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(y.T)[0]))[0])
    common_na = list(set(na_loc_x).intersection(na_loc_y))
    all_na = list(set(na_loc_x + na_loc_y))
    x_no_na = np.delete(x, all_na)
    y_no_na = np.delete(y, all_na)
    if len(np.matrix.tolist(x_no_na)[0]) != 0 and len(np.matrix.tolist(y_no_na)[0]) != 0:
        a, b, obj = simple_lr(x_no_na, y_no_na)
        imp_y = a * x + b
        new_y = copy.copy(y)
        for i in na_loc_y:
            new_y[i, 0] = imp_y[i, 0]
    else:
        new_y = copy.copy(y)
        obj = 10000
    return new_y, obj


def fst_iter_imp(df1, df2):
    nrow = df1.shape[0]
    ncol = df1.shape[1]
    df2_new = copy.copy(df2)
    for i in range(ncol):
        vector_x = np.matrix(df1.iloc[:, i].as_matrix()).T
        vector_y = np.matrix(df2.iloc[:, i].as_matrix()).T
        new_y, obj = slr_imp(vector_x, vector_y)
        df2_new.iloc[:, i] = new_y
    return df2_new


# Random drop the elements of a matrix to N/A, return the set dropped indexes and the dropped dataframe
def random_dropping(data_frame, dropping_rate):
    new_frame = copy.copy(data_frame)
    nrow = data_frame.shape[0]
    ncol = data_frame.shape[1]
    mylist = [[rand.randint(0, nrow-1), rand.randint(0, ncol-1)] for k in range(round(nrow * ncol * dropping_rate))]
    for i in range(len(mylist)):
        new_frame.iloc[mylist[i][0], mylist[i][1]] = np.NaN
    return mylist, new_frame


# Reconstruct the index sets to a nested list
def recons_index(data_frame, mylist):
    ncol = data_frame.shape[1]
    lst = [[] for _ in range(ncol)]
    for i in range(len(mylist)):
        lst[mylist[i][1]].append(mylist[i][0])
    return lst


# Force a hard constraint on departure of the data
def hard_constraint(input_, comparison, beta):
    if input_ >= comparison * (1 + beta):
        input_ = comparison * (1 + beta)
    if input_ <= comparison * (1 - beta):
        input_ = comparison * (1 - beta)
    return input_


# This is the main part of solving the optimization problem
# Input: a vector of the form, i.e.:
# x = np.matrix(([1, 2, 3, 4, 6, np.NaN, 5, 9, np.NaN])).T
# y = np.matrix(([3, 2, 1, np.NaN, np.NaN, 4, np.NaN, 12, np.NaN])).T
# Note that it is suggested that y has more N/A's than x
# Maybe we need to add warning on solver
def create_prob(x, y, x_truncated, y_0_truncated, beta):
    m = len(x)
    a = cvx.Variable(1)
    b = cvx.Variable(1)
    obj = (1 / m) * cvx.norm(y - (a * x + b), 2) + beta * cvx.norm((a * x_truncated + b) - y_0_truncated, 'inf')
    prob = cvx.Problem(cvx.Minimize(obj))
    result = prob.solve(solver='SCS')
    a_star = a.value
    b_star = b.value
    return a_star, b_star, result


def imp_cons(x, y, y_0, beta, delete_list):
    na_loc_x = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(x.T)[0]))[0])
    na_loc_y = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(y.T)[0]))[0])
    # Common NaN of the two
    common_na = list(set(na_loc_x).intersection(na_loc_y))
    all_na = list(set(na_loc_x + na_loc_y))
    x_no_na = np.delete(x, all_na)
    y_no_na = np.delete(y, all_na)
    x_truncated_na = np.take(x, delete_list)
    y_0_truncated_na = np.take(y_0, delete_list)
    na_loc_x_ = np.ndarray.tolist(np.where(np.isnan(np.ndarray.tolist(x_truncated_na)[0]))[0])
    na_loc_y_ = np.ndarray.tolist(np.where(np.isnan(np.ndarray.tolist(y_0_truncated_na)[0]))[0])
    all_na_ = list(set(na_loc_x_ + na_loc_y_))
    x_truncated = np.delete(x_truncated_na, all_na_)
    y_0_truncated = np.delete(y_0_truncated_na, all_na_)
    # x_truncated = x_truncated_na[~np.isnan(x_truncated_na)]
    # y_0_truncated = y_0_truncated_na[~np.isnan(y_0_truncated_na)]
    if(len(np.matrix.tolist(x_no_na)[0]) != 0 
       and len(np.matrix.tolist(y_no_na)[0]) != 0 
       and len(np.matrix.tolist(x_truncated)[0]) != 0 
       and len(np.matrix.tolist(y_0_truncated)[0]) != 0):
            a, b, obj = create_prob(x_no_na, y_no_na, x_truncated, y_0_truncated, beta)
            if (not a) or (not b):
                new_y = copy.copy(y)
                obj = 10000
            else:
                imp_y = a * x + b
                new_y = copy.copy(y)
                for i in na_loc_y:
                    new_y[i, 0] = imp_y[i, 0]
    else:
        new_y = copy.copy(y)
        obj = 10000
    return new_y, obj


# Apply imp_cons with some corner cases
def rdti(df1, df2, dropping_rate, beta, beta_hard=0.1, hard_threshold=True):
    nrow = df1.shape[0]
    ncol = df1.shape[1]
    drop_index, df2_dropped = random_dropping(df2, dropping_rate)
    drop_list = recons_index(df1, drop_index)
    df2_new = copy.copy(df2)
    for i in range(ncol):
        vector_x = np.matrix(df1.iloc[:, i].as_matrix()).T
        vector_y = np.matrix(df2_dropped.iloc[:, i].as_matrix()).T
        vector_y_0 = np.matrix(df2.iloc[:, i].as_matrix()).T
        delete_list = drop_list[i]
        if len(vector_x) != 0 and len(vector_y) != 0 and len(vector_y_0) != 0:
            new_y, obj = imp_cons(vector_x, vector_y, vector_y_0, beta, delete_list)
            df2_new.iloc[:, i] = new_y
        else:
            pass
    if hard_threshold:
        for i in range(nrow):
            for j in range(ncol):
                if not np.isnan(df2_new.iloc[i, j]):
                    df2_new.iloc[i, j] = hard_constraint(df2_new.iloc[i, j], df2.iloc[i, j], beta_hard)
    return df1, df2_new


# df1: the one with less N/A's
# df2: the one with more N/A's


# def aicm(df1, df2, iteration=80, dropping_rate=0.05, beta=0.1, neighbor = 'None', switch=True, beta_hard=0.1,
#          hard_threshold=True):
#     df2_imp = fst_iter_imp(df1, df2)
#     if switch:
#         for i in range(iteration):
#             if i % 2 == 0:
#                 df1, df2_imp = rdti(df1, df2_imp, dropping_rate, beta, beta_hard, hard_threshold)
#             else:
#                 df2_imp, df1 = rdti(df2_imp, df1, dropping_rate, beta, beta_hard, hard_threshold)
#     else:
#         for i in range(iteration):
#             df1, df2_imp = rdti(df1, df2_imp, dropping_rate, beta)
#     return df1, df2_imp
#
#
# def aicm_alt(df1, df2, iteration=20, dropping_rate=0.1, beta=0.5, neighbor = 'None', switch=True, beta_hard=0.1,
#          hard_threshold=False):
#     df2_imp = fst_iter_imp(df1, df2)
#     df1_imp = fst_iter_imp(df2, df1)
#     if switch:
#         for i in range(iteration):
#             if i % 2 == 0:
#                 df1_imp, df2_imp = rdti(df1_imp, df2_imp, dropping_rate, beta, beta_hard, hard_threshold)
#             else:
#                 df2_imp, df1_imp = rdti(df2_imp, df1_imp, dropping_rate, beta, beta_hard, hard_threshold)
#     else:
#         for i in range(iteration):
#             df1_imp, df2_imp = rdti(df1, df2_imp, dropping_rate, beta)
#     return df1_imp, df2_imp


def check_row_correlation(df1, df2):
    corr_vector = np.zeros(df1.shape[0])
    for i in range(df1.shape[0]):
        vector_x = np.matrix(df1.iloc[i,:].as_matrix()).T
        vector_y = np.matrix(df2.iloc[i,:].as_matrix()).T
        na_loc_x = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(vector_x.T)[0]))[0])
        na_loc_y = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(vector_y.T)[0]))[0])
        all_na = list(set(na_loc_x + na_loc_y))
        x_no_na = np.delete(vector_x, all_na)
        y_no_na = np.delete(vector_y, all_na)
        corr_vector[i] = scipy.stats.pearsonr(x_no_na.T, y_no_na.T)[0][0]
    return(corr_vector)


def check_spearman_row_correlation(df1, df2):
    corr_vector = np.zeros(df1.shape[0])
    for i in range(df1.shape[0]):
        vector_x = np.matrix(df1.iloc[i,:].as_matrix()).T
        vector_y = np.matrix(df2.iloc[i,:].as_matrix()).T
        na_loc_x = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(vector_x.T)[0]))[0])
        na_loc_y = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(vector_y.T)[0]))[0])
        all_na = list(set(na_loc_x + na_loc_y))
        x_no_na = np.delete(vector_x, all_na)
        y_no_na = np.delete(vector_y, all_na)
        corr_vector[i] = scipy.stats.spearmanr(x_no_na.T, y_no_na.T)[0]
    return(corr_vector)


def check_spearman_col_correlation(df1, df2):
    corr_vector = np.zeros(df1.shape[1])
    for i in range(df1.shape[1]):
        vector_x = np.matrix(df1.iloc[:,i].as_matrix()).T
        vector_y = np.matrix(df2.iloc[:,i].as_matrix()).T
        na_loc_x = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(vector_x.T)[0]))[0])
        na_loc_y = np.ndarray.tolist(np.where(np.isnan(np.matrix.tolist(vector_y.T)[0]))[0])
        all_na = list(set(na_loc_x + na_loc_y))
        x_no_na = np.delete(vector_x, all_na)
        y_no_na = np.delete(vector_y, all_na)
        corr_vector[i] = scipy.stats.spearmanr(x_no_na.T, y_no_na.T)[0]
    return(corr_vector)


def throw_abnormal_IC50(dataset, threshold=15):
    dataset = dataset.mask(dataset > threshold)
    return dataset


def mask_abnormal_IC50(dataset, threshold=8):
    dataset = dataset.clip_upper(threshold)
    return dataset


## This function returns a nested list of where ith list is that
## ith ROW's index on where the data set has the clip threshold value

def get_clip_index(dataset, clip_threshold=8):
    # return a nested list that contains all values matching the clip_threshold
    nrow = dataset.shape[0]
    lst = []
    for i in range(nrow):
        idx_list = np.where(GDSC_V_mask.iloc[i, :] == clip_threshold)[0].tolist()
        lst.append(idx_list)
    return lst
