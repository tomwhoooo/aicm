import os
import sys
from optparse import OptionParser
import pandas as pd
import numpy as np
import utils as utils


def main():
    usage = "%prog data1_input data2_input output_prefix"
    parser = OptionParser(usage=usage)
    parser.add_option('--max_iter', dest='max_iter', default=80,
                      help='Number of iterations that AICM shall be run: default = %default')
    parser.add_option('--dropping_rate', dest='dropping_rate', default=0.05,
                      help='Percentage of the data dropped each iteration: default = %default')
    parser.add_option('--reg_method', dest='reg_method', default='l0',
                      help='Regression method that needs to be adopted, currently only support l0: default = %default')
    parser.add_option('--beta_1', dest='beta_1', default=0.1,
                      help='Coefficient of the (first) regularization term')
    parser.add_option('--beta_hard', dest='beta_hard', default=0.1,
                      help='The hard proportional threshold for each value: default = %default')
    parser.add_option('--switch', dest='switch', default=True,
                      help='If switching the role of response and variable matrix or not: default = %default')
    parser.add_option('--print_iter', dest='print_iter', default=20,
                      help='Print iterations during intermediate steps, can be disabled by setting to a huge number: '
                           'default = %default')
    parser.add_option('--discard_NA', dest='discard_NA', default=True,
                      help='Decide if discarding the original missing value at the final stage, default = %default')
    options, args = parser.parse_args()

    data1_input = args[0]
    data2_input = args[1]
    output_prefix = args[2]

    max_iter = int(options.max_iter)
    dropping_rate = float(options.dropping_rate)
    regression_method = str(options.reg_method)
    beta_1 = float(options.beta_1)
    beta_hard = float(options.beta_hard)
    switch = bool(options.switch)
    print_iter = int(options.print_iter)
    discard_NA = bool(options.discard_NA)

    if dropping_rate >= 1 or dropping_rate < 0 or beta_hard >= 1 or beta_hard < 0:
        raise ValueError('Please check your hyperparameter setting!')

    df1 = pd.read_csv(data1_input, index_col=0)
    df2 = pd.read_csv(data2_input, index_col=0)
    ori_corr = np.median(utils.check_spearman_row_correlation(df1, df2))
    print('Original median Spearman row correlation is {}'.format(ori_corr))

    if df1.shape != df2.shape:
        raise ValueError('Please check the dimension of your matrices!')
    df2_imp = utils.fst_iter_imp(df1, df2)
    df1_imp = utils.fst_iter_imp(df2, df1)
    if regression_method == 'l0':
        if switch:
            for i in range(max_iter):
                if i % 2 == 0:
                    df1_imp, df2_imp = utils.rdti(df1_imp, df2_imp, dropping_rate, beta_1, beta_hard)
                else:
                    df2_imp, df1_imp = utils.rdti(df2_imp, df1_imp, dropping_rate, beta_1, beta_hard)
                if i % print_iter == 0:
                    print('We are at iteration {}'.format(i))
        else:
            for i in range(max_iter):
                df1_imp, df2_imp = utils.rdti(df1, df2_imp, dropping_rate, beta)

    if discard_NA:
        for i in range(df1.shape[0]):
            for j in range(df1.shape[1]):
                if np.isnan(df1.iloc[i, j]):
                    df1_imp.iloc[i, j] = np.nan

        for i in range(df2.shape[0]):
            for j in range(df2.shape[1]):
                if np.isnan(df2.iloc[i, j]):
                    df2_imp.iloc[i, j] = np.nan

    new_corr = np.median(utils.check_spearman_row_correlation(df1_imp, df1_imp))
    print('New median Spearman row correlation is {}'.format(new_corr))

    pd.DataFrame.to_csv(df1_imp, output_prefix+'/df1_imp.csv')
    pd.DataFrame.to_csv(df2_imp, output_prefix+'/df2_imp.csv')


if __name__ == '__main__':
    main()
