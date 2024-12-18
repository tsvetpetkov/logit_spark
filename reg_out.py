from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import pyspark.sql.functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression

import pandas as pd
import numpy as np

from operator import add
from functools import reduce

import math


# initialize data, outcome, regressors, and sample weights

data = spark.createDataFrame(pd.read_csv('glm-logit.csv'))

y = 'grade'
X = ['gpa', 'tuce', 'psi']
w = 'wt'


# mirror lower diag matrix helper

def mirror_diag_mat(vars_lbls, vals_dict):
    mat = []
    for i_idx, i in enumerate(vars_lbls):
        row = []
        for j_idx, j in enumerate(vars_lbls):
            if i_idx >= j_idx:
                row.extend([vals_dict[i, j]])
            else:
                row.extend([vals_dict[j, i]])
    mat.extend([row])


# assemble data

data = data.select([y] + X + [w]).dropna()
data = VectorAssembler(inputCols = X, outputCol = 'X').transform(data)


# normalize weights

obs_cnt = data.count()
wt_sum = data.agg(F.sum(F.col(w))).collect()[0][0]

data = data.withColumn(wt, F.col(wt) * obs_cnt / wt_sum)


# collect logit beta coefs

fit = GeneralizedLinearRegression(labelCol = y, featuresCol = 'X', weightCol = w, fitIntercept = True, \
                                  family = 'binomial', link = 'logit', maxIter = 100, tol = 1e-8).fit(data)

expl_vars = ['cons'] + features
beta_coefs = [fit.intercept] + fit.coefficients.tolist()
betas_dict = dict(zip(expl_vars, beta_coefs))


# collect logit preds, resids, and classification

data = data.withColumn('cons', F.lit(1))
data = data.withColumn('Xb', reduce(add, [F.col(i) * betas_dict[i] for i in expl_vars]))

data = data.withColumn('pred', 1 / (1 + F.exp(-F.col('Xb'))))
data = data.withColumn('resid', F.col(y) - F.col('pred'))
data = data.withColumn('class', F.when(F.col('pred') > 0.5, 1).otherwise(0))


# collect logit beta coefs SEs

expl_vars_pairs = [(i, j) for i_idx, i in enumerate(expl_vars) for j_idx, j in enumerate(expl_vars) if j_idx >= j_ind]

sum_prods = data.select([F.sum(F.col(i[0]) * F.col(i[1]) * F.col(w) * F.col('pred') * (1 - F.col('pred'))) for i in expl_vars_pairs]).collect()[0]
sum_prods_dict = dict(zip(expl_vars_pairs, sum_prods))

vcv_mat = np.linalg.inv(mirror_diag_mat(expl_vars, sum_prods_dict))
std_errs_dict = dict(zip(expl_vars, np.sqrt(np.diag(vcv))))


# collect marginal effects

reg_vars = [y] + X
sums = data.select([F.sum(F.col(i) * F.col(w)) for i in reg_vars]).collect()[0]

means = [i / obs_cnt for i in sums]
means_dict = dict(zip(reg_vars, means))

marg_fx_dict = {i: betas_dict[i] * means_dict[y] * (1 - means_dict[y]) for i in expl_vars}


# collect logit standardized beta coefs

prod_sums = data.select([F.sum(F.pow(F.col(i) - means_dict[i], 2) * F.col(w)) for i in reg_vars]).collect()[0]

std_devs = [math.sqrt(i / (obs_cnt - 1)) for i in prod_sums]
std_devs_dict = dict(zip(reg_vars, std_devs))

xb_avg = data.agg(F.sum(F.col('Xb') * F.col(w))).collect()[0][0] / obs_cnt
xb_var = data.agg(F.sum(F.pow(F.col('Xb') - xb_avg, 2) * F.col(w))).collect()[0][0] / (obs_cnt - 1)

ystar_var = xb_var + pow(math.pi, 2) / 3

std_betas_dict = {i: betas_dict[i] * std_devs_dict[i] / math.sqrt(ystar_var) for i in X}

