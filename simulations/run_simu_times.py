import sys, os, smtplib

sys.path.append('../')
import pandas as pd
import numpy as np
from time import time
from sklearn.externals.joblib import Parallel, delayed
#from tick.preprocessing.features_binarizer import FeaturesBinarizer
#from tick.survival import CoxRegression, SimuCoxRegWithCutPoints
from binacox import get_p_values_j
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE

from sklearn.preprocessing import Binarizer, OneHotEncoder
def FeaturesBinarizer(X):
    binarizer = Binarizer()
    X_binarized = binarizer.fit_transform(X)
    return X_binarized

from lifelines import CoxPHFitter
def CoxRegression(df, duration_col='time', event_col='event'):
    cox_model = CoxPHFitter()
    cox_model.fit(df, duration_col=duration_col, event_col=event_col)
    return cox_model
def SimuCoxRegWithCutPoints(n_samples):
    np.random.seed(42)
    # Simulate features and survival times
    X = np.random.randn(n_samples, 5)
    durations = np.random.exponential(scale=10, size=n_samples)
    events = np.random.binomial(1, 0.5, size=n_samples)

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['time'] = durations
    df['event'] = events
    return df


def get_times1(n_simu, n_samples, n_features, n_cut_points):
    print("  n_simu=%s" % n_simu)
    seed = n_simu
    simu = SimuCoxRegWithCutPoints(n_samples=n_samples, n_features=n_features,
                                   seed=seed, verbose=False,
                                   n_cut_points=n_cut_points,
                                   shape=2, scale=.1, cov_corr=cov_corr,
                                   sparsity=sparsity)
    X, Y, delta, cut_points, beta_star, S = simu.simulate()

    # Binacox method
    n_cuts = 50
    binarizer = FeaturesBinarizer(n_cuts=n_cuts)
    X_bin = binarizer.fit_transform(X)
    blocks_start = binarizer.blocks_start
    blocks_length = binarizer.blocks_length
    boundaries = binarizer.boundaries['0']

    solver = 'agd'
    learner = CoxRegression(penalty='binarsity', tol=1e-5,
                            solver=solver, verbose=False,
                            max_iter=100, step=0.3,
                            blocks_start=blocks_start,
                            blocks_length=blocks_length,
                            C=25, warm_start=True)
    learner._solver_obj.linesearch = False
    learner.fit(X_bin, Y, delta)
    tac = time()
    time_bina = tac - tic

    # Auto Cutoff Method
    X = np.array(X)
    epsilon = 10
    p1 = np.percentile(X, epsilon)
    p2 = np.percentile(X, 100 - epsilon)
    values_to_test = X[np.where((X <= p2) & (X >= p1))]
    tic = time()
    get_p_values_j(X, 0, Y, delta, values_to_test, epsilon)
    tac = time()
    time_ac_all = tac - tic

    tic = time()
    p1 = np.percentile(X, epsilon)
    p2 = np.percentile(X, 100 - epsilon)
    values_to_test = boundaries[
        np.where((boundaries <= p2) & (boundaries >= p1))]
    get_p_values_j(X, 0, Y, delta, values_to_test, epsilon)
    tac = time()
    time_ac_grid = tac - tic

    return n_samples, time_bina, time_ac_all, time_ac_grid


def get_times2(n_simu, n_samples, n_features, n_cut_points):
    print("  n_simu=%s" % n_simu)
    seed = n_simu
    simu = SimuCoxRegWithCutPoints(n_samples=n_samples, n_features=n_features,
                                   seed=seed, verbose=False,
                                   n_cut_points=n_cut_points,
                                   shape=2, scale=.1, cov_corr=cov_corr,
                                   sparsity=sparsity)
    X, Y, delta, cut_points, beta_star, S = simu.simulate()

    # Binacox method
    tic = time()
    n_cuts = 50
    binarizer = FeaturesBinarizer(n_cuts=n_cuts)
    X_bin = binarizer.fit_transform(X)
    blocks_start = binarizer.blocks_start
    blocks_length = binarizer.blocks_length
    solver = 'agd'
    learner = CoxRegression(penalty='binarsity', tol=1e-5,
                            solver=solver, verbose=False,
                            max_iter=100, step=0.3,
                            blocks_start=blocks_start,
                            blocks_length=blocks_length,
                            C=25, warm_start=True)
    learner._solver_obj.linesearch = False
    learner.fit(X_bin, Y, delta)
    tac = time()

    return tac - tic


# first setting
print("\nBinacox vs. Auto Cutoff computing times")
n_features = 1
n_cut_points = 2
cov_corr = .5
sparsity = .2
N_simu = 100
n_samples_grid = [300, 500, 1000, 2000, 4000]

result_ = pd.DataFrame(columns=["n_samples", "time_bina", "time_ac_all",
                                "time_ac_grid"])
for i, n_samples in enumerate(n_samples_grid):
    print("n_samples: %d/%d " % ((i + 1), len(n_samples_grid)))
    result_n = Parallel(n_jobs=10)(
        delayed(get_times1)(n_simu, n_samples, n_features,
                            n_cut_points)
        for n_simu in range(N_simu))
    result_n = pd.DataFrame(result_n,
                            columns=["n_samples", "time_bina", "time_ac_all",
                                     "time_ac_grid"])
    result_ = result_.append(result_n, ignore_index=True)

result = pd.DataFrame(columns=["n", "method", "time"])
tmp = pd.DataFrame(columns=["n", "method", "time"])
tmp.n = result_.n_samples
tmp.method = "Binacox"
tmp.time = result_.time_bina
result = result.append(tmp, ignore_index=True)

tmp.n = result_.n_samples
tmp.method = "AC all"
tmp.time = result_.time_ac_all
result = result.append(tmp, ignore_index=True)

tmp.n = result_.n_samples
tmp.method = "AC grid"
tmp.time = result_.time_ac_grid
result = result.append(tmp, ignore_index=True)

result.to_json("./results_data/time1")

# second setting
print("\nBinacox computing times = f(p)")
n_samples = 2000
N_features = 25
n_features_min = 2
n_features_max = 100
n_features_grid = np.unique(np.geomspace(n_features_min, n_features_max,
                                         N_features).astype(int))
result = pd.DataFrame()
for i, n_features in enumerate(n_features_grid):
    print("n_features: %d/%d " % ((i + 1), N_features))

    result_n = Parallel(n_jobs=10)(
        delayed(get_times2)(n_simu, n_samples, n_features,
                            n_cut_points)
        for n_simu in range(N_simu))
    result[n_features] = result_n

result.to_json("./results_data/time2")

# compress results and send it by email
os.system('say "computation finished"')
os.system('zip -r results.zip results_data')

send_from = 'simon.bussy@upmc.fr'
send_to = ['simon.bussy@gmail.com']

subject = "computation finished for computing times"
text = "results available \n"
files = "./results.zip"

msg = MIMEMultipart()
msg['From'] = send_from
msg['To'] = COMMASPACE.join(send_to)
msg['Subject'] = subject

msg.attach(MIMEText(text))

with open(files, "rb") as fil:
    part = MIMEApplication(
        fil.read(),
        Name="results.zip"
    )
    part[
        'Content-Disposition'] = 'attachment; filename="results.zip"'
    msg.attach(part)

try:
    smtp = smtplib.SMTP('smtp.upmc.fr')
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()
    print("Successfully sent email")
except smtplib.SMTPException:
    print("Error: unable to send email")
