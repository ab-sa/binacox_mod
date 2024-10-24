import sys, os, smtplib

sys.path.append('../')
import pandas as pd
import numpy as np
from sys import stdout, argv
from time import time
#from tick.preprocessing.features_binarizer import FeaturesBinarizer
#from tick.survival import CoxRegression, SimuCoxRegWithCutPoints
from binacox import compute_score, multiple_testing, get_groups, p_value_cut
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


n_features = int(argv[1])
n_cut_points = int(argv[2])
model = argv[3]

N_simu = 100
N_samples = 20
n_samples_max = 4000
n_samples_grid = np.unique(np.geomspace(300, n_samples_max,
                                       N_samples).astype(int))

if model == "binacox":
    result = pd.DataFrame(columns=["n_samples", "cut_points", "S",
                                   "cut_points_estimates", "C_chosen", "time"])
else:
    result = pd.DataFrame(columns=["n_samples", "cut_points", "S",
                                   "cut_points_estimates", "p_values_min",
                                   "n_tested", "p_values_corr", "time"])


def simu_autocutoff(n_simu, n_samples, n_features, n_cut_points):
    seed = n_simu
    cov_corr = .5
    sparsity = .2
    simu = SimuCoxRegWithCutPoints(n_samples=n_samples,
                                   n_features=n_features,
                                   n_cut_points=n_cut_points,
                                   seed=seed, verbose=False,
                                   shape=2, scale=.1, cov_corr=cov_corr,
                                   sparsity=sparsity)
    X, Y, delta, cut_points, beta_star, S = simu.simulate()

    # binarize data
    n_cuts = 50
    binarizer = FeaturesBinarizer(n_cuts=n_cuts)
    binarizer.fit_transform(X)
    boundaries = binarizer.boundaries
    epsilon = 10

    tic = time()
    multiple_testing_rslt = multiple_testing(X, boundaries, Y, delta, epsilon=epsilon)
    # Lausen & Schumacher correction
    p_values_corr, p_values_min, cut_points_estimates = [], [], []
    n_tested = []
    for j in range(n_features):
        p_values_j = multiple_testing_rslt[j]
        n_tested.append(p_values_j.values_to_test.shape[0])
        p_values_min.append(p_values_j.p_values.min())
        p_values_corr.append(
            p_value_cut(p_values_j.p_values, p_values_j.values_to_test,
                        X[:, j], epsilon))
        idx_cut_points = p_values_j.p_values.argmin()
        cut_points_estimate_j = p_values_j.values_to_test[
            idx_cut_points]
        cut_points_estimates.append(cut_points_estimate_j)
    tac = time()

    print(p_values_min)

    return [n_samples, cut_points, S,
            cut_points_estimates, p_values_min,
            n_tested, p_values_corr, tac - tic]


tic_init = time()
n_result = 0
for n_samples_idx, n_samples in enumerate(n_samples_grid):

    if model == "binacox":

        for n_simu in range(N_simu):
            stdout.write("\rn_samples: %s/%s, "
                         "n_simu: %s/%s" % (n_samples_idx + 1,
                                            len(n_samples_grid),
                                            n_simu + 1,
                                            N_simu))
            stdout.flush()
            seed = n_simu
            cov_corr = .5
            sparsity = .2
            simu = SimuCoxRegWithCutPoints(n_samples=n_samples,
                                           n_features=n_features,
                                           n_cut_points=n_cut_points,
                                           seed=seed, verbose=False,
                                           shape=2, scale=.1, cov_corr=cov_corr,
                                           sparsity=sparsity)
            X, Y, delta, cut_points, beta_star, S = simu.simulate()

            # binarize data
            n_cuts = 50
            binarizer = FeaturesBinarizer(n_cuts=n_cuts)
            X_bin = binarizer.fit_transform(X)
            blocks_start = binarizer.blocks_start
            blocks_length = binarizer.blocks_length
            boundaries = binarizer.boundaries

            tic = time()

            solver = 'agd'
            learner = CoxRegression(penalty='binarsity', tol=1e-5,
                                    solver=solver, verbose=False,
                                    max_iter=100, step=0.3,
                                    blocks_start=blocks_start,
                                    blocks_length=blocks_length,
                                    warm_start=True)
            learner._solver_obj.linesearch = False

            # cross-validation
            n_folds = 10
            grid_size = 30
            grid_C = np.logspace(0, 3, grid_size)
            scores_cv = pd.DataFrame(columns=['ll_test', 'test_std'])
            for i, C in enumerate(grid_C):
                stdout.write("\rbinacox n_samples: %s/%s, "
                             "n_simu: %s/%s, "
                             "CV: %d%%" % (n_samples_idx + 1,
                                           len(n_samples_grid),
                                           n_simu + 1,
                                           N_simu,
                                           (i + 1) * 100 / grid_size))
                stdout.flush()
                scores_cv.loc[i] = compute_score(learner, X, X_bin, Y, delta,
                                                 blocks_start,
                                                 blocks_length, boundaries, C,
                                                 n_folds)
            idx_min = scores_cv.ll_test.argmin()
            idx_chosen = min([i for i, j in enumerate(
                list(scores_cv.ll_test <= scores_cv.ll_test.min()
                     + scores_cv.test_std[idx_min])) if j])
            C_chosen = grid_C[idx_chosen]

            # final estimation
            learner.C = C_chosen
            learner.fit(X_bin, Y, delta)
            coeffs = learner.coeffs
            cut_points_estimates = {}
            for j, start in enumerate(blocks_start):
                coeffs_j = coeffs[start:start + blocks_length[j]]
                all_zeros = not np.any(coeffs_j)
                if all_zeros:
                    cut_points_estimate_j = np.array([-np.inf, np.inf])
                    groups_j = blocks_length[j] * [0]
                else:
                    groups_j = get_groups(coeffs_j)
                    jumps_j = np.where(groups_j[1:] - groups_j[:-1] != 0)[0] + 1
                    if len(jumps_j) == 0:
                        cut_points_estimate_j = np.array([-np.inf, np.inf])
                    else:
                        cut_points_estimate_j = boundaries[str(j)][jumps_j]
                        if cut_points_estimate_j[0] != -np.inf:
                            cut_points_estimate_j = np.insert(
                                cut_points_estimate_j, 0,
                                -np.inf)
                        if cut_points_estimate_j[-1] != np.inf:
                            cut_points_estimate_j = np.append(
                                cut_points_estimate_j,
                                np.inf)
                cut_points_estimates[str(j)] = cut_points_estimate_j
            tac = time()

            # save results
            result.loc[n_result] = [n_samples, cut_points, S,
                                    cut_points_estimates, C_chosen, tac - tic]

    if model == "multiple_testing":
        stdout.write("\rn_samples: %s/%s" % (n_samples_idx + 1,
                                             len(n_samples_grid)))
        stdout.flush()

        for n_simu in range(N_simu):
            # save results
            result.loc[n_result] = simu_autocutoff(n_simu, n_samples,
                                                   n_features, n_cut_points)
            n_result += 1

directory = "./results_data/p_%s" % n_features
try:
    os.stat(directory)
except:
    os.mkdir(directory)

result.to_json("%s/results_%s_n_cut_points_%s" % (directory, model,
                                                  n_cut_points))

tac_final = time()
final_msg = "\nDone montecarlo p=%s, n_cut_points=%s " \
            "in %.2e seconds." % (n_features, n_cut_points,
                                  tac_final - tic_init)
print(final_msg)

time = open("./results_data/time.txt", "w")
time.write(final_msg)
time.close()

# compress results and send it by email
os.system('say "computation finished"')
os.system('zip -r results.zip results_data')

send_from = 'simon.bussy@upmc.fr'
send_to = ['simon.bussy@gmail.com']

subject = "computation finished for %s p=%s, K=%s, " % (model, n_features,
                                                        n_cut_points)
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
