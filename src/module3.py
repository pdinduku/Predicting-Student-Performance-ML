import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_excel("U.xlsx")
df.head()

df_subjects = pd.read_excel("V.xlsx")
df_join = pd.merge(df_subjects, df, how='inner', on='Paper_ID')
df_join.head()

def get_marks_matrix(df, train_size=0.75):
    student_to_row = {}
    subject_to_column = {}
    df_values = df.values
    n_dims = 10
    parameters = {}

    uniq_students = np.unique(df_values[:, 17])
    uniq_subjects = np.unique(df_values[:, 0])

    for i, Student_ID in enumerate(uniq_students):
        student_to_row[Student_ID] = i

    for j, Paper_ID in enumerate(uniq_subjects):
        subject_to_column[Paper_ID] = j

    n_students = len(uniq_students)
    n_subjects = len(uniq_subjects)

    R = np.zeros((n_students, n_subjects))

    df_copy = df.copy()
    train_set = df_copy.sample(frac=train_size, random_state=50)
    test_set = df_copy.drop(train_set.index)

    for index, row in train_set.iterrows():
        i = student_to_row[row.Student_ID]
        j = subject_to_column[row.Paper_ID]
        R[i, j] = row.Marks

    return R, train_set, test_set, n_dims, n_students, n_subjects, student_to_row, subject_to_column

R, train_set, test_set, n_dims, n_students, n_subjects, student_to_row, subject_to_column = get_marks_matrix(df_join, 0.75)
parameters = {}

pp=pd.read_excel("V.xlsx")
corr_mat = pp.corr()
ax = sb.heatmap(corr_mat, linewidth=0.5,cmap="YlGnBu")
plt.show()

def initialize_parameters(lambda_U, lambda_V):
    U = np.zeros((n_dims, n_students), dtype=np.float64)
    V = np.random.normal(0.0, 1.0 / lambda_V, (n_dims, n_subjects))
    parameters['U'] = U
    parameters['V'] = V
    parameters['lambda_U'] = lambda_U
    parameters['lambda_V'] = lambda_V



def update_parameters():
    U = parameters['U']
    V = parameters['V']
    lambda_U = parameters['lambda_U']
    lambda_V = parameters['lambda_V']

    for i in range(n_students):
        V_j = V[:, R[i, :] > 0]
        U[:, i] = np.dot(np.linalg.inv(np.dot(V_j, V_j.T) + lambda_U * np.identity(n_dims)), np.dot(R[i, R[i, :] > 0], V_j.T))

    for j in range(n_subjects):
        U_i = U[:, R[:, j] > 0]
        V[:, j] = np.dot(np.linalg.inv(np.dot(U_i, U_i.T) + lambda_V * np.identity(n_dims)), np.dot(R[R[:, j] > 0, j], U_i.T))

    parameters['U'] = U
    parameters['V'] = V

def log_a_posteriori():
    lambda_U = parameters['lambda_U']
    lambda_V = parameters['lambda_V']
    U = parameters['U']
    V = parameters['V']

    UV = np.dot(U.T, V)
    R_UV = (R[R > 0] - UV[R > 0])

    return -0.5 * (np.sum(np.dot(R_UV, R_UV.T)) + lambda_U * np.sum(np.dot(U, U.T)) + lambda_V * np.sum(np.dot(V, V.T)))

def predict(Student_ID, Paper_ID):
    U = parameters['U']
    V = parameters['V']

    r_ij = U[:,student_to_row[Student_ID]].T.reshape(1, -1) @ V[:, subject_to_column[Paper_ID]].reshape(-1, 1)

    max_mark = parameters['max_mark']
    min_mark = parameters['min_mark']

    return 0 if max_mark == min_mark else ((r_ij[0][0] - min_mark) / (max_mark - min_mark)) * 100

def evaluate(dataset):
    ground_truths = []
    predictions = []

    for index, row in dataset.iterrows():
        ground_truths.append(row.loc['Marks'])
        predictions.append(predict(row.loc['Student_ID'], row.loc['Paper_ID']))

    return mean_squared_error(ground_truths, predictions, squared=False)

def update_max_min_marks():
    U = parameters['U']
    V = parameters['V']

    R = U.T @ V
    min_mark = np.min(R)
    max_mark = np.max(R)

    parameters['min_mark'] = min_mark
    parameters['max_mark'] = max_mark

def train(n_epochs):
    initialize_parameters(0.5, 0.5)
    log_aps = []
    rmse_train = []
    rmse_test = []

    update_max_min_marks()
    rmse_train.append(evaluate(train_set))
    rmse_test.append(evaluate(test_set))

    for k in range(n_epochs):
        update_parameters()
        log_ap = log_a_posteriori()
        log_aps.append(log_ap)

        if (k + 1) % 10 == 0:
            update_max_min_marks()

            rmse_train.append(evaluate(train_set))
            rmse_test.append(evaluate(test_set))
            print('Log p a-posteriori at iteration', k + 1, ':', log_ap)

    update_max_min_marks()

    return log_aps, rmse_train, rmse_test

log_ps, rmse_train, rmse_test = train(150)

print('RMSE of training set:', evaluate(train_set))
print('RMSE of testing set:', evaluate(test_set))

sb.heatmap(R,cmap="YlGnBu")
plt.show()
