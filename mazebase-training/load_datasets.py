import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

def inject_bug(X,y, bug_type):
  which_has_bugs = np.zeros((X.shape[0],))
  ind_greater_50k = np.where(y==1)[0].tolist()
  inds = np.random.choice(ind_greater_50k, int(0.25*len(ind_greater_50k))).tolist()
  which_has_bugs[inds] = 1
  if bug_type == 'labelleak':
    X[inds,0] = y[inds]
  elif bug_type == 'corruptfeat':
    X[inds,0] = -1.
  elif bug_type == 'labelerror':
    y[inds] = 0


def load_income_dataset(has_bug, bug_type=None):
	# load the dataset as a numpy array
	dataframe = pd.read_csv('adult-all.csv', header=None, na_values='?')
  
	# drop rows with missing
	dataframe = dataframe.dropna()
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	X.columns = ['Age', 'Workclass', 'FinalWeight', 'Education', 'EducationNumberOfYears', 'Marital-status',
              'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-gain', 'Capital-loss', 'Hours-per-week',
              'Native-country']
	# select categorical and numerical features
	cat_ix = X.select_dtypes(include=['object', 'bool']).columns
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns

	scaler = MinMaxScaler()
	train_X_scaled = scaler.fit_transform(X[num_ix])

	train_X_new = SelectKBest(chi2, k=3).fit_transform(train_X_scaled, y)  
	y = LabelEncoder().fit_transform(y)

	if has_bug:
		train_X_new, which_has_bugs = inject_bug(train_X_new,y, bug_type)

	inds = np.random.choice(train_X_new.shape[0], 10000)  
	X_subset = train_X_new[inds,:]
	y_subset = y[inds]
	which_has_bugs = which_has_bugs[inds]

	return X_subset, y_subset, which_has_bugs

def load_xor_dataset(has_bug,bug_type=None):
  n=2500
  dataset = np.random.rand(n,3)
  labels = np.zeros(n)

  for i in range(n):
    if dataset[i][0] > 0.5 and dataset[i][1] > 0.5:
      labels[i] = 1.
    elif dataset[i][0] < 0.5 and dataset[i][1] < 0.5:
      labels[i] = 1.
    else:
      labels[i] = 0.
  if has_bug:
		train_X_new, which_has_bugs = inject_bug(train_X_new,y, bug_type)

  inds = np.random.choice(dataset.shape[0], 1000)  
	X_subset = dataset[inds,:]
	y_subset = labels[inds]
	which_has_bugs = which_has_bugs[inds]

  return dataset, labels, which_has_bugs

def load_wifi_dataset(has_bug, bug_type=None):
  data = np.loadtxt('wifi_localization.txt', delimiter='\t')
  X = data[:,:-1]
  y = data[:,-1]
  scaler = MinMaxScaler()
  train_X_scaled = scaler.fit_transform(X)
  y = LabelEncoder().fit_transform(y)

  train_X_new = SelectKBest(chi2, k=3).fit_transform(train_X_scaled, y) # just pick these 3

  #add bug here. 
  if has_bug:
		train_X_new, which_has_bugs = inject_bug(train_X_new,y, bug_type)

  inds = np.random.choice(X.shape[0], 1000)  
	X_subset = X[inds,:]
	y_subset = labels[inds]
	which_has_bugs = which_has_bugs[inds]

  return X_subset, y_subset, which_has_bugs