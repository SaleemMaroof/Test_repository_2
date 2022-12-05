#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[27]:


# Import main data and get list of SMILES
molecules = pd.read_csv('./fathead_minnow_dataset.csv')  # Load the photoswitch dataset using pandas
smiles_list = list(molecules.SMILES.values)


# In[28]:


len(smiles_list)


# In[29]:


# Initiate list of rdkit molecules
rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]


# In[30]:


# Get Morgan fingerprints, note the parameters!
morgan_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, radius = 3, nBits = 2048) for mol in rdkit_mols]
morgan_fingerprints = np.asarray(morgan_fingerprints)


# In[44]:


# Turn into pandas dataframe and add smiles as a first column
morgan_fingerprints = pd.DataFrame(data = morgan_fingerprints)
morgan_fingerprints.insert(0, 'SMILES', smiles_list)


# In[52]:


morgan_fingerprints


# In[53]:


morgan_fingerprints.duplicated().sum


# In[12]:


morgan_fingerprints.to_csv('morgan_fingerprints.csv')


# In[ ]:





# In[13]:


# Next, rdkit's own descriptors
from rdkit.Chem import Descriptors


# In[14]:


# A list of desriptors
Descriptors.descList


# In[15]:


# Write a dictionary of name:function pairs for all descriptors
all_descriptors = {d[0]: d[1] for d in Descriptors.descList}


# In[16]:


# Initialise a new pandas df
rdkit_descriptors = pd.DataFrame(data = {'SMILES': np.array((smiles_list)) })
rdkit_descriptors


# In[31]:


# Compute each descriptor (outer loop) for each molecule(inside)
for feature in all_descriptors:
    values = []
    for mol in rdkit_mols:
        values += [all_descriptors[feature](mol)]
    rdkit_descriptors[feature] = values

rdkit_descriptors


# In[18]:


rdkit_descriptors.to_csv('rdkit_descriptors.csv')


# In[71]:


rdkit_descriptors.duplicated().sum
sns.countplot(rdkit_descriptors['qed'])


# In[58]:


rdkit_descriptors['SMILES'].unique()


# In[72]:


sns.countplot(rdkit_descriptors['qed'])


# In[64]:


rdkit_descriptors.isnull().sum()


# In[67]:


rdkit_descriptors.dtypes


# In[76]:


rdkit_descriptors[['qed']].boxplot()


# In[77]:


rdkit_descriptors.corr()


# In[33]:


pip install mordred


# In[34]:


# Finally, mordred descriptors
from mordred import Calculator, descriptors, error


# In[35]:


# Initialise a calculator -- mordred works weirdly this way...
calc = Calculator(descriptors)


# In[36]:


# Wow, many descriptors, much wow
len(calc.descriptors)


# In[37]:


mordred_descriptors = calc.pandas(rdkit_mols)


# In[38]:


# It seems that unfortunately some descriptors cannot be computed. To filter this, 
# we find all columns that are of data type "object", since those contain non-numerical values usually.
error_columns = []
for i, e in enumerate(mordred_descriptors.dtypes):
    if e == 'object':
        error_columns += [i]
error_columns


# In[39]:


# use .drop to remove the affected columns 
mordred_descriptors = mordred_descriptors.drop(mordred_descriptors.columns[error_columns], axis = 1)


# In[40]:


# and remove columns containing NA data, but I don't think this actually does anything...
mordred_descriptors = mordred_descriptors.dropna(axis = 1)


# In[41]:


# again, insert first SMILES column
mordred_descriptors.insert(0, 'SMILE', smiles_list)
mordred_descriptors


# In[42]:


mordred_descriptors.to_csv('mordred_descriptors.csv')


# In[ ]:





# In[43]:


# finally, generate images of molecules
from rdkit.Chem import Draw
for i,mol in enumerate(rdkit_mols):
    Draw.MolToFile(mol, filename = 'molecular_images/' + str(i) + '.png')


# In[30]:


minnow = pd.read_csv("fathead_minnow_dataset.csv")
mordred = pd.read_csv("mordred_descriptors.csv")
morgan = pd.read_csv("morgan_fingerprints.csv")
rdkit_des = pd.read_csv("rdkit_descriptors.csv")
name = minnow['SMILES']
name2 = rdkit_des['SMILES']
toxicity  = minnow["LC50_(mg/L)"] / rdkit_des['MolWt']
y = (toxicity<0.5)
data = morgan.iloc[:, 2:].dropna()
X = data.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


# In[85]:


# Import MLPClassifer 
from sklearn.neural_network import MLPClassifier

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)

# Fit data onto the model
clf.fit(X_train,y_train)


# In[82]:


# Make prediction on test dataset
ypred=clf.predict(X_test)

# Import accuracy score 
from sklearn.metrics import accuracy_score

# Calcuate accuracy
accuracy_score(y_test,ypred)


# In[15]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[13]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[26]:


classifier = Sequential()


# In[84]:


classifier.add(Dense(11, activation = 'relu', input_dim = 11))
classifier.add(Dense(11, input_dim = 11, activation = 'relu'))
classifier.add(Dense(11, input_dim = 11, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X, y, epochs=1, batch_size=10)


# In[72]:


classifier.summary()


# In[24]:


X.shape


# In[17]:


from sklearn.linear_model import Perceptron
per_clf = Perceptron()
per_clf.fit(X, y)
y_pred = per_clf.predict([[2048]])


# In[30]:


y.shape


# In[16]:


from sklearn.linear_model import LinearRegression, LogisticRegression

# Fit a linear regression and print intercept and coefficient
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)



# In[18]:


X_b = np.c_[np.ones((554, 2048)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


# In[19]:


X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
y_predict


# In[24]:


# Initialise data for plotting the regression line
X_new = np.linspace(min(X), max(X)).reshape(-1,1)
y_predict = lin_reg.predict(X_new)

# Plot the regression
plt.figure(figsize = (8, 6))
plt.plot(X, y, 'r-', linewidth = 2, label = 'Predictions')
plt.plot(X, y, 'b.')
plt.xlabel('atomic number')
plt.ylabel('atomic weight')
plt.legend(loc = 'upper left', fontsize = 14)
plt.show()

