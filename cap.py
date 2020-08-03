import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import *
from sklearn.metrics import auc
import numpy as np

#https://github.com/danielmarostica/python-cap

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})

# load the data
data = pd.read_csv('data.csv')
y = data[data.columns[1]]

# sort
data_sorted_ypred = data.sort_values(by=[data.columns[2]], ascending=False)
data_sorted_y = data.sort_values(by=[data.columns[1]], ascending=False)

# total records
total_records = len(data)
# total amount of positives
total_positive = len(data[data[data.columns[1]] == 1])

# proportion of the total records (x axis)
x = [(i+1)/total_records for i in range(total_records)]

# proportion of positives out of total
proportion_of_positive = total_positive/total_records
# random select
random_select = [(i+1)*proportion_of_positive for i in range(total_records)]
# out of the random select, proportion of positives (y axis)
random_select_proportion_of_positive = [random_select[i]/total_positive for i in range(total_records)]

# model select
model_select = [sum(data_sorted_ypred.iloc[0:i+1,1]) for i in range(total_records)]
# out of the model select, proportion of positives (y axis)
model_select_proportion_of_positive = [model_select[i]/total_positive for i in range(total_records)]

# perfect select
perfect_select = [sum(data_sorted_y.iloc[0:i+1,1]) for i in range(total_records)]
# out of the perfect select, proportion of positives (y axis)
perfect_select_proportion_of_positive = [perfect_select[i]/total_positive for i in range(total_records)]

auc_random = auc(x, random_select_proportion_of_positive)
auc_model = auc(x, model_select_proportion_of_positive)
auc_perfect = auc(x, perfect_select_proportion_of_positive)

acc_ratio = (auc_model-auc_random)/(auc_perfect-auc_random)

#print(auc_perfect)
#x = np.arange(len(data['Survived']))/len(data['Survived'])

#Voilà
plt.plot(x, random_select_proportion_of_positive, color='red', label='Random')
plt.plot(x, model_select_proportion_of_positive, color='green', label='Model')
plt.plot(x, perfect_select_proportion_of_positive, color='blue', label='Perfect')
plt.text(1,0,'Accuracy Ratio: {:.2f}'.format(acc_ratio), ha='right')
plt.title('Cumulative Accuracy Profile (CAP)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(frameon=False)
plt.savefig('cap.jpg', dpi=150)
#plt.show()
plt.close()
