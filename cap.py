import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import *

#https://github.com/danielmarostica/python-cap

rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})

#Load the data
data = pd.read_csv('data.csv')

#Do some magic
total_1 = len(data[data['Survived'] == 1])
total_records = len(data.index)
percent_of_1 = total_1/total_records
percent_of_total_records = [(i+1)/total_records for i in range(total_records)]
random_select = [(i+1)*percent_of_1 for i in range(total_records)]
random_select_percent_of_1 = [random_select[i]/total_1 for i in range(total_records)]
model_select = [sum(data.iloc[0:i+1,1]) for i in range(total_records)]
model_select_percent_of_1 = [model_select[i]/total_1 for i in range(total_records)]

#Voilà
plt.plot(percent_of_total_records, random_select_percent_of_1, color='red', label='Random')
plt.plot(percent_of_total_records, model_select_percent_of_1, color='green', label='Model')
plt.title('Cumulative Accuracy Profile (CAP)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('cap.jpg', dpi=150)
plt.legend()
#plt.show()
plt.close()
