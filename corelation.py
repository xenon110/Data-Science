import pandas as pd
import matplotlib.pyplot as plt
location = r'C:\Users\mayan\Downloads\03+-+corr.csv'
dataset = pd.read_csv(location)
dataset['t0'] = pd.to_numeric(dataset['t0'], downcast='float')
plt.acorr(dataset['t0'],maxlags=12) # How the data is related to previous 12 data
plt.title('Autocorrelation Plot')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.show()

t1 = dataset['t0'].shift(+1).to_frame()  # Shift is also known as the sliding windo approch
t2 = dataset['t0'].shift(+3).to_frame()
print(t1)
print(t2)