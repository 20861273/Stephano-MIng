import math
import numpy as np
import pandas as pd

arr = [1,1,3,9,5,5,7,8,14,10]
  
numbers_series = pd.Series(arr)
        
avg = numbers_series.rolling(window=3).mean()

# avg = avg.tolist()

# print(avg)

for i in range(len(arr)):
    if math.isnan(avg[i]): avg[i] = arr[i]

avg = avg.tolist()
print(avg)