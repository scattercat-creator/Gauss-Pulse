# goal 1: read s2p data into panda dataframe
import skrf as rf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

network = rf.Network('filter_time_series.s2p')

# save all parameters in s2p file
freq = network.f
s11 = network.s[:,0,0] # how much signal at port 1 bounces back
s12 = network.s[:,0,1] # gain / loss 
s21 = network.s[:,1,0] # how much signal leaks backward through device
s22 = network.s[:,1,1] # how much signal bounces back from output

# create panda dataframe

df = pd.DataFrame({
    'frequency': freq,
    's11_real': s11.real,
    's11_imag': s11.imag,
    's12_real': s12.real,
    's12_imag': s12.imag,
    's21_real': s21.real,
    's21_imag': s21.imag,
    's22_real': s22.real,
    's22_imag': s22.imag
})

# plot the gain (s21)

# get rid of points i don't want
df = df.drop(df.index[-75:])

# get the s21 magnitude
s21_magnitude_db = 20 * np.log10(np.sqrt(df['s21_real']**2 + df['s21_imag']**2))

# plot it
plt.figure()
plt.plot(df['frequency']/1e9, s21_magnitude_db)
plt.xlabel('frequency (GHz)')
plt.ylabel('s21 Magnitude (dB)')
plt.show()