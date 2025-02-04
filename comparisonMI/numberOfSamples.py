from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt

p = np.array([0.99, 0.95, 0.05, 0.01])
df = np.array(np.arange(2, 2**12)).reshape(-1, 1)
table = chi2.isf(p, df)

l = np.divide(table[:,1].reshape(-1, ),df.reshape(-1, )-1)
u = np.divide(table[:,2].reshape(-1, ),df.reshape(-1, )-1)

l_ = np.divide(table[:,0].reshape(-1, ),df.reshape(-1, )-1)
u_ = np.divide(table[:,3].reshape(-1, ),df.reshape(-1, )-1)

l = np.sqrt(l)
u = np.sqrt(u)
l_ = np.sqrt(l_)
u_ = np.sqrt(u_)

fig = plt.figure(figsize=[8,4])
ax = fig.add_subplot(1, 1, 1)
ax.plot(df,u_,label="",color="#44bf70")
ax.plot(df,l_,label="Limits of confidence level 99%",color="#44bf70")
ax.plot(df,u,label="",color="#482475")
ax.plot(df,l,label="Limits of confidence level 95%",color="#482475")

ax.set_xscale('log', base=2)
ax.set_ylim([0,2])
ax.set_xlabel("S [-]")
ax.set_ylabel(r'$\sqrt{\sigma_{true}^2/\sigma^2}$ [-]')
plt.title(r"Confidence level limits of $\sqrt{\sigma_{true}^2/\sigma^2}$")
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f"NoC_confidence.png", dpi=600)

plt.tight_layout()

plt.show()