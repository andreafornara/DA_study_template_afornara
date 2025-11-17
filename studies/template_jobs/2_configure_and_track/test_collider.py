# %%
import xtrack as xt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xpart as xp
#import configure_and_track as configure_and_track

# %%
collider = xt.Multiline.from_json('collider_final.json')

# %%
collider.build_trackers()

# %%
# collider.vars['beambeam_scale'] = 0 
twiss_b1 = collider['lhcb1'].twiss()
twiss_b2 = collider['lhcb2'].twiss().reverse()

s_ip1 = twiss_b1['s', 'ip1']
s_ip2 = twiss_b1['s', 'ip2']
s_ip5 = twiss_b1['s', 'ip5']
s_ip8 = twiss_b1['s', 'ip8']
# %%
fs = 18
legend_fs = 14
suptitle_fs = 18
fig, ax = plt.subplots(2,1, sharex=True, figsize=(15,10))
ax[0].plot(twiss_b1.s, twiss_b1.x, 'b', label='Beam 1')
ax[0].plot(twiss_b2.s, twiss_b2.x, 'r', label='Beam 2')
ax[0].axvline(s_ip1, color='k', linestyle='--', label='IP1')
ax[0].axvline(s_ip2, color='g', linestyle='--', label='IP2')
ax[0].axvline(s_ip5, color='darkorange', linestyle='--', label='IP5')
ax[0].axvline(s_ip8, color='c', linestyle='--', label='IP8')    
ax[0].set_ylabel(r'Orbit x [m]', fontsize=fs)
ax[0].legend(fontsize=legend_fs)
ax[0].grid(True)
ax[0].tick_params(axis='both', labelsize=fs)
ax[1].plot(twiss_b1.s, twiss_b1.y, 'b', label='Beam 1')
ax[1].plot(twiss_b2.s, twiss_b2.y, 'r', label='Beam 2')
ax[1].axvline(s_ip1, color='k', linestyle='--', label='IP1')
ax[1].axvline(s_ip2, color='g', linestyle='--', label='IP2')
ax[1].axvline(s_ip5, color='darkorange', linestyle='--', label='IP5')
ax[1].axvline(s_ip8, color='c', linestyle='--', label='IP8')
ax[1].set_ylabel(r'Orbit y [m]', fontsize=fs)
ax[1].set_xlabel('s [m]', fontsize=fs)
ax[1].legend(fontsize=legend_fs)
ax[1].grid(True)
ax[1].tick_params(axis='both', labelsize=fs)
plt.suptitle('Orbits with Beam-Beam ON', fontsize=suptitle_fs)
plt.tight_layout()
# %%
fs = 18
legend_fs = 14
suptitle_fs = 18

fig, ax = plt.subplots(2,1, sharex=True, figsize=(15,10))
ax[0].plot(twiss_b1.s, twiss_b1.betx, 'b', label='Beam 1')
ax[0].plot(twiss_b2.s, twiss_b2.betx, 'r', label='Beam 2')
ax[0].axvline(s_ip1, color='k', linestyle='--', label='IP1')
ax[0].axvline(s_ip2, color='g', linestyle='--', label='IP2')
ax[0].axvline(s_ip5, color='darkorange', linestyle='--', label='IP5')
ax[0].axvline(s_ip8, color='c', linestyle='--', label='IP8')
ax[0].set_ylabel(r'$\beta_x$ [m]', fontsize=fs)
ax[0].legend(fontsize=legend_fs)
ax[0].grid(True)
ax[0].tick_params(axis='both', labelsize=fs)

ax[1].plot(twiss_b1.s, twiss_b1.bety, 'b', label='Beam 1')
ax[1].plot(twiss_b2.s, twiss_b2.bety, 'r', label='Beam 2')
ax[1].axvline(s_ip1, color='k', linestyle='--', label='IP1')
ax[1].axvline(s_ip2, color='g', linestyle='--', label='IP2')
ax[1].axvline(s_ip5, color='darkorange', linestyle='--', label='IP5')
ax[1].axvline(s_ip8, color='c', linestyle='--', label='IP8')
ax[1].set_ylabel(r'$\beta_y$ [m]', fontsize=fs)
ax[1].set_xlabel('s [m]', fontsize=fs)
ax[1].legend(fontsize=legend_fs)
ax[1].grid(True)
ax[1].tick_params(axis='both', labelsize=fs)

plt.suptitle('Beta functions with Beam-Beam ON', fontsize=suptitle_fs)
plt.tight_layout()
# %%