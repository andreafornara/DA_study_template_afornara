# %%
import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load collider with Multiline structure
collider = xt.Multiline.from_json('collider_final.json')

# %%
# Load collider with Environment structure
env = xt.Environment.from_json('collider_final.json')

# %%
# Build trackers for both structures
collider.build_trackers()
env.build_trackers()

# %%
# Compute twiss for both structures
twiss_b1_coll = collider['lhcb1'].twiss()
twiss_b2_coll = collider['lhcb2'].twiss()

twiss_b1_env = env['lhcb1'].twiss()
twiss_b2_env = env['lhcb2'].twiss()

# %%
# Print betx, bety, px, py at IPs and tunes
print("\nBeam 1:")
for ip in ['ip1', 'ip5', 'ip2', 'ip8']:
    qx = twiss_b1_env['qx']
    qy = twiss_b1_env['qy']
    betx = twiss_b1_env['betx', ip]
    bety = twiss_b1_env['bety', ip]
    px = twiss_b1_env['px', ip]
    py = twiss_b1_env['py', ip]
    print(f"{ip}: betx={betx:.3f} m, bety={bety:.3f} m, px={px:.6f}, py={py:.6f}, qx={qx:.6f}, qy={qy:.6f}")

print("\nBeam 2:")
for ip in ['ip1', 'ip5', 'ip2', 'ip8']:
    qx = twiss_b2_env['qx']
    qy = twiss_b2_env['qy']
    betx = twiss_b2_env['betx', ip]
    bety = twiss_b2_env['bety', ip]
    px = twiss_b2_env['px', ip]
    py = twiss_b2_env['py', ip]
    print(f"{ip}: betx={betx:.3f} m, bety={bety:.3f} m, px={px:.6f}, py={py:.6f}, qx={qx:.6f}, qy={qy:.6f}")

# %%
# Plot orbit (x, y) for both beams
s_ip1 = twiss_b1_env['s', 'ip1']
s_ip2 = twiss_b1_env['s', 'ip2']
s_ip5 = twiss_b1_env['s', 'ip5']
s_ip8 = twiss_b1_env['s', 'ip8']

fs = 18
legend_fs = 14

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
ax[0].plot(twiss_b1_env.s, twiss_b1_env.x, 'b', label='Beam 1')
ax[0].plot(twiss_b2_env.s, twiss_b2_env.x, 'r', label='Beam 2')
ax[0].axvline(s_ip1, color='k', linestyle='--', label='IP1')
ax[0].axvline(s_ip2, color='g', linestyle='--', label='IP2')
ax[0].axvline(s_ip5, color='darkorange', linestyle='--', label='IP5')
ax[0].axvline(s_ip8, color='c', linestyle='--', label='IP8')
ax[0].set_ylabel(r'Orbit x [m]', fontsize=fs)
ax[0].legend(fontsize=legend_fs)
ax[0].grid(True)
ax[0].tick_params(axis='both', labelsize=fs)

ax[1].plot(twiss_b1_env.s, twiss_b1_env.y, 'b', label='Beam 1')
ax[1].plot(twiss_b2_env.s, twiss_b2_env.y, 'r', label='Beam 2')
ax[1].axvline(s_ip1, color='k', linestyle='--', label='IP1')
ax[1].axvline(s_ip2, color='g', linestyle='--', label='IP2')
ax[1].axvline(s_ip5, color='darkorange', linestyle='--', label='IP5')
ax[1].axvline(s_ip8, color='c', linestyle='--', label='IP8')
ax[1].set_ylabel(r'Orbit y [m]', fontsize=fs)
ax[1].set_xlabel('s [m]', fontsize=fs)
ax[1].legend(fontsize=legend_fs)
ax[1].grid(True)
ax[1].tick_params(axis='both', labelsize=fs)
plt.suptitle('Orbit', fontsize=fs)
plt.tight_layout()

# %%
# Plot beta functions (betx, bety) for both beams
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
ax[0].plot(twiss_b1_env.s, twiss_b1_env.betx, 'b', label='Beam 1')
ax[0].plot(twiss_b2_env.s, twiss_b2_env.betx, 'r', label='Beam 2')
ax[0].axvline(s_ip1, color='k', linestyle='--', label='IP1')
ax[0].axvline(s_ip2, color='g', linestyle='--', label='IP2')
ax[0].axvline(s_ip5, color='darkorange', linestyle='--', label='IP5')
ax[0].axvline(s_ip8, color='c', linestyle='--', label='IP8')
ax[0].set_ylabel(r'$\beta_x$ [m]', fontsize=fs)
ax[0].legend(fontsize=legend_fs)
ax[0].grid(True)
ax[0].tick_params(axis='both', labelsize=fs)

ax[1].plot(twiss_b1_env.s, twiss_b1_env.bety, 'b', label='Beam 1')
ax[1].plot(twiss_b2_env.s, twiss_b2_env.bety, 'r', label='Beam 2')
ax[1].axvline(s_ip1, color='k', linestyle='--', label='IP1')
ax[1].axvline(s_ip2, color='g', linestyle='--', label='IP2')
ax[1].axvline(s_ip5, color='darkorange', linestyle='--', label='IP5')
ax[1].axvline(s_ip8, color='c', linestyle='--', label='IP8')
ax[1].set_ylabel(r'$\beta_y$ [m]', fontsize=fs)
ax[1].set_xlabel('s [m]', fontsize=fs)
ax[1].legend(fontsize=legend_fs)
ax[1].grid(True)
ax[1].tick_params(axis='both', labelsize=fs)
plt.suptitle('Beta functions', fontsize=fs)
plt.tight_layout()

# %%
# Check beam-beam lens installation
elements_b1 = env['lhcb1'].element_names
elements_b2 = env['lhcb2'].element_names

hobb_b1 = [elem for elem in elements_b1 if elem.startswith('bb_ho')]
lrbb_b1 = [elem for elem in elements_b1 if elem.startswith('bb_lr')]
hobb_b2 = [elem for elem in elements_b2 if elem.startswith('bb_ho')]
lrbb_b2 = [elem for elem in elements_b2 if elem.startswith('bb_lr')]

for beam_name, hobb, lrbb in [('Beam 1', hobb_b1, lrbb_b1), ('Beam 2', hobb_b2, lrbb_b2)]:
    print(f"\n{beam_name}:")
    for ip in ['1', '2', '5', '8']:
        n_ho_left = sum(1 for elem in hobb if f'.l{ip}' in elem)
        n_ho_center = sum(1 for elem in hobb if f'.c{ip}' in elem)
        n_ho_right = sum(1 for elem in hobb if f'.r{ip}' in elem)
        n_lr_left = sum(1 for elem in lrbb if f'.l{ip}' in elem)
        n_lr_right = sum(1 for elem in lrbb if f'.r{ip}' in elem)
        print(f"  IP{ip}: HO({n_ho_left}L + {n_ho_center}C + {n_ho_right}R), LR({n_lr_left}L + {n_lr_right}R)")

# %%
# Define particles and track for 100 turns
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
delta_max = 27.e-5

sigma_min = 0.1
sigma_max = 8.0
n_particles = 3

x_norm = np.linspace(sigma_min, sigma_max, n_particles)
y_norm = np.linspace(sigma_min, sigma_max, n_particles)
amplitudes = np.sqrt(x_norm**2 + y_norm**2)

particles = env['lhcb1'].build_particles(
    x_norm=x_norm,
    y_norm=y_norm,
    delta=delta_max,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y
)

env['lhcb1'].track(particles, num_turns=2000, turn_by_turn_monitor=True)
xs = env['lhcb1'].record_last_track.x
ys = env['lhcb1'].record_last_track.y
pxs = env['lhcb1'].record_last_track.px
pys = env['lhcb1'].record_last_track.py

# %%
fig, ax = plt.subplots(2, 1, figsize=(15, 10))

for i in range(n_particles):
    ax[0].scatter(xs[i, :], pxs[i, :], c=amplitudes[i]*np.ones(xs.shape[1]),
                  cmap='viridis', vmin=amplitudes.min(), vmax=amplitudes.max(), s=10)
    ax[1].scatter(ys[i, :], pys[i, :], c=amplitudes[i]*np.ones(ys.shape[1]),
                  cmap='viridis', vmin=amplitudes.min(), vmax=amplitudes.max(), s=10)

ax[0].set_ylabel('px', fontsize=fs)
ax[0].set_xlabel('x [m]', fontsize=fs)
ax[0].grid(True)
ax[0].tick_params(axis='both', labelsize=fs)

ax[1].set_ylabel('py', fontsize=fs)
ax[1].set_xlabel('y [m]', fontsize=fs)
ax[1].grid(True)
ax[1].tick_params(axis='both', labelsize=fs)

plt.suptitle('Phase Space', fontsize=fs)
plt.tight_layout()

# %%
