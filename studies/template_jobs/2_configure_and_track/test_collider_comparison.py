# %%
import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load collider with Multiline structure
collider = xt.Multiline.from_json('collider_final.json')

# %%
# Load collider with Environment structure
env = xt.load('collider_final.json')

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
# Print betx, bety, px, py at IPs
print("\nBeam 1:")
for ip in ['ip1', 'ip5', 'ip2', 'ip8']:
    betx = twiss_b1_env['betx', ip]
    bety = twiss_b1_env['bety', ip]
    px = twiss_b1_env['px', ip]
    py = twiss_b1_env['py', ip]
    print(f"{ip}: betx={betx:.3f} m, bety={bety:.3f} m, px={px:.6f}, py={py:.6f}")

print("\nBeam 2:")
for ip in ['ip1', 'ip5', 'ip2', 'ip8']:
    betx = twiss_b2_env['betx', ip]
    bety = twiss_b2_env['bety', ip]
    px = twiss_b2_env['px', ip]
    py = twiss_b2_env['py', ip]
    print(f"{ip}: betx={betx:.3f} m, bety={bety:.3f} m, px={px:.6f}, py={py:.6f}")

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
# Define a particle and track for 10 turns
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6
delta_max = 27.e-5

particles = env['lhcb1'].build_particles(
    x_norm=1.0,
    y_norm=1.0,
    delta=delta_max,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y
)

env['lhcb1'].track(particles, num_turns=10, turn_by_turn_monitor=True)

print(f"\nParticle tracking completed for 10 turns")
print(f"Final state: {particles.state[0]}")
print(f"Final x: {particles.x[0]:.6e} m")
print(f"Final y: {particles.y[0]:.6e} m")

# %%
