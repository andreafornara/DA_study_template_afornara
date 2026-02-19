# %% [markdown]
# # Interactive Collider Configuration and Tracking
# This notebook provides an interactive version of `2_configure_and_track.py`.
# Each cell can be run independently, allowing you to analyze intermediate results
# and modify parameters before continuing.

# %% Imports
import contextlib
import json
import logging
import os
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker

import xmask as xm
import xmask.lhc as xlhc
import xobjects as xo
import xtrack as xt
from misc import (
    compute_PU,
    generate_orbit_correction_setup,
    get_worst_bunch,
    load_and_check_filling_scheme,
    luminosity_leveling_ip1_5,
    return_fingerprint,
)

# Initialize yaml reader
ryaml = ruamel.yaml.YAML()

# %% Configuration path
# Set the path to your configuration file
config_path = "config.yaml"

# %% Read configuration files
def read_configuration(config_path="config.yaml"):
    """Read configuration for simulations from current and parent generation."""
    with open(config_path, "r") as fid:
        config_gen_2 = ryaml.load(fid)

    try:
        with open("../" + config_path, "r") as fid:
            config_gen_1 = ryaml.load(fid)
    except Exception:
        with open("../1_build_distr_and_collider/" + config_path, "r") as fid:
            config_gen_1 = ryaml.load(fid)

    return config_gen_1, config_gen_2

config_gen_1, config_gen_2 = read_configuration(config_path)

# Inspect the configurations
print("=== Config Gen 1 (MAD config) ===")
print(f"Keys: {list(config_gen_1.keys())}")
print("\n=== Config Gen 2 (Simulation config) ===")
print(f"Keys: {list(config_gen_2.keys())}")

# %% Get context (CPU/GPU)
def get_context(configuration):
    """Get the xobjects context for computation."""
    device_number = configuration.get("device_number", None)

    if configuration["context"] == "cupy":
        return xo.ContextCupy(device=device_number)
    elif configuration["context"] == "opencl":
        return xo.ContextPyopencl()
    elif configuration["context"] == "cpu":
        return xo.ContextCpu()
    else:
        logging.warning("context not recognized, using cpu")
        return xo.ContextCpu()

context = get_context(config_gen_2)
print(f"Using context: {type(context).__name__}")

# %% Generate orbit correction configuration files
def generate_configuration_correction_files(output_folder="correction"):
    """Generate configuration files for orbit correction."""
    correction_setup = generate_orbit_correction_setup()
    os.makedirs(output_folder, exist_ok=True)
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"{output_folder}/corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)

generate_configuration_correction_files()
print("Orbit correction configuration files generated in 'correction/' folder")

# %% Extract simulation and collider configurations
config_sim = config_gen_2["config_simulation"]
config_collider = config_gen_2["config_collider"]

print("=== Simulation Config ===")
for k, v in config_sim.items():
    print(f"  {k}: {v}")

# %% Load the collider
print("Loading collider...")
if config_sim["collider_file"].endswith(".zip"):
    with ZipFile(config_sim["collider_file"], "r") as zip_ref:
        zip_ref.extractall()
    collider = xt.Environment.from_json(
        config_sim["collider_file"].split("/")[-1].replace(".zip", "")
    )
else:
    collider = xt.Environment.from_json(config_sim["collider_file"])

print(f"Collider loaded. Lines: {list(collider.lines.keys())}")

# %% Install beam-beam interactions
def install_beam_beam(collider, config_collider):
    """Install beam-beam lenses (inactive and not configured)."""
    config_bb = config_collider["config_beambeam"]

    collider.install_beambeam_interactions(
        clockwise_line="lhcb1",
        anticlockwise_line="lhcb2",
        ip_names=["ip1", "ip2", "ip5", "ip8"],
        delay_at_ips_slots=[0, 891, 0, 2670],
        num_long_range_encounters_per_side=config_bb["num_long_range_encounters_per_side"],
        num_slices_head_on=config_bb["num_slices_head_on"],
        harmonic_number=35640,
        bunch_spacing_buckets=config_bb["bunch_spacing_buckets"],
        sigmaz=config_bb["sigma_z"],
    )

    return collider, config_bb

collider, config_bb = install_beam_beam(collider, config_collider)
print("Beam-beam interactions installed")
print(f"Number of particles per bunch: {config_bb['num_particles_per_bunch']:.2e}")
print(f"Sigma_z: {config_bb['sigma_z']} m")

# %% Build trackers
collider.build_trackers()
print("Trackers built")

# %% Set knobs
def set_knobs(config_collider, collider):
    """Set all knobs (crossing angles, dispersion correction, RF, crab cavities, etc.)."""
    conf_knobs_and_tuning = config_collider["config_knobs_and_tuning"]

    for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
        collider.vars[kk] = vv

    return collider, conf_knobs_and_tuning

collider, conf_knobs_and_tuning = set_knobs(config_collider, collider)
print("Knobs set. Current knob values:")
for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
    print(f"  {kk}: {vv}")

# %% Inspect tune and chromaticity BEFORE matching
print("=== Tune and chromaticity BEFORE matching ===")
for line_name in ["lhcb1", "lhcb2"]:
    print(f'Looking at line {line_name}')
    for ii in collider[line_name].element_names:
        if ii.startswith('acsca'):
            print(ii)
            print(collider[line_name][ii])
            collider[line_name][ii].lag = 180.000000001
    tw = collider[line_name].twiss()
    print(f"\n{line_name}:")
    print(f"  Qx = {tw.qx:.6f}, Qy = {tw.qy:.6f}")
    print(f"  dQx = {tw.dqx:.2f}, dQy = {tw.dqy:.2f}")
    print(f"  c_minus = {tw.c_minus:.6f}")

# %% Match tune and chromaticity
def match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True):
    """Match tune and chromaticity for both beams."""
    for line_name in ["lhcb1", "lhcb2"]:
        knob_names = conf_knobs_and_tuning["knob_names"][line_name]

        targets = {
            "qx": conf_knobs_and_tuning["qx"][line_name],
            "qy": conf_knobs_and_tuning["qy"][line_name],
            "dqx": conf_knobs_and_tuning["dqx"][line_name],
            "dqy": conf_knobs_and_tuning["dqy"][line_name],
        }

        xm.machine_tuning(
            line=collider[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=match_linear_coupling_to_zero,
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
            line_co_ref=collider[line_name + "_co_ref"],
            co_corr_config=conf_knobs_and_tuning["closed_orbit_correction"][line_name],
        )

    return collider

print("Matching tune and chromaticity (with linear coupling correction to zero)...")
collider = match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True)
print("Tune and chromaticity matched")

# %% Inspect tune and chromaticity AFTER matching
print("=== Tune and chromaticity AFTER matching ===")
for line_name in ["lhcb1", "lhcb2"]:
    tw = collider[line_name].twiss()
    target_qx = conf_knobs_and_tuning["qx"][line_name]
    target_qy = conf_knobs_and_tuning["qy"][line_name]
    target_dqx = conf_knobs_and_tuning["dqx"][line_name]
    target_dqy = conf_knobs_and_tuning["dqy"][line_name]
    print(f"\n{line_name}:")
    print(f"  Qx = {tw.qx:.6f} (target: {target_qx})")
    print(f"  Qy = {tw.qy:.6f} (target: {target_qy})")
    print(f"  dQx = {tw.dqx:.2f} (target: {target_dqx})")
    print(f"  dQy = {tw.dqy:.2f} (target: {target_dqy})")
    print(f"  c_minus = {tw.c_minus:.6f}")


# %% Set filling scheme and bunch tracked
def set_filling_and_bunch_tracked(config_bb, ask_worst_bunch=False):
    """Set the filling scheme and determine which bunch to track."""
    filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]
    filling_scheme_path = load_and_check_filling_scheme(filling_scheme_path)
    config_bb["mask_with_filling_pattern"]["pattern_fname"] = filling_scheme_path

    n_LR = config_bb["num_long_range_encounters_per_side"]["ip1"]

    if config_bb["mask_with_filling_pattern"]["i_bunch_b1"] is None:
        worst_bunch_b1 = get_worst_bunch(
            filling_scheme_path, numberOfLRToConsider=n_LR, beam="beam_1"
        )
        if ask_worst_bunch:
            while config_bb["mask_with_filling_pattern"]["i_bunch_b1"] is None:
                bool_inp = input(
                    f"Use bunch {worst_bunch_b1} (worst bunch for beam 1)? (y/n): "
                )
                if bool_inp == "y":
                    config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = worst_bunch_b1
                elif bool_inp == "n":
                    config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = int(
                        input("Enter the bunch number for beam 1: ")
                    )
        else:
            config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = worst_bunch_b1

    if config_bb["mask_with_filling_pattern"]["i_bunch_b2"] is None:
        worst_bunch_b2 = get_worst_bunch(
            filling_scheme_path, numberOfLRToConsider=n_LR, beam="beam_2"
        )
        config_bb["mask_with_filling_pattern"]["i_bunch_b2"] = worst_bunch_b2

    return config_bb

config_bb = set_filling_and_bunch_tracked(config_bb, ask_worst_bunch=False)
print(f"Filling scheme: {config_bb['mask_with_filling_pattern']['pattern_fname']}")
print(f"Bunch B1: {config_bb['mask_with_filling_pattern']['i_bunch_b1']}")
print(f"Bunch B2: {config_bb['mask_with_filling_pattern']['i_bunch_b2']}")

# %% Compute number of collisions at each IP
def compute_collision_from_scheme(config_bb):
    """Compute the number of collisions in each IP from the filling scheme."""
    filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

    with open(filling_scheme_path, "r") as fid:
        filling_scheme = json.load(fid)

    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    assert len(array_b1) == len(array_b2) == 3564
    n_collisions_ip1_and_5 = array_b1 @ array_b2
    n_collisions_ip2 = np.roll(array_b1, 891) @ array_b2
    n_collisions_ip8 = np.roll(array_b1, 2670) @ array_b2

    return n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8

n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8 = compute_collision_from_scheme(config_bb)
print(f"Number of collisions:")
print(f"  IP1 & IP5: {n_collisions_ip1_and_5}")
print(f"  IP2: {n_collisions_ip2}")
print(f"  IP8: {n_collisions_ip8}")

# %% Check crab cavity status
crab = False
if "on_crab1" in config_collider["config_knobs_and_tuning"]["knob_settings"]:
    crab_val = float(config_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab1"])
    if abs(crab_val) > 0:
        crab = True
print(f"Crab cavities active: {crab}")

# %% Luminosity leveling (optional)
def do_levelling(
    config_collider,
    config_bb,
    n_collisions_ip2,
    n_collisions_ip8,
    collider,
    n_collisions_ip1_and_5,
    crab,
):
    """Perform luminosity leveling in IP1/5, IP2, and IP8."""
    config_lumi_leveling = config_collider["config_lumi_leveling"]
    config_lumi_leveling["ip2"]["num_colliding_bunches"] = int(n_collisions_ip2)
    config_lumi_leveling["ip8"]["num_colliding_bunches"] = int(n_collisions_ip8)

    initial_I = config_bb["num_particles_per_bunch"]

    # Level luminosity in IP 1/5 by changing intensity
    if (
        "config_lumi_leveling_ip1_5" in config_collider
        and not config_collider["config_lumi_leveling_ip1_5"]["skip_leveling"]
    ):
        print("Leveling luminosity in IP 1/5 varying the intensity")
        config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"] = int(
            n_collisions_ip1_and_5
        )

        try:
            bunch_intensity = luminosity_leveling_ip1_5(
                collider, config_collider, config_bb, crab=crab
            )
        except ValueError:
            print("Problem during luminosity leveling in IP1/5... Ignoring it.")
            bunch_intensity = config_bb["num_particles_per_bunch"]

        config_bb["num_particles_per_bunch"] = float(bunch_intensity)

    # Level in IP2 and IP8
    xlhc.luminosity_leveling(
        collider, config_lumi_leveling=config_lumi_leveling, config_beambeam=config_bb
    )

    # Update configuration
    config_bb["num_particles_per_bunch_before_optimization"] = float(initial_I)
    config_collider["config_lumi_leveling"]["ip2"]["final_on_sep2h"] = float(
        collider.vars["on_sep2h"]._value
    )
    config_collider["config_lumi_leveling"]["ip2"]["final_on_sep2v"] = float(
        collider.vars["on_sep2v"]._value
    )
    config_collider["config_lumi_leveling"]["ip8"]["final_on_sep8h"] = float(
        collider.vars["on_sep8h"]._value
    )
    config_collider["config_lumi_leveling"]["ip8"]["final_on_sep8v"] = float(
        collider.vars["on_sep8v"]._value
    )

    return collider, config_collider

if "config_lumi_leveling" in config_collider and not config_collider["skip_leveling"]:
    print("Performing luminosity leveling...")
    collider, config_collider = do_levelling(
        config_collider,
        config_bb,
        n_collisions_ip2,
        n_collisions_ip8,
        collider,
        n_collisions_ip1_and_5,
        crab,
    )
    print("Luminosity leveling completed")
else:
    print("Skipping luminosity leveling (not configured or skip_leveling=True)")

# %% Add linear coupling
def add_linear_coupling(conf_knobs_and_tuning, collider, config_mad):
    """Add linear coupling as the target in the base collider was 0."""
    version_hllhc = config_mad["ver_hllhc_optics"]
    version_run = config_mad["ver_lhc_run"]

    if version_run == 3.0:
        collider.vars["cmrs.b1_sq"] += conf_knobs_and_tuning["delta_cmr"]
        collider.vars["cmrs.b2_sq"] += conf_knobs_and_tuning["delta_cmr"]
    elif version_hllhc in [1.6, 1.5]:
        collider.vars["c_minus_re_b1"] += conf_knobs_and_tuning["delta_cmr"]
        collider.vars["c_minus_re_b2"] += conf_knobs_and_tuning["delta_cmr"]
    else:
        raise ValueError(f"Unknown version of the optics/run: {version_hllhc}, {version_run}.")

    return collider

config_mad = config_gen_1["config_mad"]
collider = add_linear_coupling(conf_knobs_and_tuning, collider, config_mad)
print(f"Linear coupling added: delta_cmr = {conf_knobs_and_tuning['delta_cmr']}")

# %% Rematch tune and chromaticity (without coupling correction)
print("Rematching tune and chromaticity (keeping linear coupling)...")
collider = match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=False)
print("Rematch completed")

# %% Verify tune, chromaticity, and coupling
def assert_tune_chroma_coupling(collider, conf_knobs_and_tuning):
    """Assert that tune, chromaticity and linear coupling are correct."""
    results = {}
    for line_name in ["lhcb1", "lhcb2"]:
        tw = collider[line_name].twiss()
        results[line_name] = {
            "qx": tw.qx,
            "qy": tw.qy,
            "dqx": tw.dqx,
            "dqy": tw.dqy,
            "c_minus": tw.c_minus,
        }

        assert np.isclose(tw.qx, conf_knobs_and_tuning["qx"][line_name], atol=1e-4), \
            f"tune_x incorrect for {line_name}"
        assert np.isclose(tw.qy, conf_knobs_and_tuning["qy"][line_name], atol=1e-4), \
            f"tune_y incorrect for {line_name}"
        assert np.isclose(tw.dqx, conf_knobs_and_tuning["dqx"][line_name], rtol=1e-2), \
            f"chromaticity_x incorrect for {line_name}"
        assert np.isclose(tw.dqy, conf_knobs_and_tuning["dqy"][line_name], rtol=1e-2), \
            f"chromaticity_y incorrect for {line_name}"
        assert np.isclose(tw.c_minus, conf_knobs_and_tuning["delta_cmr"], atol=5e-3), \
            f"linear coupling incorrect for {line_name}"

    return results

print("=== Final verification ===")
results = assert_tune_chroma_coupling(collider, conf_knobs_and_tuning)
for line_name, vals in results.items():
    print(f"\n{line_name}:")
    print(f"  Qx = {vals['qx']:.6f} (target: {conf_knobs_and_tuning['qx'][line_name]})")
    print(f"  Qy = {vals['qy']:.6f} (target: {conf_knobs_and_tuning['qy'][line_name]})")
    print(f"  dQx = {vals['dqx']:.2f} (target: {conf_knobs_and_tuning['dqx'][line_name]})")
    print(f"  dQy = {vals['dqy']:.2f} (target: {conf_knobs_and_tuning['dqy'][line_name]})")
    print(f"  c_minus = {vals['c_minus']:.6f} (target: {conf_knobs_and_tuning['delta_cmr']})")
print("\nAll assertions passed!")

# %% (Optional) Save collider before beam-beam
# Uncomment to save the collider before beam-beam configuration
collider_before_bb = xt.Environment.from_dict(collider.to_dict())
collider_before_bb.to_json("collider_before_bb.json")
# print("Collider before beam-beam saved")

# %% Configure beam-beam
def configure_beam_beam(collider, config_bb):
    """Configure beam-beam interactions."""
    collider.configure_beambeam_interactions(
        num_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    if "mask_with_filling_pattern" in config_bb and (
        "pattern_fname" in config_bb["mask_with_filling_pattern"]
        and config_bb["mask_with_filling_pattern"]["pattern_fname"] is not None
    ):
        fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
        with open(fname, "r") as fid:
            filling = json.load(fid)
        filling_pattern_cw = filling["beam1"]
        filling_pattern_acw = filling["beam2"]

        i_bunch_cw = config_bb["mask_with_filling_pattern"].get("i_bunch_b1", None)
        i_bunch_acw = config_bb["mask_with_filling_pattern"].get("i_bunch_b2", None)

        collider.apply_filling_pattern(
            filling_pattern_cw=filling_pattern_cw,
            filling_pattern_acw=filling_pattern_acw,
            i_bunch_cw=i_bunch_cw,
            i_bunch_acw=i_bunch_acw,
        )
    return collider

if not config_bb["skip_beambeam"]:
    print("Configuring beam-beam...")
    collider = configure_beam_beam(collider, config_bb)
    print("Beam-beam configured")
else:
    print("Skipping beam-beam configuration (skip_beambeam=True)")

# %% Compute and record final luminosity
def record_final_luminosity(collider, config_bb, l_n_collisions, crab):
    """Compute and record the final luminosity at all IPs."""
    l_ip = ["ip1", "ip2", "ip5", "ip8"]

    def twiss_and_compute_lumi(collider, config_bb, l_n_collisions, crab):
        twiss_b1 = collider["lhcb1"].twiss()
        twiss_b2 = collider["lhcb2"].twiss()
        l_lumi = []
        l_PU = []
        for n_col, ip in zip(l_n_collisions, l_ip):
            try:
                L = xt.lumi.luminosity_from_twiss(
                    n_colliding_bunches=n_col,
                    num_particles_per_bunch=config_bb["num_particles_per_bunch"],
                    ip_name=ip,
                    nemitt_x=config_bb["nemitt_x"],
                    nemitt_y=config_bb["nemitt_y"],
                    sigma_z=config_bb["sigma_z"],
                    twiss_b1=twiss_b1,
                    twiss_b2=twiss_b2,
                    crab=crab,
                )
                PU = compute_PU(L, n_col, twiss_b1["T_rev0"])
            except Exception:
                print(f"Problem computing luminosity in {ip}... Ignoring it.")
                L = 0
                PU = 0
            l_lumi.append(L)
            l_PU.append(PU)
        return l_lumi, l_PU

    # Without beam-beam
    collider.vars["beambeam_scale"] = 0
    l_lumi_no_bb, l_PU_no_bb = twiss_and_compute_lumi(collider, config_bb, l_n_collisions, crab)

    for ip, L, PU in zip(l_ip, l_lumi_no_bb, l_PU_no_bb):
        config_bb[f"luminosity_{ip}_without_beam_beam"] = float(L)
        config_bb[f"Pile-up_{ip}_without_beam_beam"] = float(PU)

    # With beam-beam
    collider.vars["beambeam_scale"] = 1
    l_lumi_bb, l_PU_bb = twiss_and_compute_lumi(collider, config_bb, l_n_collisions, crab)

    for ip, L, PU in zip(l_ip, l_lumi_bb, l_PU_bb):
        config_bb[f"luminosity_{ip}_with_beam_beam"] = float(L)
        config_bb[f"Pile-up_{ip}_with_beam_beam"] = float(PU)

    return config_bb, l_ip, l_lumi_no_bb, l_lumi_bb, l_PU_no_bb, l_PU_bb

l_n_collisions = [n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip1_and_5, n_collisions_ip8]
config_bb, l_ip, l_lumi_no_bb, l_lumi_bb, l_PU_no_bb, l_PU_bb = record_final_luminosity(
    collider, config_bb, l_n_collisions, crab
)

print("=== Luminosity Results ===")
print(f"{'IP':<6} {'L (no BB)':<15} {'L (with BB)':<15} {'PU (no BB)':<12} {'PU (with BB)':<12}")
for ip, L_no, L_bb, PU_no, PU_bb in zip(l_ip, l_lumi_no_bb, l_lumi_bb, l_PU_no_bb, l_PU_bb):
    print(f"{ip:<6} {L_no:<15.3e} {L_bb:<15.3e} {PU_no:<12.2f} {PU_bb:<12.2f}")

# %% Save updated configuration
with open(config_path, "w") as fid:
    ryaml.dump(config_gen_2, fid)
print(f"Configuration saved to {config_path}")

# %% (Optional) Save final collider
# Uncomment to save the final configured collider
# collider.to_json("collider_final.json")
# print("Final collider saved to 'collider_final.json'")

# %% [markdown]
# ---
# # Tracking Section
# The following cells prepare and execute particle tracking.

# %% Prepare particle distribution
def prepare_particle_distribution(collider, context, config_sim, config_bb):
    """Prepare the particle distribution for tracking."""
    beam = config_sim["beam"]

    particle_df = pd.read_parquet(config_sim["particle_file"])

    r_vect = particle_df["normalized amplitude in xy-plane"].values
    theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180

    A1_in_sigma = r_vect * np.cos(theta_vect)
    A2_in_sigma = r_vect * np.sin(theta_vect)

    particles = collider[beam].build_particles(
        x_norm=A1_in_sigma,
        y_norm=A2_in_sigma,
        delta=config_sim["delta_max"],
        scale_with_transverse_norm_emitt=(config_bb["nemitt_x"], config_bb["nemitt_y"]),
        _context=context,
    )

    particle_id = particle_df.particle_id.values
    return particles, particle_id, r_vect, theta_vect

particles, particle_id, l_amplitude, l_angle = prepare_particle_distribution(
    collider, context, config_sim, config_bb
)

print(f"Particles prepared: {len(particle_id)} particles")
print(f"Amplitude range: {l_amplitude.min():.2f} - {l_amplitude.max():.2f} sigma")
print(f"Angle range: {l_angle.min()*180/np.pi:.1f} - {l_angle.max()*180/np.pi:.1f} deg")

# %% Compute collider fingerprint (before tracking optimization)
fingerprint = return_fingerprint(config_sim["beam"], collider)
hash_fingerprint = hash(fingerprint)
print(f"Collider fingerprint hash: {hash_fingerprint}")

# %% (Optional) Reset tracker for GPU
# Uncomment if you need to switch to GPU context
# if config_gen_2["context"] in ["cupy", "opencl"]:
#     collider.discard_trackers()
#     collider.build_trackers(_context=context)
#     print("Trackers rebuilt for GPU context")

# %% Track particles
def track(collider, particles, config_sim, save_input_particles=False):
    """Perform the particle tracking."""
    beam = config_sim["beam"]
    line = collider[beam]

    # Optimize line for tracking
    line.optimize_for_tracking()

    if save_input_particles:
        pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")

    num_turns = config_sim["n_turns"]
    print(f"Tracking {particles._capacity} particles for {num_turns} turns...")

    a = time.time()
    line.track(particles, turn_by_turn_monitor=False, num_turns=num_turns)
    b = time.time()

    print(f"Elapsed time: {b-a:.2f} s")
    print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6:.2f} us")

    return particles

# Uncomment the following line to run tracking
# particles = track(collider, particles, config_sim, save_input_particles=True)

# %% [markdown]
# **Note:** Tracking is commented out by default. Uncomment the cell above to run tracking.

# %% Process and save tracking results
def process_and_save_results(particles, particle_id, l_amplitude, l_angle,
                             fingerprint, hash_fingerprint, config_gen_1, config_gen_2):
    """Process tracking results and save to parquet file."""
    particles_dict = particles.to_dict()
    particles_df = pd.DataFrame(particles_dict)

    # Sort by parent_particle_id
    particles_df = particles_df.sort_values("parent_particle_id")

    # Assign the original particle IDs
    particles_df["particle_id"] = particle_id

    # Add amplitude and angle
    particles_df["normalized amplitude in xy-plane"] = l_amplitude
    particles_df["angle in xy-plane [deg]"] = l_angle * 180 / np.pi

    # Add metadata
    particles_df.attrs["hash"] = hash_fingerprint
    particles_df.attrs["fingerprint"] = fingerprint
    particles_df.attrs["configuration_gen_1"] = config_gen_1
    particles_df.attrs["configuration_gen_2"] = config_gen_2
    particles_df.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save
    particles_df.to_parquet("output_particles.parquet")
    print("Results saved to 'output_particles.parquet'")

    return particles_df

# Uncomment after tracking is complete
# particles_df = process_and_save_results(
#     particles, particle_id, l_amplitude, l_angle,
#     fingerprint, hash_fingerprint, config_gen_1, config_gen_2
# )

# %% Cleanup
# Remove correction folder and temporary files
# with contextlib.suppress(Exception):
#     os.system("rm -rf correction")
#     os.system("rm -f *.cc")
# print("Cleanup complete")

# %% [markdown]
# ---
# # Analysis Helpers
# Below are some useful cells for analyzing intermediate results.

# %% Twiss analysis helper
def analyze_twiss(collider, line_name="lhcb1"):
    """Perform twiss analysis and display key parameters."""
    tw = collider[line_name].twiss()

    print(f"=== Twiss analysis for {line_name} ===")
    print(f"Qx = {tw.qx:.6f}, Qy = {tw.qy:.6f}")
    print(f"dQx = {tw.dqx:.2f}, dQy = {tw.dqy:.2f}")
    print(f"c_minus = {tw.c_minus:.6f}")
    print(f"T_rev = {tw['T_rev0']*1e6:.3f} us")

    # Beta functions at IPs
    for ip in ["ip1", "ip2", "ip5", "ip8"]:
        try:
            idx = tw.name == ip
            if any(idx):
                print(f"\n{ip}:")
                print(f"  betx = {tw.betx[idx][0]:.4f} m, bety = {tw.bety[idx][0]:.4f} m")
                print(f"  x = {tw.x[idx][0]*1e6:.2f} um, y = {tw.y[idx][0]*1e6:.2f} um")
        except Exception:
            pass

    return tw

# Example: analyze_twiss(collider, "lhcb1")

# %% Knob modification helper
def modify_knob(collider, knob_name, new_value):
    """Modify a knob value and display the change."""
    old_value = collider.vars[knob_name]._value
    collider.vars[knob_name] = new_value
    print(f"{knob_name}: {old_value} -> {new_value}")
    return collider

# Example: collider = modify_knob(collider, "on_x1", 150)