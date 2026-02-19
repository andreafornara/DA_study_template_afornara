"""This script is used to configure the collider and track the particles. Functions in this script
are called sequentially, in the order in which they are defined. Modularity has been favored over
simple scripting for reproducibility, to allow rebuilding the collider from a different program
(e.g. dahsboard)."""

import contextlib

# ==================================================================================================
# --- Imports
# ==================================================================================================
# Import standard library modules
import json
import logging
import os
import time
from zipfile import ZipFile

# Import third-party modules
import numpy as np
import pandas as pd
import ruamel.yaml
import tree_maker

# Import user-defined modules
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


# ==================================================================================================
# --- Function for tree_maker tagging
# ==================================================================================================
def tree_maker_tagging(config, tag="started"):
    # Start tree_maker logging if log_file is present in config
    if tree_maker is not None and "log_file" in config:
        tree_maker.tag_json.tag_it(config["log_file"], tag)
    else:
        logging.warning("tree_maker loging not available")


# ==================================================================================================
# --- Function to get context
# ==================================================================================================
def get_context(configuration):
    # Get device number
    if "device_number" not in configuration:
        device_number = None
    else:
        device_number = configuration["device_number"]

    if configuration["context"] == "cupy":
        return xo.ContextCupy(device=device_number)
    elif configuration["context"] == "opencl":
        return xo.ContextPyopencl()
    elif configuration["context"] == "cpu":
        return xo.ContextCpu()
    else:
        logging.warning("context not recognized, using cpu")
        return xo.ContextCpu()


# ==================================================================================================
# --- Functions to read configuration files and generate configuration files for orbit correction
# ==================================================================================================
def read_configuration(config_path="config.yaml"):
    # Read configuration for simulations
    with open(config_path, "r") as fid:
        config_gen_2 = ryaml.load(fid)

    # Also read configuration from previous generation
    try:
        with open("../" + config_path, "r") as fid:
            config_gen_1 = ryaml.load(fid)
    except Exception:
        with open("../1_build_distr_and_collider/" + config_path, "r") as fid:
            config_gen_1 = ryaml.load(fid)

    return config_gen_1, config_gen_2


def generate_configuration_correction_files(output_folder="correction"):
    # Generate configuration files for orbit correction
    correction_setup = generate_orbit_correction_setup()
    os.makedirs(output_folder, exist_ok=True)
    for nn in ["lhcb1", "lhcb2"]:
        with open(f"{output_folder}/corr_co_{nn}.json", "w") as fid:
            json.dump(correction_setup[nn], fid, indent=4)


# ==================================================================================================
# --- Function to install beam-beam
# ==================================================================================================
def install_beam_beam(collider, config_collider):
    # Load config
    config_bb = config_collider["config_beambeam"]

    # Install beam-beam lenses (inactive and not configured)
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


# ==================================================================================================
# --- Function to match knobs and tuning
# ==================================================================================================
def set_knobs(config_collider, collider):
    # Read knobs and tuning settings from config file
    conf_knobs_and_tuning = config_collider["config_knobs_and_tuning"]

    # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
    # experimental magnets, etc.)
    for kk, vv in conf_knobs_and_tuning["knob_settings"].items():
        collider.vars[kk] = vv

    return collider, conf_knobs_and_tuning


def match_tune_and_chroma(collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True):
    # Tunings
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


# ==================================================================================================
# --- Function to convert the filling scheme for xtrack, and set the bunch numbers
# ==================================================================================================
def set_filling_and_bunch_tracked(config_bb, ask_worst_bunch=False):
    # Get the filling scheme path
    filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

    # Load and check filling scheme, potentially convert it
    filling_scheme_path = load_and_check_filling_scheme(filling_scheme_path)

    # Correct filling scheme in config, as it might have been converted
    config_bb["mask_with_filling_pattern"]["pattern_fname"] = filling_scheme_path

    # Get number of LR to consider
    n_LR = config_bb["num_long_range_encounters_per_side"]["ip1"]

    # If the bunch number is None, the bunch with the largest number of long-range interactions is used
    if config_bb["mask_with_filling_pattern"]["i_bunch_b1"] is None:
        # Case the bunch number has not been provided
        worst_bunch_b1 = get_worst_bunch(
            filling_scheme_path, numberOfLRToConsider=n_LR, beam="beam_1"
        )
        if ask_worst_bunch:
            while config_bb["mask_with_filling_pattern"]["i_bunch_b1"] is None:
                bool_inp = input(
                    "The bunch number for beam 1 has not been provided. Do you want to use the bunch"
                    " with the largest number of long-range interactions? It is the bunch number "
                    + str(worst_bunch_b1)
                    + " (y/n): "
                )
                if bool_inp == "y":
                    config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = worst_bunch_b1
                elif bool_inp == "n":
                    config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = int(
                        input("Please enter the bunch number for beam 1: ")
                    )
        else:
            config_bb["mask_with_filling_pattern"]["i_bunch_b1"] = worst_bunch_b1

    if config_bb["mask_with_filling_pattern"]["i_bunch_b2"] is None:
        worst_bunch_b2 = get_worst_bunch(
            filling_scheme_path, numberOfLRToConsider=n_LR, beam="beam_2"
        )
        # For beam 2, just select the worst bunch by default
        config_bb["mask_with_filling_pattern"]["i_bunch_b2"] = worst_bunch_b2

    return config_bb


# ==================================================================================================
# --- Function to compute the number of collisions in the IPs (used for luminosity leveling)
# ==================================================================================================
def compute_collision_from_scheme(config_bb):
    # Get the filling scheme path (in json or csv format)
    filling_scheme_path = config_bb["mask_with_filling_pattern"]["pattern_fname"]

    # Load the filling scheme
    if not filling_scheme_path.endswith(".json"):
        raise ValueError(
            f"Unknown filling scheme file format: {filling_scheme_path}. It you provided a csv"
            " file, it should have been automatically convert when running the script"
            " 001_make_folders.py. Something went wrong."
        )

    with open(filling_scheme_path, "r") as fid:
        filling_scheme = json.load(fid)

    # Extract booleans beam arrays
    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    # Assert that the arrays have the required length, and do the convolution
    assert len(array_b1) == len(array_b2) == 3564
    n_collisions_ip1_and_5 = array_b1 @ array_b2
    n_collisions_ip2 = np.roll(array_b1, 891) @ array_b2
    n_collisions_ip8 = np.roll(array_b1, 2670) @ array_b2

    return n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8


# ==================================================================================================
# --- Function to do the Levelling
# ==================================================================================================
def do_levelling(
    config_collider,
    config_bb,
    n_collisions_ip2,
    n_collisions_ip8,
    collider,
    n_collisions_ip1_and_5,
    crab,
):
    # Read knobs and tuning settings from config file (already updated with the number of collisions)
    config_lumi_leveling = config_collider["config_lumi_leveling"]

    # Update the number of bunches in the configuration file
    config_lumi_leveling["ip2"]["num_colliding_bunches"] = int(n_collisions_ip2)
    config_lumi_leveling["ip8"]["num_colliding_bunches"] = int(n_collisions_ip8)

    # Initial intensity
    initial_I = config_bb["num_particles_per_bunch"]

    # First level luminosity in IP 1/5 changing the intensity
    if (
        "config_lumi_leveling_ip1_5" in config_collider
        and not config_collider["config_lumi_leveling_ip1_5"]["skip_leveling"]
    ):
        print("Leveling luminosity in IP 1/5 varying the intensity")
        # Update the number of bunches in the configuration file
        config_collider["config_lumi_leveling_ip1_5"]["num_colliding_bunches"] = int(
            n_collisions_ip1_and_5
        )

        # Do the levelling
        try:
            bunch_intensity = luminosity_leveling_ip1_5(
                collider,
                config_collider,
                config_bb,
                crab=crab,
            )
        except ValueError:
            print("There was a problem during the luminosity leveling in IP1/5... Ignoring it.")
            bunch_intensity = config_bb["num_particles_per_bunch"]

        config_bb["num_particles_per_bunch"] = float(bunch_intensity)

    # Do levelling in IP2 and IP8
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


# ==================================================================================================
# --- Function to add linear coupling
# ==================================================================================================
def add_linear_coupling(conf_knobs_and_tuning, collider, config_mad):
    # Get the version of the optics
    version_hllhc = config_mad["ver_hllhc_optics"]
    version_run = config_mad["ver_lhc_run"]

    # Add linear coupling as the target in the tuning of the base collider was 0
    # (not possible to set it the target to 0.001 for now)
    if version_run == 3.0:
        collider.vars["cmrs.b1_sq"] += conf_knobs_and_tuning["delta_cmr"]
        collider.vars["cmrs.b2_sq"] += conf_knobs_and_tuning["delta_cmr"]
    elif version_hllhc in [1.6, 1.5]:
        collider.vars["c_minus_re_b1"] += conf_knobs_and_tuning["delta_cmr"]
        collider.vars["c_minus_re_b2"] += conf_knobs_and_tuning["delta_cmr"]
    else:
        raise ValueError(f"Unknown version of the optics/run: {version_hllhc}, {version_run}.")

    return collider


# ==================================================================================================
# --- Function to assert that tune, chromaticity and linear coupling are correct before beam-beam
#     configuration
# ==================================================================================================
def assert_tune_chroma_coupling(collider, conf_knobs_and_tuning):
    for line_name in ["lhcb1", "lhcb2"]:
        tw = collider[line_name].twiss()
        assert np.isclose(tw.qx, conf_knobs_and_tuning["qx"][line_name], atol=1e-4), (
            f"tune_x is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['qx'][line_name]}, got {tw.qx}"
        )
        assert np.isclose(tw.qy, conf_knobs_and_tuning["qy"][line_name], atol=1e-4), (
            f"tune_y is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['qy'][line_name]}, got {tw.qy}"
        )
        assert np.isclose(
            tw.dqx,
            conf_knobs_and_tuning["dqx"][line_name],
            rtol=1e-2,
        ), (
            f"chromaticity_x is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['dqx'][line_name]}, got {tw.dqx}"
        )
        assert np.isclose(
            tw.dqy,
            conf_knobs_and_tuning["dqy"][line_name],
            rtol=1e-2,
        ), (
            f"chromaticity_y is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['dqy'][line_name]}, got {tw.dqy}"
        )

        assert np.isclose(
            tw.c_minus,
            conf_knobs_and_tuning["delta_cmr"],
            atol=5e-3,
        ), (
            f"linear coupling is not correct for {line_name}. Expected"
            f" {conf_knobs_and_tuning['delta_cmr']}, got {tw.c_minus}"
        )


# ==================================================================================================
# --- Function to configure beam-beam
# ==================================================================================================
def configure_beam_beam(collider, config_bb):
    collider.configure_beambeam_interactions(
        num_particles=config_bb["num_particles_per_bunch"],
        nemitt_x=config_bb["nemitt_x"],
        nemitt_y=config_bb["nemitt_y"],
    )

    # Configure filling scheme mask and bunch numbers
    if "mask_with_filling_pattern" in config_bb and (
        "pattern_fname" in config_bb["mask_with_filling_pattern"]
        and config_bb["mask_with_filling_pattern"]["pattern_fname"] is not None
    ):
        fname = config_bb["mask_with_filling_pattern"]["pattern_fname"]
        with open(fname, "r") as fid:
            filling = json.load(fid)
        filling_pattern_cw = filling["beam1"]
        filling_pattern_acw = filling["beam2"]

        # Initialize bunch numbers with empty values
        i_bunch_cw = None
        i_bunch_acw = None

        # Only track bunch number if a filling pattern has been provided
        if "i_bunch_b1" in config_bb["mask_with_filling_pattern"]:
            i_bunch_cw = config_bb["mask_with_filling_pattern"]["i_bunch_b1"]
        if "i_bunch_b2" in config_bb["mask_with_filling_pattern"]:
            i_bunch_acw = config_bb["mask_with_filling_pattern"]["i_bunch_b2"]

        # Note that a bunch number must be provided if a filling pattern is provided
        # Apply filling pattern
        collider.apply_filling_pattern(
            filling_pattern_cw=filling_pattern_cw,
            filling_pattern_acw=filling_pattern_acw,
            i_bunch_cw=i_bunch_cw,
            i_bunch_acw=i_bunch_acw,
        )
    return collider


# ==================================================================================================
# --- Function to compute luminosity once the collider is configured
# ==================================================================================================
def record_final_luminosity(collider, config_bb, l_n_collisions, crab):
    
    # Define the IPs in which the luminosity will be computed
    l_ip = ["ip1", "ip2", "ip5", "ip8"]
    
    # Function to compute the luminosity and pile-up
    def twiss_and_compute_lumi(collider, config_bb, l_n_collisions, crab):
        twiss_b1 = collider["lhcb1"].twiss()
        twiss_b2 = collider["lhcb2"].twiss()
        l_lumi = []
        l_PU = []
        for n_col, ip in zip(l_n_collisions, l_ip):
            try:
                L = xt.lumi.luminosity_from_twiss(  # type: ignore
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
                print(f"There was a problem during the luminosity computation in {ip}... Ignoring it.")
                L = 0
                PU = 0
            l_lumi.append(L)
            l_PU.append(PU)
            
        return l_lumi, l_PU
    
    # Get the final luminoisty in all IPs, without beam-beam
    collider.vars["beambeam_scale"] = 0
    l_lumi, l_PU = twiss_and_compute_lumi(collider, config_bb, l_n_collisions, crab)
    
    # Update configuration
    for ip, L, PU in zip(l_ip, l_lumi, l_PU):
        config_bb[f"luminosity_{ip}_without_beam_beam"] = float(L)
        config_bb[f"Pile-up_{ip}_without_beam_beam"] = float(PU)
        
    # Same with beam-beam
    collider.vars["beambeam_scale"] = 1
    l_lumi, l_PU = twiss_and_compute_lumi(collider, config_bb, l_n_collisions, crab)    

    # Update configuration
    for ip, L, PU in zip(l_ip, l_lumi, l_PU):
        config_bb[f"luminosity_{ip}_with_beam_beam"] = float(L)
        config_bb[f"Pile-up_{ip}_with_beam_beam"] = float(PU)
        
    return config_bb


# ==================================================================================================
# --- Main function for collider configuration
# ==================================================================================================
def configure_collider(
    config,
    config_mad,
    context,
    save_collider=False,
    save_config=False,
    return_collider_before_bb=False,
    config_path="config.yaml",
):
    # Generate configuration files for orbit correction
    generate_configuration_correction_files()

    # Get configurations
    config_sim = config["config_simulation"]
    config_collider = config["config_collider"]

    # Rebuild collider
    if config_sim["collider_file"].endswith(".zip"):
        # Uncompress file locally
        with ZipFile(config_sim["collider_file"], "r") as zip_ref:
            zip_ref.extractall()
        collider = xt.Environment.from_json(
            config_sim["collider_file"].split("/")[-1].replace(".zip", "")
        )
    else:
        collider = xt.Environment.from_json(config_sim["collider_file"])

    # Install beam-beam
    collider, config_bb = install_beam_beam(collider, config_collider)

    # Build trackers
    # For now, start with CPU tracker due to a bug with Xsuite
    # Refer to issue https://github.com/xsuite/xsuite/issues/450
    collider.build_trackers()  # (_context=context)

    # Set knobs
    collider, conf_knobs_and_tuning = set_knobs(config_collider, collider)

    # Fix RF cavity lag for both beams
    for line_name in ["lhcb1", "lhcb2"]:
        for ii in collider[line_name].element_names:
            if ii.startswith('acsca'):
                collider[line_name][ii].lag = 180.000000001

    # Match tune and chromaticity
    collider = match_tune_and_chroma(
        collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=True
    )

    config_bb = set_filling_and_bunch_tracked(config_bb, ask_worst_bunch=False)

    # Compute the number of collisions in the different IPs
    (
        n_collisions_ip1_and_5,
        n_collisions_ip2,
        n_collisions_ip8,
    ) = compute_collision_from_scheme(config_bb)

    # Get crab cavities
    crab = False
    if "on_crab1" in config_collider["config_knobs_and_tuning"]["knob_settings"]:
        crab_val = float(config_collider["config_knobs_and_tuning"]["knob_settings"]["on_crab1"])
        if abs(crab_val) > 0:
            crab = True

    # Do the leveling if requested
    if "config_lumi_leveling" in config_collider and not config_collider["skip_leveling"]:
        collider, config_collider = do_levelling(
            config_collider,
            config_bb,
            n_collisions_ip2,
            n_collisions_ip8,
            collider,
            n_collisions_ip1_and_5,
            crab,
        )

    else:
        print(
            "No leveling is done as no configuration has been provided, or skip_leveling"
            " is set to True."
        )

    # Add linear coupling
    collider = add_linear_coupling(conf_knobs_and_tuning, collider, config_mad)

    # Rematch tune and chromaticity
    collider = match_tune_and_chroma(
        collider, conf_knobs_and_tuning, match_linear_coupling_to_zero=False
    )

    # Assert that tune, chromaticity and linear coupling are correct one last time
    assert_tune_chroma_coupling(collider, conf_knobs_and_tuning)

    # Return twiss and survey before beam-beam if requested
    collider_before_bb = None
    if return_collider_before_bb:
        print("Saving collider before beam-beam configuration")
        collider_before_bb = xt.Environment.from_dict(collider.to_dict())

    if not config_bb["skip_beambeam"]:
        # Configure beam-beam
        collider = configure_beam_beam(collider, config_bb)

    # Update configuration with luminosity now that bb is known
    l_n_collisions = [
        n_collisions_ip1_and_5,
        n_collisions_ip2,
        n_collisions_ip1_and_5,
        n_collisions_ip8,
    ]
    config_bb = record_final_luminosity(collider, config_bb, l_n_collisions, crab)

    # Drop update configuration
    with open(config_path, "w") as fid:
        ryaml.dump(config, fid)

    if save_collider:
        # Save the final collider before tracking
        print('Saving "collider_final.json')
        if save_config:
            config_dict = {
                "config_mad": config_mad,
                "config_collider": config_collider,
            }
            collider.metadata = config_dict
        # Dump collider
        collider.to_json("collider_final.json")

    return collider, config_sim, config_bb, collider_before_bb


# ==================================================================================================
# --- Function to prepare particles distribution for tracking
# ==================================================================================================
def prepare_particle_distribution(collider, context, config_sim, config_bb):
    beam = config_sim["beam"]

    particle_df = pd.read_parquet(config_sim["particle_file"])

    r_vect = particle_df["normalized amplitude in xy-plane"].values
    theta_vect = particle_df["angle in xy-plane [deg]"].values * np.pi / 180  # type: ignore # [rad]

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


# ==================================================================================================
# --- Function to do the tracking
# ==================================================================================================
def track(collider, particles, config_sim, save_input_particles=False):
    # Get beam being tracked
    beam = config_sim["beam"]

    # Get the line reference BEFORE optimization (important!)
    # optimize_for_tracking() disables xdeps expressions, which can break Environment's ref_manager
    line = collider[beam]

    # Optimize line for tracking
    line.optimize_for_tracking()

    # Save initial coordinates if requested
    if save_input_particles:
        pd.DataFrame(particles.to_dict()).to_parquet("input_particles.parquet")

    # Track (use the line reference directly, not collider[beam])
    num_turns = config_sim["n_turns"]
    a = time.time()
    line.track(particles, turn_by_turn_monitor=False, num_turns=num_turns)
    b = time.time()

    print(f"Elapsed time: {b-a} s")
    print(f"Elapsed time per particle per turn: {(b-a)/particles._capacity/num_turns*1e6} us")

    return particles


# ==================================================================================================
# --- Main function for collider configuration and tracking
# ==================================================================================================
def configure_and_track(config_path="config.yaml"):
    # Get configuration
    config_gen_1, config_gen_2 = read_configuration(config_path)

    # Get context
    context = get_context(config_gen_2)

    # Tag start of the job
    tree_maker_tagging(config_gen_2, tag="started")

    # Configure collider (not saved, since it may trigger overload of afs)
    collider, config_sim, config_bb, _ = configure_collider(
        config_gen_2,
        config_gen_1["config_mad"],
        context,
        save_collider=config_gen_2["dump_collider"],
        save_config=config_gen_2["dump_config_in_collider"],
        config_path=config_path,
        return_collider_before_bb=False,
    )

    # Compute collider fingerprint
    # (need to be done before tracking as collider can't be twissed after optimization)
    fingerprint = return_fingerprint(config_sim["beam"], collider)
    hash_fingerprint = hash(fingerprint)

    # Reset the tracker to go to GPU if needed
    if config_gen_2["context"] in ["cupy", "opencl"]:
        collider.discard_trackers()
        collider.build_trackers(_context=context)

    # Prepare particle distribution
    particles, particle_id, l_amplitude, l_angle = prepare_particle_distribution(
        collider, context, config_sim, config_bb
    )

    # Track
    particles = track(collider, particles, config_sim)

    # Get particles dictionnary
    particles_dict = particles.to_dict()

    # Convert to dataframe
    particles_df = pd.DataFrame(particles_dict)

    # ! Very important, otherwise the particles will be mixed in each subset
    # Sort by parent_particle_id
    particles_df = particles_df.sort_values("parent_particle_id")

    # Assign the old id to the sorted dataframe
    particles_df["particle_id"] = particle_id

    # Register the amplitude and angle in the dataframe
    particles_df["normalized amplitude in xy-plane"] = l_amplitude
    particles_df["angle in xy-plane [deg]"] = l_angle * 180 / np.pi

    # Add some metadata to the output for better interpretability
    particles_df.attrs["hash"] = hash_fingerprint
    particles_df.attrs["fingerprint"] = fingerprint
    particles_df.attrs["configuration_gen_1"] = config_gen_1
    particles_df.attrs["configuration_gen_2"] = config_gen_2
    particles_df.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save output
    particles_df.to_parquet("output_particles.parquet")

    # Remove the correction folder, and potential C files remaining
    with contextlib.suppress(Exception):
        os.system("rm -rf correction")
        os.system("rm -f *.cc")
    # Tag end of the job
    tree_maker_tagging(config_gen_2, tag="completed")


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    configure_and_track()
