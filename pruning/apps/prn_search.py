from __future__ import annotations

import json
import os
import time

import dynamic_programming_pulsar_search as dpps
import numpy as np
import pulsar_utils as pu

dedisp_files_dtype = np.float32
N_bytes_data_type = 4
C = pu.C
# TODO Add option to plot control plots in some dedicated directory
# TODO Make the log-files written more informative
# TODO Make automatic diagnostic plots for found pulsars
# TODO Make found pulsars report in the first point they are significant

must_have_fields = [
    "fname_signal_e",
    "fname_log",
    "fname_threshold_scheme",
    "dt",
    "pulsar_period_min",
    "pulsar_period_max",
    "p_log_jumps",
    "orbital_period_min",
    "max_mass_function",
    "pulse_profile",
    "max_eccentricity",
    "tolerance_time",
    "log_2_n_segments",
    "n_max_suggestions",
    "poly_order",
    "sigma_clipping_flag",
    "sigma_clipping_bound",
]
default_base_params_dic = dict(
    period_min=0.5e-3,
    period_max=20e-3,
    period_log_jumps=2e-2,
    orbital_period_min="0.8 of the signal's duration",
    p_log_chunk=1e-1,
    n_max_suggestions=2**20,
    poly_order=15,
    max_mass_function=2,
    pulse_profile="single",
    max_eccentricity=0.2,
    sigma_clipping_flag=True,
    sigma_clipping_bound=6,
    log_2_n_segments=8,
)

# Stages in the analysis:
# 1) Copy file to cluster
# 1.5) Split data to files (so that single processes could chew less data)
# 2) Clean data (Whiten/detrend + zeroize RFI) - (Script exists - not distributed yet)
# 3) Dedisperse (Script exists - not distributed yet)
# 4) Prepare json files (V?)
# 5) Run jobs (V?)


def validate_dictionary(dic, fields):
    missing_fields = []
    for field in fields:
        if field not in dic.keys():
            missing_fields.append(field)
    return len(missing_fields) == 0, missing_fields


def create_master_config_file(
    fname_target, signal_fnames, var_fnames, dt_list, save_dir, log_dir, **kwargs
):
    params_dic = default_base_params_dic.copy()
    params_dic["dt_list"] = dt_list
    params_dic["signal_fnames"] = signal_fnames
    params_dic["var_fnames"] = var_fnames
    params_dic["save_dir"] = save_dir
    params_dic["log_dir"] = log_dir
    for item in kwargs.items():
        params_dic[item[0]] = item[1]
    open(fname_target, "wb").write(json.dumps(params_dic).encode())


def create_job_config_files(
    job_dir, master_json_file_fname, n_jobs, str_identifier="pruning", max_n_bins=128
):
    master_dic = json.loads(open(master_json_file_fname, "rb").read().decode())
    signal_fname_list = master_dic.pop("signal_fnames")
    var_fname_list = master_dic.pop("var_fnames")
    dt_list = master_dic.pop("dt_list")
    jobs_per_file = n_jobs // len(signal_fname_list)
    if n_jobs < len(signal_fname_list):
        raise Exception("Number of jobs is not sufficient")
    if n_jobs % len(signal_fname_list) != 0:
        raise Warning(
            "Number of jobs is not an integer multiple of the number of files. %d jobs will be created"
            % (jobs_per_file * len(signal_fname_list))
        )
    period_ticks = 2 ** np.linspace(
        np.log2(master_dic.pop("period_min")),
        np.log2(master_dic.pop("period_max")),
        jobs_per_file + 1,
    )
    period_ranges = [
        [period_ticks[i], period_ticks[i + 1]] for i in range(jobs_per_file)
    ]
    job_index = 0
    for file_ind, fname in enumerate(signal_fname_list):
        for j in range(jobs_per_file):
            job_index += 1
            cur_dic = master_dic.copy()
            if type(cur_dic["orbital_period_min"]) == str:
                cur_dic["orbital_period_min"] = (
                    0.8 * (os.path.getsize(fname) / 4) * dt_list[file_ind]
                )
            cur_dic["pulsar_period_min"] = period_ranges[j][0]
            cur_dic["pulsar_period_max"] = period_ranges[j][1]
            cur_dic["fname_threshold_scheme"] = (
                cur_dic["save_dir"]
                + str_identifier
                + "_threshold_scheme_job_"
                + str(job_index)
                + ".num"
            )
            cur_dic["fname_log"] = (
                cur_dic["log_dir"]
                + str_identifier
                + "_log_file_job_"
                + str(job_index)
                + ".txt"
            )
            cur_dic["fname_signal_e"] = fname
            cur_dic["fname_signal_v"] = var_fname_list[file_ind]
            cur_dic["dt"] = dt_list[file_ind]
            cur_dic["tolerance_time"] = max(
                period_ranges[j][0] / max_n_bins, cur_dic["dt"]
            )
            fname_job_file = job_dir + str_identifier + "_" + str(job_index) + ".json"
            open(fname_job_file, "wb").write(json.dumps(cur_dic).encode())
            print(
                "written job",
                job_index,
                "to file:",
                fname_job_file,
                "with signal file:",
                fname,
                "and with period range",
                period_ranges[j],
            )


def search_time_series(params_dic, computing_clients=None, return_dyp_flag=False):
    fname_signal_e = params_dic["fname_signal_e"]
    fname_signal_v = params_dic.get("fname_signal_v", "unspecified")
    # fname_rfi_mask = params_dic.get("fname_rfi_mask","")
    dt = params_dic["dt"]
    log_file_name = params_dic["fname_log"]

    log_file = open(log_file_name, "a")
    log_file.write("Initializing all the data. start_time:" + time.ctime() + "\n")
    log_file.flush()
    start = params_dic.get("start", 0)
    end = params_dic.get("end", "default")
    signal_e = np.fromfile(fname_signal_e, dtype=dedisp_files_dtype)
    if end == "default":
        end = len(signal_e)
    signal_e = signal_e[start:end]

    if fname_signal_v == "unspecified":
        signal_v = np.ones(signal_e.shape, dtype=np.float32) * np.var(signal_e)
    else:
        signal_v = np.fromfile(fname_signal_v, dtype=dedisp_files_dtype)[start:end]

    log_file.write("Duration of loaded data:" + str(len(signal_e) * dt) + "\n")
    p_pulsar_min = params_dic["pulsar_period_min"]
    p_pulsar_max = params_dic["pulsar_period_max"]
    p_pulsar_log_jumps = params_dic["period_log_jumps"]
    periods = np.e ** (
        np.arange(np.log(p_pulsar_min), np.log(p_pulsar_max), p_pulsar_log_jumps)
    )

    orbital_period_min = params_dic["orbital_period_min"]
    omega_max = 2 * np.pi / orbital_period_min

    max_companion_mass = params_dic["max_mass_function"]  # in units of solar mass
    x_max = pu.semi_major_axis(
        max_companion_mass, orbital_period_min
    ).cgs.value  # light seconds
    pulse_profile = params_dic["pulse_profile"]
    ecc_max = params_dic["max_eccentricity"]
    tolerance_time = params_dic["tolerance_time"]
    log_2_n_segments = params_dic["log_2_n_segments"]
    n_segments = 2**log_2_n_segments
    n_max = params_dic["n_max_suggestions"]
    poly_order = params_dic["poly_order"]
    sigma_clipping_flag = params_dic["sigma_clipping_flag"]
    sigma_clipping_bound = params_dic["sigma_clipping_bound"]
    threshold_scheme_fname = params_dic["fname_threshold_scheme"]
    if log_2_n_segments == 7:
        survival_prob_changes = params_dic.get(
            "survival_prob_changes", [2, 2, 2, 4, 1, 1, 1, 1, 2]
        )
    if log_2_n_segments == 8:
        survival_prob_changes = params_dic.get(
            "survival_prob_changes", [4, 4, 2, 2, 1, 1, 2, 1, 1] + [0.95] * 30
        )
    else:
        survival_prob_changes = params_dic.get("survival_prob_changes", [])

    pos_trials = params_dic.get("pos_trials", range(0, n_segments, n_segments // 16))

    log_file.write("performing a search with:\n")
    log_file.write("x_max = " + str(x_max) + " light seconds" + "\n")
    log_file.write("orbital period min: " + str(orbital_period_min) + "\n")
    log_file.write(
        "maximal velocity difference (as a fraction of C): "
        + str(2 * x_max * omega_max)
        + "\n"
    )
    log_file.write("pulse_profile = " + pulse_profile + "\n")
    log_file.write("n_segments = " + str(n_segments) + "\n")

    for per_min in periods:
        log_file.write("-------------------------------------------------\n")
        t_total = time.time()
        log_file.write(
            "Processing period: " + str(np.round(per_min * 1000, 4)) + " [ms]" + "\n"
        )
        # 20% overlap, to make sure no pulsar is "falling between the chairs"
        per_max = min(per_min * (1 + p_pulsar_log_jumps * 1.1), p_pulsar_max)
        # duty_cycle = np.max([1/64.,2*dt/per_max])
        # period_range = [per_min,per_max]

        n_bins = params_dic.get("n_bins", int(per_min / tolerance_time))
        log_file.write("Using " + str(n_bins) + " bins in a fold.\n")

        tolerance_bins = tolerance_time / dt

        length = len(signal_e)
        log_file.write(
            "need jerk dynamic programming?: "
            + str(
                x_max
                * omega_max**3
                * ((length / n_segments * dt) ** 3 / 6)
                / tolerance_time
            )
            + "\n"
        )
        a_min, a_max = -x_max * C * omega_max**2, x_max * C * omega_max**2

        t_init = time.time()

        # print((len(signal_e), len(signal_v), dt, per_min, per_max, a_min, a_max, n_bins, tolerance_bins))
        print(type(signal_e), type(signal_v))
        dyp = dpps.acceleration_search_dynamic_programming(
            signal_e,
            signal_v,
            dt,
            per_min,
            per_max,
            a_min,
            a_max,
            n_bins,
            tolerance_bins,
        )
        dyp.init_data_structure()
        t_init = time.time() - t_init
        log_file.write("log_2_n_segments = " + str(log_2_n_segments) + "\n")
        n_segments = 2**log_2_n_segments
        n_dynamic_programming_iterations = (
            int(np.log2(dyp.data_structure.shape[0])) - log_2_n_segments
        )
        dyp.do_iterations(n_dynamic_programming_iterations)

        if sigma_clipping_flag:
            log_file.write(
                "Performing sigma clipping with sigma_clipping_bound = "
                + str(sigma_clipping_bound)
                + "\n"
            )
            # BUG!!!! The sigma clipping destroys the dyp!!!
            bad_fold_indicator = np.sum(
                (
                    dyp.data_structure[:, :, :, 0, :] ** 2
                    / dyp.data_structure[:, :, :, 1, :]
                )
                > sigma_clipping_bound**2,
                -1,
            )
            dyp.data_structure = pu.mask_zipped(dyp.data_structure, bad_fold_indicator)

        dyp_shape = dyp.data_structure.shape
        log_file.write("DYP dimensions: " + str(dyp_shape) + "\n")
        log_file.write("n_max_suggestions: " + str(np.round(np.log2(n_max), 2)) + "\n")
        log_file.flush()
        if return_dyp_flag:
            return dyp

        t_prune = time.time()
        prn = dpps.chebychev_search_pruning(
            dyp,
            dt,
            tolerance_time,
            x_max,
            orbital_period_min,
            ecc_max,
            poly_order,
            n_max_suggestions=n_max,
            pulse_profile=pulse_profile,
        )
        if params_dic.get("threshold_scheme", None) is None and not os.path.isfile(
            params_dic.get("fname_threshold_scheme", "")
        ):
            print("Initializing threshold scheme:")

            threshold_scheme, branching_pattern = prn.generate_threshold_scheme(
                survival_prob_changes, n_segments, ref_snr="default"
            )

            open(threshold_scheme_fname, "wb").write(
                np.array(threshold_scheme).astype(np.float32).tostring()
            )
            params_dic["threshold_scheme"] = threshold_scheme
            log_file.write(
                "Threshold scheme written to: " + threshold_scheme_fname + "\n"
            )

            n_enum_options = sum(np.log2(np.array(branching_pattern))) + 15 + 20
            log_file.write("look elsewhere effect: " + str(n_enum_options) + "\n")

            bound = params_dic.get("bound", np.sqrt((n_enum_options) * np.log(2) * 2))

        elif params_dic.get("threshold_scheme", None) == "approx":
            log_file.write("using an approximate threshold scheme\n")
            branching_pattern = prn.get_branching_pattern(n_segments)
            n_enum_options = sum(np.log2(np.array(branching_pattern))) + 15 + 20
            bound = params_dic.get("bound", np.sqrt((n_enum_options) * np.log(2) * 2))
            threshold_scheme = dpps.default_threshold_scheme(n_segments, bound)
        else:
            if params_dic.get("fname_threshold_scheme", None) is None:
                threshold_scheme = params_dic.get("threshold_scheme")
                log_file.write("Using previously calculated threshold scheme\n")

            else:
                threshold_scheme = np.fromfile(
                    params_dic.get("fname_threshold_scheme"), np.float32
                )
                log_file.write(
                    "Reading threshold scheme from: "
                    + params_dic.get("fname_threshold_scheme")
                    + "\n"
                )

            bound = params_dic.get("bound", threshold_scheme[-1] + 2)

        log_file.write("used bound: " + str(bound) + "\n")
        res = prn.pruning_enumeration(
            threshold_scheme,
            dyp.data_structure,
            bound,
            pos_trials,
            computing_clients=computing_clients,
            log_file_name=log_file_name,
            lazy=True,
        )
        t_total = time.time() - t_total
        t_prune = time.time() - t_prune

        log_file.write(
            "total_time:"
            + str(t_total)
            + "("
            + str(t_init)
            + ","
            + str(t_prune)
            + ")"
            + "\n"
        )

        if len(res) > 0:
            res_ind = np.argmax(prn.suggestion_structure[0])
            log_file.write("(*) found significant result!" + "\n")
            log_file.write(
                "(*) score:" + str(np.max(prn.suggestion_structure[0])) + "\n"
            )
            log_file.write(
                "(*) period:" + str(prn.suggestion_structure[1][res_ind][-2, 0]) + "\n"
            )
            log_file.write("(*)" + str(prn.suggestion_structure[1][res_ind]) + "\n")
            log_file.write("(*)reference index" + str(prn.data_access_scheme[0]) + "\n")
            log_file.write("(*) fname:" + fname_signal_e + "\n")
            if params_dic.get("manual_inspection", False):
                return prn, dyp, res

        del prn, dyp


def _main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Will peform pruning search on a clean, dedispersed file"
    )
    parser.add_argument(
        "params_dic_fname",
        type=str,
        help="a json file containing a dictionary with all keyword argumetns",
    )
    args = parser.parse_args()
    params_dic = json.loads(open(args.params_dic_fname, "rb").read())
    valid, missing_fields = validate_dictionary(params_dic, must_have_fields)
    if valid:
        search_time_series(params_dic)
    else:
        print("Failed to validate all parameters in the json file")
        # TODO add descriptions for all fields and print their description.
        print("Missing Fields: ", missing_fields)


if __name__ == "__main__":
    _main()

# parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
# parser.add_argument('fname_sig_e', help="signal E channel, assumed to be already cleaned and dedispersed")
# parser.add_argument('fname_sig_v', help="signal V channel")
# parser.add_argument('fname_rfi_mask', help ="file containing the rfi masks")
# parser.add_argument('log_file_name', help="name of file to log results and trials")

# parser.add_argument('--dt',type=float, default=8.192e-5, help="the time interval between subsequent samples")
# parser.add_argument('--p_min',type=float, default=1e-3, help="the minimal pulsar period for the scan")
# parser.add_argument('--p_max',type=float, default=1e-1, help="the maximal pulsar period for the scan")
# parser.add_argument('--p_log_jumps',type=float, default=1e-2, help="the jumps in pulsar period to be scanned together (should be larger than V_max/C ~= 1e-3)")
# parser.add_argument('--orbital_period_min', type=float, default=10000, help="The minimal orbital period to be scanned")
# parser.add_argument('--tolerance_time',type=float,default=1.6384e-4, help = "The effective time resolution for all the folds in the algorithm")
# parser.add_argument('--t_end', type = int, default=-1, help = 'End position to be used [integer number of bins]')
# parser.add_argument('--t_start', type = int, default = 0, help = 'Starting position to be used [integer number of bins]')

# def inspect_result(running_line_text, p_min, p_max, mass = 4):
#
#     #text = "E:/Observations/PulsarScott/dedispersed_54873/dedisp_e238.75.dedispersed E:/Observations/PulsarScott/dedispersed_54873/dedisp_v238.75.dedispersed E:/Observations/PulsarScott/dedispersed_54873/rfi_masks/rfi_mask_134217728_268435456.mask E:/Observations/PulsarScott/search_logs/DM238.75_new.txt --orbital_period_min=10995.1162778 --p_min=0.0011 --p_max=0.001221 --dt=8.192e-05 --tolerance_time=0.00016384"
#     fname_sig_e, fname_sig_v, fname_rfi_mask ,log_file_name  = running_line_text.split()[:4]
#     kwargs = {x.split('=')[0][2:]:float(x.split('=')[1]) for x in running_line_text.split()[4:]}
#     dt = kwargs.pop('dt')
#     kwargs['p_min'] = p_min
#     kwargs['p_max'] = p_max
#     kwargs['max_companion_mass'] = mass
#     prn, dyp, res, signal_e, signal_v = search_time_series(fname_sig_e, fname_sig_v, fname_rfi_mask, dt, log_file_name, manual_inspection = True, **kwargs)
#     pl = dyp.param_list[1]
#     best_ind = np.argmax(prn.suggestion_structure[0])
#     recovered_per_list = [pl[prn.resolve_func(prn.suggestion_structure[1][best_ind], prn.data_access_scheme[0], i)[0][1]] for i in range(128)]
#     recovered_acc_list = [pl[prn.resolve_func(prn.suggestion_structure[1][best_ind], prn.data_access_scheme[0], i)[0][0]] for i in range(128)]
#     time_axis = np.arange(128)*prn.chunk_duration
#     return prn, dyp, res, signal_e, signal_v, recovered_per_list, recovered_acc_list, time_axis


# example usage: pss.search_time_series_in_chunks('E:\Observations\PulsarScott\dedispersed2\dedisp_e239.0.dedispersed','E:\Observations\PulsarScott\dedispersed2\dedisp_v239.0.dedispersed', 'E:\Observations\PulsarScott\search_logs\DM239_try', 1e-3, 2e-3, 1e-2, 10000)

# %% example usage:
# existing interesting pulsars: period = 11.5632	DM = 242.1  orbital_period = 6535
# DM = 238.75
# time_stamp = time.ctime().replace(' ','_').replace(':','-')
# fname_signal_e = 'E:/Observations/PulsarScott/dedispersed_54873/dedisp_e'+str(DM)+'.dedispersed'
# fname_signal_v = 'E:/Observations/PulsarScott/dedispersed_54873/dedisp_v'+str(DM)+'.dedispersed'
# fname_mask = 'E:/Observations/PulsarScott/dedispersed_54873/rfi_masks/rfi_mask_0_268435456.mask'
# fname_log = 'E:/Observations/PulsarScott/search_logs/DM'+ str(DM)+ '_test' + time_stamp + '.txt'
# fname_params = 'E:/Observations/PulsarScott/search_logs/params_file_' + time_stamp + '.json'
# threshold_scheme_fname = 'E:/Observations/PulsarScott/search_logs/threshold_scheme' + time_stamp + '.num'
# period_min = 2.02e-3
# period_max = 4e-3
# period_log_jumps = 5e-3 #should be small, and simultaneusly bigger than (maximal velocity) / C
# dt = 8.192e-5
# tolerance_time = 8.192e-5
# orbital_period_min = 2**28*8.192e-5
# p_log_chunk = 5e-2
# search_params = dict(max_mass_function = 1, n_max_suggestions = 2**19, log_2_n_segments = 8, threshold_scheme_fname = threshold_scheme_fname)
#
# example usage: pss.search_time_series_in_chunks(pss.fname_signal_e, pss.fname_signal_v,pss.fname_mask,pss.fname_log, pss.period_min, pss.period_max, pss.p_log_chunk, pss.dt,pss.tolerance_time,pss.orbital_period_min,search_params = pss.search_params, params_dic_fname = pss.fname_params)
# %%


# def search_time_series_in_chunks(fname_signal_e, fname_signal_v, fname_rfi_mask, log_file_name, p_min,
#                        p_max , p_log_chunk, dt,tolerance_time, orbital_period_min, search_params = {}, params_dic_fname = 'empty'):
#     if params_dic_fname != 'empty':
#         open(params_dic_fname,'wb').write(json.dumps(search_params))
#         print("saving the search params in:" + params_dic_fname)
#     import os
#     running_line = "python pulsar_search_script.py "
#     running_line += fname_signal_e + ' '
#     running_line += fname_signal_v + ' '
#     running_line += fname_rfi_mask + ' '
#     running_line += log_file_name + ' '
#     running_line += '--params_dic_fname=' + params_dic_fname + ' '
#     running_line += '--orbital_period_min=' + str(orbital_period_min) + ' '
#     for per_min in np.e**np.arange(np.log(p_min), np.log(p_max), p_log_chunk):
#         running_line_tmp = running_line + '--p_min=' + str(per_min) + ' --p_max=' + str(per_min*(1+p_log_chunk*1.1)) + ' --dt=' + str(dt) + ' --tolerance_time=' + str(tolerance_time)
#         with open(log_file_name, 'a', 0) as log_file:
#             log_file.write('(#)' + running_line_tmp + '\n')
#         os.system(running_line_tmp)
#     return


# def clean_data():
#     if not clean:
#         signal_e = np.fromfile(fname_signal_e, dtype=dedisp_files_dtype)[t_start:t_end]
#         log_file.write('Cleaning_data.\n')
#         tmp = time.time()
#         mask_t,mask_f = pickle.loads(open(fname_rfi_mask,'rb').read())
#         pu.clean_dedispersed_e(signal_e, dt, mask_t, mask_f)
#         log_file.write('Saving cleaned data to: ' + fname_signal_e +'_' + str(t_start) + '_' + str(t_end) +'.clean'+'\n')
#         open(fname_signal_e +'_' + str(t_start) + '_' + str(t_end) +'.clean','wb').write(signal_e.tostring())
#         log_file.write('Clean time = ' + str(time.time()-tmp) +'\n')

#     else:
#        signal_v = np.fromfile(fname_signal_v, dtype=dedisp_files_dtype)[t_start:t_end]
#        pu.clean_dedispersed_v(signal_v, mask_t)
#        log_file.write('Saving variance of cleaned data to: ' + fname_signal_v + '.clean'+'\n')
#        open(fname_signal_v +'_' + str(t_start) + '_' + str(t_end) +'.clean','wb').write(signal_v.tostring())
