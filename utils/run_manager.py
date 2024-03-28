

import os
import glob
import numpy as np
import pandas as pd

from utils.evaluate import *


def check_directories(runs, n_images):
    flag = True
    for directory in runs:
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            flag = False
            continue
        png_files = glob.glob(f"{directory}/*.png")
        if not (len(png_files) >= n_images):
            print(f"Directory {directory} has only {len(png_files)}/{n_images} PNG images")
            flag = False

    return flag

def get_flags_for_experiment(experiment):
    assert isinstance(experiment, str), "get_flags_for_experiment's input must be a string"

    if experiment == "adjusted_emb":
        return {"baseline": True, "adjusted": True, "embedding": True, "no_flex": False, "first": False, "flex": False, "shift1": False}
    elif experiment == "adjusted_flex":
        return {"baseline": True, "adjusted": True, "embedding": False, "no_flex": False, "first": False, "flex": True, "shift1": False}
    elif experiment == "all":
        return {"baseline": True, "adjusted": True, "embedding": True, "no_flex": True, "first": True, "flex": True, "shift1": False}
    elif experiment == "pre":
        return {"baseline": True, "adjusted": False, "embedding": True, "no_flex": True, "first": True, "flex": True, "shift1": False}
    elif experiment == "flex":
        return {"baseline": True, "adjusted": False, "embedding": True, "no_flex": False, "first": True, "flex": True, "shift1": False}
    elif experiment == "shift1":
        return {"baseline": True, "adjusted": True, "embedding": True, "no_flex": False, "first": False, "flex": False, "shift1": True}
    elif experiment == "expexp":
        return {"baseline": True, "adjusted": True, "embedding": True, "no_flex": False, "first": False, "flex": False, "shift1": False, "expexp": True}
    elif experiment == "variants":
        return {"baseline": True, "adjusted": True, "embedding": True, "shift1": True, "expexp": True}
    elif experiment == "ablation":
        return {"adjusted": True, "embedding": True, "ablation": True}
    elif experiment == "QN":
        return {"adjusted": True, "embedding": True, "expexp": True, "exact": True}
    elif experiment == "sr":
        return {"adjusted": True, "embedding": True, "nosr": True}
    elif experiment == "nearest":
        return {"embedding": True, "nosr": True}
    else:
        raise ValueError(f"Unknown experiment: {experiment}, must be one of {list_experiments()}")

def list_experiments():
    return ["adjusted_emb", "adjusted_flex", "all", "pre", "flex", "shift1", "expexp", "variants", "ablation", "QN", "sr","nearest"]

def get_runs_and_names(experiment,  n_steps, prompt = "morgana2", directory = "images", check_for = 0, experiment_flags = None,
                       baseline = True, adjusted = False, embedding = False, no_flex = False, first = False, flex = False, 
                       shift1 = False, expexp = False, ablation = False, exact = False, nosr = False):
    
    if experiment_flags is not None:
        experiment_flags = get_flags_for_experiment(experiment_flags)
        return get_runs_and_names(experiment, n_steps, prompt, directory, check_for, **experiment_flags)


    if experiment == "baseline":
        runs = [
        f'{directory}/{prompt}x{n_steps}_M23E8',
        f'{directory}/{prompt}x{n_steps}_M10E5_all',
        f'{directory}/{prompt}x{n_steps}_M7E8_all',
        ]

        row_names = ["fp32", "M10E5", "M7E8"]
    elif experiment in ["M10E5", "M7E8"]:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}')
        row_names = ["vanilla"]
    else:
        runs = []
        row_names = []
        if baseline:
            runs.append(f'{directory}/{prompt}x{n_steps}_M23E8')
            row_names.append("fp32")
        if no_flex:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}')
            row_names.append(f"vanilla no flex, no emb")
        if flex:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex')
            row_names.append(f"vanilla no emb")

        if adjusted and flex:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_adjusted')
            row_names.append(f"adjusted no emb")
        if embedding:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding')
            row_names.append(f"vanilla")
        if embedding and nosr:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_nearest')
            row_names.append(f"nearest")

        if adjusted and embedding:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_adjusted')
            row_names.append(f"adjusted")
        if first:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_not_embedding')
            row_names.append(f"quantized in/out")
        if shift1:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_adjusted_shift1')
            row_names.append(f"shifted Q")
        if expexp:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_adjusted_QN_expexp')
            row_names.append(f"expexp interp")

        if ablation:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_max_adjusted')
            row_names.append(f"Q -> max(Q)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_average_adjusted')
            row_names.append(f"Q-> mean(Q)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_min_adjusted')
            row_names.append(f"Q-> min(Q)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_sqrt_adjusted')
            row_names.append(f"Q-> sqrt(Q)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_square_adjusted')
            row_names.append(f"Q-> Q**2")
        if exact:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_adjusted_QN_exact')
            row_names.append(f"Exact Q")
        if nosr and adjusted and embedding:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_adjusted_nearest')
            row_names.append(f"adjusted nearest")

    runs = [(run if run != "images/morgana2x400_M3E4_flex_embedding_adjusted" else "images/morgana2x400_M3E4_flex_embedding_adjusted_again") for run in runs]

    if check_for:
        check_directories(runs, check_for)

    return runs, row_names


def highlight_max(df, exclude = 0):
    # Create a style function
    def style_func(s):
        if exclude == 0:
            is_max = s == s.iloc[1:].max()
        else:
            is_max = s == s.iloc[1:-exclude].max()
        return ['font-weight: bold' if v else '' for v in is_max]

    # Apply the style function to the DataFrame, excluding the first column
    df_styled = df.style.apply(style_func, axis=1)
    # Set the precision
    df_styled.format("{:.3f}")

    return df_styled

def to_pd(results_dict, results_dict_steps,  col_names):
    experiments = list(results_dict.keys())
    steps = results_dict_steps[experiments[0]]
    row_names  = []
    rows = []

    for experiment in experiments:
        for i,step in enumerate(steps):
            row_names += [f"{experiment} {step}"]
            rows.append([np.round(res[i],3) for res in results_dict[experiment].values()])

    df = pd.DataFrame(rows,  index = row_names, columns = col_names)



    return df


def get_results(experiments = ["M4E3","M3E4"], steps = [400,800], images_per_run = 64,  directory = "images" ,experiment_flags = "ablation", 
                dry = False, quiet = False,
                pvals = []):
    
    if not isinstance(pvals, list):
        pvals = [pvals]

    results_dict_ssim = {}
    results_dict_ssim_std = {}
    results_dict_steps = {}

    for experiment in experiments:
        results_dict_ssim[experiment] = {}
        results_dict_ssim_std[experiment] = {}
        results_dict_steps[experiment] = []
        for n_steps in steps:
            runs, row_names = get_runs_and_names(experiment, n_steps, directory=directory, experiment_flags= experiment_flags)
            if check_directories(runs, images_per_run):
                if not quiet:
                    print(f"Experiment: {experiment}, n_steps: {n_steps}")
                if dry:
                    continue
                ssim, std = eval_mse_matrix(runs, images_per_run, fn_desc = "ssim+", stop = True)




                ssim = ssim[0,:]
                std = std[0,:]

                for k in range(len(row_names)):
                    if row_names[k] not in results_dict_ssim[experiment]:
                        results_dict_ssim[experiment][row_names[k]] = []
                        results_dict_ssim_std[experiment][row_names[k]] = []
                    results_dict_ssim[experiment][row_names[k]] += [ssim[k]]
                    results_dict_ssim_std[experiment][row_names[k]] += [std[k]]

                for u,v in pvals:
                    p = eval_mse_pval(runs[0],runs[u], runs[v], images_per_run  ,  fn_desc = "ssim+")
                    p_row_name = f"Pval({row_names[u]} > {row_names[v]})"
                    row_names.append(p_row_name)
                    if p_row_name not in results_dict_ssim[experiment]:
                        results_dict_ssim[experiment][p_row_name] = []
                        results_dict_ssim_std[experiment][p_row_name] = []
                    results_dict_ssim[experiment][p_row_name] += [p]
                    results_dict_ssim_std[experiment][p_row_name] += [0]

                results_dict_steps[experiment] += [n_steps]
    df = to_pd(results_dict_ssim, results_dict_steps, row_names)

    df_styled = highlight_max(df, len(pvals))

    return df_styled





def merge(df1, df2, *args):
    
    if isinstance(df1, pd.io.formats.style.Styler):
        df1 = df1.data
    if isinstance(df2, pd.io.formats.style.Styler):
        df2 = df2.data
    
    # Merge the dataframes
        
    merged_idx = df1.index.intersection(df2.index)
    merged = pd.merge(df1, df2, how='outer', on=df1.columns.intersection(df2.columns).tolist())

    # Fill missing values with "-"
    merged.fillna(np.nan, inplace=True)

    merged.set_index(df1.index, inplace=True)

    # Check for similarity in shared fields
    shared_columns = df1.columns.intersection(df2.columns)
    # for column in shared_columns:
    #     assert all(df1[column] == df2[column]), f"Column {column} has different values in the two dataframes"

    if args:
        new_args = args[1:]
        return merge(merged, args[0], *new_args)

    return highlight_max(merged)