

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
    if experiment == "emb":
        return {"baseline": True, "embedding": True}
    elif experiment == "adjusted_flex":
        return {"baseline": True, "adjusted": True, "embedding": False, "no_flex": False, "first": False, "flex": True, "shift1": False}
    elif experiment == "all":
        return {"baseline": True, "adjusted": True, "embedding": True, "no_flex": True, "first": True, "flex": True, "shift1": False}
    elif experiment == "pre":
        return {"baseline": True, "adjusted": False, "embedding": True, "no_flex": True, "first": True, "flex": True, "shift1": False}
    elif experiment == "flex":
        return {"baseline": True, "adjusted": False, "embedding": True, "no_flex": False, "first": False, "flex": True, "shift1": False}
    elif experiment == "shift1":
        return {"baseline": True, "adjusted": True, "embedding": True, "no_flex": False, "first": False, "flex": False, "shift1": True}
    elif experiment == "expexp":
        return {"baseline": True, "adjusted": True, "embedding": True, "no_flex": False, "first": False, "flex": False, "shift1": False, "expexp": True}
    elif experiment == "variants":
        return {"baseline": True, "adjusted": True, "embedding": True, "shift1": True, "expexp": True}
    elif experiment == "ablation":
        return {"adjusted": True, "embedding": True, "ablation": True, "shift1": True, "expexp": True}
    elif experiment == "QN":
        return {"adjusted": True, "embedding": True, "expexp": True, "exact": True}
    elif experiment == "sr":
        return {"adjusted": True, "embedding": True, "nosr": True}
    elif experiment == "nearest":
        return {"embedding": True, "nosr": True}
    elif experiment == "desperate1":
        return {"adjusted": True, "embedding": True, "Qfractions": True}
    elif experiment == "stem":
        return {"embedding": True, "stem": True, "STEM": True}
    elif experiment == "partial":
        return {"embedding": True, "partialQ": True}
    elif experiment == "stoch_w":
        return {"embedding": True, "stochastic_weights": True}
    elif experiment == "stoch_w_adj":
        return {"embedding": True, "stochastic_weights": True, "adjusted": True}
    elif experiment == "stem_emb":
        return {"embedding": True, "stem": True, "STEM": True,  "adjusted": True}
    elif experiment == "Wsr":
        return {"embedding": True, "Wsr": True}
    elif experiment == "traditional":
        return {"traditional": True}
    elif experiment == "extended":
        return {"embedding": True, "nosr": True, "extended": True}
    elif experiment == "repeated":
        return {"embedding": True, "nosr": True, "extended": True, "repeated": True}
    elif experiment == "extended4":
        return {"embedding": True, "nosr": True, "extended": True, "x4": True}
    elif experiment == "repeated4":
        return {"embedding": True, "nosr": True, "extended": True, "repeated": True, "x4": True}
    else:
        raise ValueError(f"Unknown experiment: {experiment}, must be one of {list_experiments()}")

def list_experiments():
    return ["adjusted_emb",  "emb","adjusted_flex", "all", "pre", "flex", "shift1", 
            "expexp", "variants", "ablation", "QN", "sr","nearest",
            "stem", "stem_emb", "partial", "desperate1", "stoch_w", "stoch_w_adj", 
            "Wsr", "traditional", "extended", "repeated","repeated4"]

def get_runs_and_names(experiment,  n_steps, prompt = "morgana2", fp32_baseline = True, directory = "images", check_for = 0, experiment_flags = None,
                       baseline = True, adjusted = False, embedding = False, no_flex = False, first = False, flex = False, 
                       shift1 = False, expexp = False, ablation = False, exact = False, nosr = False, stem = False, STEM=   False,
                       Qfractions = False, partialQ = False, stochastic_weights = False, Wsr = False, plus = -1, extended = False , 
                       x3 = False , x4 = False, ceil = False, traditional = False, repeated = False):
    
    if experiment_flags is not None:

        if isinstance(experiment_flags, str):
            experiment_flags = get_flags_for_experiment(experiment_flags)
            return get_runs_and_names(experiment, n_steps, prompt, fp32_baseline, directory, check_for, plus =plus, **experiment_flags)
        elif isinstance(experiment_flags, dict):
            return get_runs_and_names(experiment, n_steps, prompt, fp32_baseline, directory, check_for, plus = plus, **experiment_flags)


    runs = []
    row_names = []
    if baseline:
        if fp32_baseline:
            runs.append(f'{directory}/{prompt}x{n_steps}_fp32')
            row_names.append("fp32")
        else:
            runs.append(f'{directory}/{prompt}x{n_steps}_fp16')
            row_names.append("fp16")

    if traditional:
        runs.append(f'{directory}/{prompt}x{n_steps}_bf16')
        row_names.append("bf16")

        # runs.append(f'{directory}/{prompt}x{n_steps}_fp16')
        # row_names.append("fp16")
        runs.append(f'{directory}/{prompt}x{n_steps}_qfp16')
        row_names.append("fp16_quantized")

    if embedding and nosr:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_nearest')
        row_names.append(f"vanilla")
        if ceil:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_nearest_ceil')
            row_names.append(f"vanilla ceil")
    if no_flex:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_staticBias')
        row_names.append(f"SR no flex, no emb")
    if flex:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_noemb')
        row_names.append(f"SR no emb")


    if adjusted and flex:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_adjusted')
        row_names.append(f"SR adjusted no emb")
    if embedding:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}')
        row_names.append(f"SR")

        if ceil:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_ceil')
            row_names.append(f"SR ceil")


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

    if stem:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stem1_flex_embedding')
        row_names.append(f"stochastic p_emb")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stem2_flex_embedding')
        row_names.append(f"stochastic t_emb")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stem3_flex_embedding')
        row_names.append(f"stochastic embs")
        if adjusted:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stem1_flex_embedding_adjusted')
            row_names.append(f"stochastic p_emb (adjusted)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stem2_flex_embedding_adjusted')
            row_names.append(f"stochastic t_emb (adjusted)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stem3_flex_embedding_adjusted')
            row_names.append(f"stochastic embs (adjusted)")
    if STEM:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_STEM1_flex_embedding')
        row_names.append(f"stochastic p_emb (+refiner)")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_STEM2_flex_embedding')
        row_names.append(f"stochastic t_emb (+refiner)")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_STEM3_flex_embedding')
        row_names.append(f"stochastic embs (+refiner)")
        if adjusted:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_STEM1_flex_embedding_adjusted')
            row_names.append(f"stochastic p_emb (+refiner, adjusted)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_STEM2_flex_embedding_adjusted')
            row_names.append(f"stochastic t_emb (+refiner, adjusted)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_STEM3_flex_embedding_adjusted')
            row_names.append(f"stochastic embs (+refiner, adjusted)")
    if Qfractions:
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_half_adjusted')
        row_names.append(f"Q -> Q/2")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_third_adjusted')
        row_names.append(f"Q -> Q/3")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_quarter_adjusted')
        row_names.append(f"Q -> Q/4")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_fifth_adjusted')
        row_names.append(f"Q -> Q/5")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_eighth_adjusted')
        row_names.append(f"Q -> Q/8")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_tenth_adjusted')
        row_names.append(f"Q -> Q/10")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_thenth_adjusted')
        row_names.append(f"Q -> Q/30")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_flex_embedding_SQ_hundredth_adjusted')
        row_names.append(f"Q -> Q/100")
    if partialQ:
        runs.append(f'{directory}/{prompt}x{n_steps}_A_M23E8_W_{experiment}_flex_embedding')
        row_names.append(f"Q_w")
        runs.append(f'{directory}/{prompt}x{n_steps}_A_{experiment}_W_M23E8_flex_embedding')
        row_names.append(f"Q_a")

    if Wsr:
        if plus is None or plus == -1:
            plus = 23

        if plus >= 1:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_Wsr_M5E3')
            row_names.append(f"+1")
        if plus >= 2:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_Wsr_M6E3')
            row_names.append(f"+2")
        if plus >= 4:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_Wsr_M8E3')
            row_names.append(f"+4")

        if plus >= 6:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_Wsr')
            row_names.append(f"+6")



    if extended or repeated:
        run, name = plus_run(directory, prompt, n_steps, experiment, plus)
        runs.append(run)
        row_names.append(name)

    if extended:
        run, name = plus_run(directory, prompt, n_steps, experiment, plus, "_X2")
        runs.append(run)
        row_names.append(name + ", double")

        if x3:
            run, name = plus_run(directory, prompt, n_steps, experiment, plus, "_X3")
            runs.append(run)
            row_names.append(name +  ", triple")
        if x4:
            run, name = plus_run(directory, prompt, n_steps, experiment, plus, "_X4")
            runs.append(run)
            row_names.append(name + ", quadruple")

    if repeated:
        run, name = plus_run(directory, prompt, n_steps, experiment, plus, "_adjusted_QN_zero_eX2")
        runs.append(run)
        row_names.append(name + ", double+")

        if x3:
            run, name = plus_run(directory, prompt, n_steps, experiment, plus, "_adjusted_QN_zero_eX3")
            runs.append(run)
            row_names.append(name +  ", triple+")

        if x4:
            run, name = plus_run(directory, prompt, n_steps, experiment, plus, "_X4")
            runs.append(run)
            row_names.append(name + ", quadruple+")

    if stochastic_weights:
        # morgana2x100_M4E3_stoWeights_1_STEM3_flex_embedding
        # morgana2x100_M4E3_stoWeights_1_STEM0_flex_embedding
        # morgana2x100_M4E3_stoWeights_1_STEM1_flex_embedding
        # morgana2x100_M4E3_stoWeights_1_STEM2_flex_embedding
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stoWeights_1_STEM0_flex_embedding')
        row_names.append(f"stochastic weights (all)")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stoWeights_1_STEM1_flex_embedding')
        row_names.append(f"stochastic weights (p_emb)")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stoWeights_1_STEM2_flex_embedding')
        row_names.append(f"stochastic weights (t_emb)")
        runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stoWeights_1_STEM3_flex_embedding')
        row_names.append(f"stochastic weights (embs)")
        if adjusted:
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stoWeights_1_STEM0_flex_embedding_adjusted')
            row_names.append(f"adjusted stochastic weights (all)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stoWeights_1_STEM1_flex_embedding_adjusted')
            row_names.append(f"adjusted stochastic weights (p_emb)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stoWeights_1_STEM2_flex_embedding_adjusted')
            row_names.append(f"adjusted stochastic weights (t_emb)")
            runs.append(f'{directory}/{prompt}x{n_steps}_{experiment}_stoWeights_1_STEM3_flex_embedding_adjusted')
            row_names.append(f"adjusted stochastic weights (embs)")



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


def get_results(experiments = ["M4E3","M3E4"], steps = [400,800], prompts = "morgana2", images_per_run = 64,  directory = "images" ,experiment_flags = "ablation", 
                dry = False, quiet = False, fp32_baseline = True, plus = -1, with_margin = False,
                pvals = []):
    
    

    if not isinstance(prompts, list):
        prompts = [prompts]

    if not isinstance(pvals, list):
        pvals = [pvals]

    if not dry:
        have_everything = get_results(experiments=experiments, steps=steps, prompts=prompts, images_per_run=images_per_run, directory=directory, experiment_flags=experiment_flags,
                    dry=True, quiet=True, fp32_baseline=fp32_baseline, plus=plus, pvals=pvals)
        assert have_everything, "Some directories are missing"

    results_dict_ssim = {}
    results_dict_ssim_std = {}
    results_dict_steps = {}


    success = True

    for prompt in prompts:
        for experiment in experiments:
            if len(prompts) > 1:
                prompt_and_experiment = prompt + " " + experiment
            else:
                prompt_and_experiment = experiment
            results_dict_ssim[prompt_and_experiment] = {}
            results_dict_ssim_std[prompt_and_experiment] = {}
            results_dict_steps[prompt_and_experiment] = []
            for n_steps in steps:
                runs, row_names = get_runs_and_names(experiment, n_steps, directory=directory, prompt = prompt, 
                                                     fp32_baseline = fp32_baseline, plus = plus,
                                                     experiment_flags= experiment_flags)
                if check_directories(runs, images_per_run):
                    if not quiet:
                        print(f"Experiment: {experiment}, n_steps: {n_steps}")
                    if dry:
                        continue
                    ssim, std = eval_mse_matrix(runs, images_per_run, fn_desc = "ssim", stop = True)

                    ssim = ssim[0,:]
                    std = std[0,:]

                    for k in range(len(row_names)):
                        if row_names[k] not in results_dict_ssim[prompt_and_experiment]:
                            results_dict_ssim[prompt_and_experiment][row_names[k]] = []
                            results_dict_ssim_std[prompt_and_experiment][row_names[k]] = []
                        results_dict_ssim[prompt_and_experiment][row_names[k]] += [ssim[k]]
                        results_dict_ssim_std[prompt_and_experiment][row_names[k]] += [std[k]]

                    for u,v in pvals:
                        p = eval_mse_pval(runs[0],runs[u], runs[v], images_per_run  ,  fn_desc = "ssim")
                        p_row_name = f"Pval({row_names[u]} > {row_names[v]})"
                        row_names.append(p_row_name)
                        if p_row_name not in results_dict_ssim[prompt_and_experiment]:
                            results_dict_ssim[prompt_and_experiment][p_row_name] = []
                            results_dict_ssim_std[prompt_and_experiment][p_row_name] = []
                        results_dict_ssim[prompt_and_experiment][p_row_name] += [p]
                        results_dict_ssim_std[prompt_and_experiment][p_row_name] += [0]

                    results_dict_steps[prompt_and_experiment] += [n_steps]
                else:
                    success = False

    if dry:
        return success

    df = to_pd(results_dict_ssim, results_dict_steps, row_names)

    df_styled = highlight_max(df, len(pvals))

    if with_margin:
        df_std = to_pd(results_dict_ssim_std, results_dict_steps, row_names)
        df_std_styled = highlight_max(df_std, len(pvals))
        return df_styled, df_std_styled
    else:
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



def plus_run(directory, prompt, n_steps, experiment, plus, ending = ""):
    if plus == -1:
        name = f'{directory}/{prompt}x{n_steps}_{experiment}_Wsr{ending}'
        colname = f"+"
    else:
        name = f'{directory}/{prompt}x{n_steps}_{experiment}_Wsr_M{4+plus}E3{ending}'
        colname = f"+{plus}"

    return name, colname