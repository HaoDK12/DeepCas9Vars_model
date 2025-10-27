import os
import glob
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from Bio.Seq import Seq
from typing import List
import re
import argparse
from utils.model_cnn_MMoE2_real_aux_weights import get_model_mmoe #get_model_mmoe_fixedweight
from utils.gRNA_featurize import extract_features

##
cas9_order = ["HiFi-Cas9", "HypaCas9", "LZ3-Cas9", "Sniper-Cas9", "SpCas9",
              "SpCas9-HF1", "SpCas9-NG", "SpCas9-NRCH", "SpCas9-NRRH", "SpCas9-NRTH",
              "SpG", "SpRY", "SuperFi-Cas9", "VRQR", "eSpCas9", "evoCas9", "xCas9"]

cas9_pam_regex = {
    "SpCas9": "^NGG$", "xCas9": "^NG$|^GAA$|^GAT$", "SpCas9-NG": "^NG$", "eSpCas9": "^NGG$",
    "SpCas9-HF1": "^NGG$", "HypaCas9": "^NGG$", "evoCas9": "^NGG$", "Sniper-Cas9": "^NGG$",
    "VRQR": "^NGA$", "SpG": "^NGN$", "SpRY": "^NNN$", "SpCas9-NRRH": "^N[AG][AG][ACT]$",
    "SpCas9-NRTH": "^N[AG]T[ACT]$", "SpCas9-NRCH": "^N[AG]C[ACT]$", "HiFi-Cas9": "^NGG$",
    "LZ3-Cas9": "^NGG$", "SuperFi-Cas9": "^NGG$"
}
test_dir = os.path.join(os.getcwd(),'utils/0.001')
# ---------- main function ---------------- #

def seq_to_onehot(seq):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq])

def reverse_complement(seq):
    return str(Seq(seq).reverse_complement())

def generate_30mers_and_rc(input_seq):
    input_seq = input_seq.upper()
    mers = [input_seq[i:i+30] for i in range(len(input_seq) - 29)]
    rc_mers = [reverse_complement(m) for m in mers]
    strands = ['+'] * len(mers) + ['-'] * len(rc_mers)
    return mers + rc_mers, strands

def compute_gc(seq):
    gc_count = sum(1 for b in seq if b in "GC")
    return round(gc_count / len(seq), 4)

def predict_with_models(input_seq: str, test_dir: str, selected_cas9s: List[str]):
    #
    mers, strands = generate_30mers_and_rc(input_seq)

    df = pd.DataFrame({'used_30mer': mers,'strand': strands,'input_seq': [input_seq] * len(mers),'GC_content': [compute_gc(m[4:24]) for m in mers]})
    
    #
    X_seq = np.array([seq_to_onehot(seq) for seq in df['used_30mer']])
    #data_aux = {'GC_content': [0.65]*len(df),
    #        'Tm': [60.044643547542876]*len(df),
    #        'has_stem_loop': [1]*len(df),
    #        'min_free_energy':[-3.9]*len(df)}
    #X_aux = pd.DataFrame(data_aux).values
    X_aux = extract_features(df.reset_index(), seq_col='used_30mer')
    X_origin1 = np.ones((len(df), 1))
    X_origin0 = np.zeros((len(df), 1))


    #
    h5_files = sorted(glob.glob(os.path.join(test_dir, "*.h5")))
    all_predictions1 = []
    all_predictions0 = []

    for h5_file in h5_files:
        model = get_model_mmoe(max_len_en=30, max_dep=4, n_variants=17)
        model.load_weights(h5_file)

        pred1 = model.predict([X_seq, X_aux, X_origin1], verbose=0)
        all_predictions1.append(pred1.reshape(-1, 17))
    
        pred0 = model.predict([X_seq, X_aux, X_origin0], verbose=0)
        all_predictions0.append(pred0.reshape(-1, 17))

    all_predictions1 = np.concatenate(all_predictions1, axis=1)
    all_predictions0 = np.concatenate(all_predictions0, axis=1)
    df_results1 = pd.DataFrame(all_predictions1)
    df_results0 = pd.DataFrame(all_predictions0)

    pred_df = pd.DataFrame(columns=cas9_order)
    pred_df['HiFi-Cas9'] = 1*df_results1.loc[:,np.arange(1, 85, 17)].mean(axis=1)
    pred_df['HypaCas9'] = 0.722*df_results0.loc[:,np.arange(2, 85, 17)].mean(axis=1)+0.278*df_results1.loc[:,np.arange(2, 85, 17)].mean(axis=1)
    pred_df['LZ3-Cas9'] = 1*df_results1.loc[:,np.arange(3, 85, 17)].mean(axis=1)
    pred_df['Sniper-Cas9'] = 0.719*df_results0.loc[:,np.arange(4, 85, 17)].mean(axis=1)+0.281*df_results1.loc[:,np.arange(4, 85, 17)].mean(axis=1)
    pred_df['SpCas9'] = 0.722*df_results0.loc[:,np.arange(5, 85, 17)].mean(axis=1)+0.278*df_results1.loc[:,np.arange(5, 85, 17)].mean(axis=1)
    pred_df['SpCas9-HF1'] = 0.718*df_results0.loc[:,np.arange(6, 85, 17)].mean(axis=1)+0.282*df_results1.loc[:,np.arange(6, 85, 17)].mean(axis=1)
    pred_df['SpCas9-NG'] = 0.718*df_results0.loc[:,np.arange(7, 85, 17)].mean(axis=1)+0.282*df_results1.loc[:,np.arange(7, 85, 17)].mean(axis=1)

    pred_df['SpCas9-NRCH'] = 1*df_results0.loc[:,np.arange(8, 85, 17)].mean(axis=1)
    pred_df['SpCas9-NRRH'] = 1*df_results0.loc[:,np.arange(9, 85, 17)].mean(axis=1)
    pred_df['SpCas9-NRTH'] = 1*df_results0.loc[:,np.arange(10, 85, 17)].mean(axis=1)

    pred_df['SpG'] = 0.718*df_results0.loc[:,np.arange(11, 85, 17)].mean(axis=1)+0.282*df_results1.loc[:,np.arange(11, 85, 17)].mean(axis=1)
    pred_df['SpRY'] = 1*df_results0.loc[:,np.arange(12, 85, 17)].mean(axis=1)

    pred_df['SuperFi-Cas9'] = 1*df_results1.loc[:,np.arange(12, 85, 17)].mean(axis=1)
    pred_df['VRQR'] = 1*df_results0.loc[:,np.arange(13, 85, 17)].mean(axis=1)

    pred_df['eSpCas9'] = 0.719*df_results0.loc[:,np.arange(14, 85, 17)].mean(axis=1)+0.281*df_results1.loc[:,np.arange(14, 85, 17)].mean(axis=1)
    pred_df['evoCas9'] = 0.720*df_results0.loc[:,np.arange(15, 85, 17)].mean(axis=1)+0.280*df_results1.loc[:,np.arange(15, 85, 17)].mean(axis=1)
    pred_df['xCas9'] = 0.723*df_results0.loc[:,np.arange(16, 85, 17)].mean(axis=1)+0.277*df_results1.loc[:,np.arange(16, 85, 17)].mean(axis=1)

    #
    df_full = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    for cas9 in cas9_order: 
        if cas9 not in selected_cas9s:
            df_full.drop(columns=[cas9], inplace=True)
            continue

        pam_pattern = cas9_pam_regex.get(cas9)
        if pam_pattern:
            pam_subpatterns = [p.replace("^", "").replace("$", "") for p in pam_pattern.split("|")]
            matched_mask = pd.Series([False] * len(df_full), index=df_full.index)
            pam_values = [None] * len(df_full)

            for sub_pam in pam_subpatterns:
                if matched_mask.all():
                    break  #

                pam_len = len(re.sub(r"\[[^]]+\]", "A", sub_pam.replace("N", "A")))
                pam_regex = sub_pam.replace("N", "[ATCG]")
                pam_substr = df_full.loc[~matched_mask, 'used_30mer'].str[24: 24 + pam_len]
                match = pam_substr.str.fullmatch(pam_regex)

                matched_indices = match[match].index
                matched_mask.loc[matched_indices] = True
                for idx in matched_indices:
                    pam_values[idx] = df_full.loc[idx, 'used_30mer'][24: 24 + pam_len]

            # 
            df_full[f'pam_{cas9}'] = pam_values

            # 
            df_full.loc[~matched_mask, cas9] = np.nan

    df_full["id"] = df_full["strand"].map({"+": "p_", "-": "n_"}) + (df_full.index + 1).astype(str)
    # 
    cas9_cols = [col for col in df_full.columns if col not in ["used_30mer", "strand", "input_seq", "GC_content", "id"] and not col.startswith("pam_")]

    df_full["gRNA"] = df_full["used_30mer"].str[4:4+20]
    # 
    long_df = pd.melt(
        df_full,
        id_vars=["id", "gRNA", "strand", "GC_content"],
        value_vars=cas9_cols,
        var_name="Cas9",
        value_name="Efficiency"
    )

    #
    long_df["PAM"] = long_df.apply(lambda row: df_full.loc[df_full["id"] == row["id"], f"pam_{row['Cas9']}"].values[0], axis=1)

    long_df = long_df.dropna(subset=["Efficiency"])
    return long_df

def main():
    parser = argparse.ArgumentParser(description="Predict gRNA efficiency with DeepCas9Vars model.")
    parser.add_argument("--input_seq", type=str, required=True, help="Input DNA sequence.")
    parser.add_argument("--selected_cas9s", nargs='+', required=True, help="SpCas9 variant name (e.g., SpCas9, HiFi-Cas9, etc.)")
    parser.add_argument("--out_path", type=str, default="output.tsv", help="Output file path.")
    
    args = parser.parse_args()

    try:
        results = predict_with_models(args.input_seq, test_dir, args.selected_cas9s)
        with open(args.out_path, 'w') as f:
            header = "ID\tTarget\tStrand\tGC%\tCas9_names\tDeepCas9Vars_score\tPAM"
            f.write(header + "\n")
            for row in results.itertuples(index=False):
                f.write(f"{row.id}\t{row.gRNA}\t{row.strand}\t"
                        f"{row.GC_content:.1f}\t{row.Cas9}\t{row.Efficiency:.1f}\t{row.PAM}\n")

        print(f"^o^ Great! Job finished and results successfully saved, Check it in the {args.out_path} file ! Bye!")
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()