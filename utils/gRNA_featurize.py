import os
import pandas as pd
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq
import subprocess
import tempfile
import re

def calc_gc(seq):
    """计算GC含量"""
    gc_count = seq.count("G") + seq.count("C")
    return gc_count / len(seq)

def calc_tm(seq):
    """计算熔解温度"""
    return mt.Tm_NN(Seq(seq))

def run_rnafold(seq):
    input_data = seq
    p = subprocess.Popen("RNAfold --noPS", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = p.communicate(input=input_data.encode())
    results = stdout.decode().split("\n")
    mfe = []
    structure = []
    for line in results:
      if "(" in line:
        try:
            eng = float(line.strip().split()[-1].replace("(", "").replace(")", ""))
            mfe.append(eng)
            stru = line.strip().split()[0]
            structure.append(stru)
        except:
            mfe.append(0.0)
            structure.append('')
    return structure,mfe

def detect_stem_loop(dot_bracket):
    """通过正则匹配判断是否存在发卡结构"""
    return 1 if re.search(r"\(+\.{1,4}\)+", dot_bracket) else 0

def extract_features(df, seq_col="gRNA30"):
    """从数据中提取GC/Tm/结构特征"""
    features = {
        "GC_content": [],
        "Tm": [],
        "has_stem_loop": [],
        "min_free_energy": []
    }

    for seq in df[seq_col]:
        sub_seq = seq[4:24]  # 提取中间20bp区域
        gc = calc_gc(sub_seq)
        tm = calc_tm(sub_seq)
        structure, mfe = run_rnafold(sub_seq)
        stem = detect_stem_loop(structure[0]) if structure[0] else 0 ###change accidently

        features["GC_content"].append(gc)
        features["Tm"].append(tm)
        features["has_stem_loop"].append(stem)
        features["min_free_energy"].append(mfe[0] if mfe[0] is not None else 0.0)

    return pd.DataFrame(features)

# 示例使用（读取 gRNA30 列的表格）
# df = pd.read_csv("your_grna_file.tsv", sep="\t")
# feature_df = extract_features(df)
# enhanced_df = pd.concat([df, feature_df], axis=1)
# enhanced_df.to_csv("grna_with_features.tsv", sep="\t", index=False)

