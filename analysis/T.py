import pandas as pd
import scipy.stats as stats

# まずはpickleファイルを読み込む
df=pd.read_pickle("../work/input/LSWMD_25519.pkl")

# 'Loc'タイプのfailureTypeに対してのみdieSizeを抽出
loc_die_sizes = df[df['failureType'] == 'Loc']['dieSize']

# 'Loc'タイプ以外のfailureTypeのdieSizeを抽出
non_loc_die_sizes = df[df['failureType'] != 'Loc']['dieSize']

# 両者の平均を計算
loc_mean = loc_die_sizes.mean()
non_loc_mean = non_loc_die_sizes.mean()

# T検定を使用して平均に統計的に有意な差があるかを検証
t_stat, p_value = stats.ttest_ind(loc_die_sizes, non_loc_die_sizes)

loc_mean, non_loc_mean, t_stat, p_value

# T検定の結果を出力
print(f"Loc平均: {loc_mean}")
print(f"非Loc平均: {non_loc_mean}")
print(f"T統計量: {t_stat}")
print(f"P値: {p_value}")

# P値が0.05以下であれば、統計的に有意な差があると判断
if p_value < 0.05:
    print("Locと非LocのdieSizeには統計的に有意な差があります。")
else:
    print("Locと非LocのdieSizeに統計的に有意な差はありません。")
