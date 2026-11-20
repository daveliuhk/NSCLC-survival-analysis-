import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
def read (link):
    data = pd.read_csv(
        link,
        sep="\t"
    )
    return data

msk = read("nsclc_pd1_msk_2018_clinical_data.tsv")
tcga = read("luad_tcga_pan_can_atlas_2018_clinical_data.tsv")
mskcc = read("bm_nsclc_mskcc_2023_clinical_data.tsv")
tracerx = read("nsclc_tracerx_2017_clinical_data.tsv")
lusc = read("lusc_tcga_pan_can_atlas_2018_clinical_data.tsv")

def drop_duplicated(*dfs):
    clean_list = []

    for df in dfs:
        df = df.drop_duplicates()
        clean_list.append(df)

    return clean_list

msk, tcga, mskcc, tracerx, lusc = drop_duplicated(msk, tcga, mskcc, tracerx, lusc)

def investigate (*names):
    investigation = []

    for name in names:
        result = pd.DataFrame({
            "dytpe" : name.dtypes,
            "unique_number" : name.nunique(dropna=True),
            "missing" : name.isna().sum()
        })

        investigation.append(result)

    return investigation
def clean_columns(*names):
    clean_columns = []

    for name in names:
        name.columns = name.columns.str.strip().str.lower().str.replace(" ", "_")

        clean_columns.append(name)

    return clean_columns

msk, tcga, mskcc, tracerx, lusc = clean_columns(msk, tcga, mskcc, tracerx, lusc)

msk_report, tcga_report, mskcc_report, tracerx_report, lusc_report = investigate(msk,tcga,mskcc, tracerx, lusc)

duplicate_msk = msk[msk["patient_id"].isin(mskcc["patient_id"])]
duplicate_tcga = tcga[tcga["patient_id"].isin(lusc["patient_id"])]
duplicate_tracerx = tracerx[tracerx["patient_id"].isin(lusc["patient_id"])]

combined_msk_mskcc = pd.concat(
    [msk,mskcc],
    ignore_index=True
)

combined_msk_mskcc = combined_msk_mskcc.groupby(
    "patient_id",
    as_index=False,
    sort=False
).first()

tracerx_duplicate = tracerx[tracerx["patient_id"].duplicated(keep=False)]

tracerx_duplicate_report, = investigate(tracerx_duplicate)

tracerx_first = tracerx.groupby(
    "patient_id",
    as_index=False,
    sort=False
).first()

tracerx_first_report, = investigate(tracerx_first)

combined_all = pd.concat(
    [combined_msk_mskcc, tcga, tracerx_first, lusc],
    ignore_index = True
)

combined_all["progression_free_status"] = combined_all["progression_free_status"].str.strip().str.lower()


progress = ["1:progression", "1:progressed"]
not_progress = ["0:not progressed", "0:censored"]
combined_all["pfs_status"] = combined_all["progression_free_status"].isin(progress).astype("boolean")
unknown = ~combined_all["progression_free_status"].isin(progress + not_progress)
combined_all.loc[unknown, "pfs_status"] = pd.NA

mt_surv = combined_all[[
    "patient_id",
    "mutation_count",
    "progress_free_survival_(months)",
    "sex",
    "smoker",
    "tmb_(nonsynonymous)",
    "nsclc_subtype",
    "smoking_status",
    "tumor_type",
    "histology",
    "smoking_history",
    "pfs_status"
]].copy()

smoker = ["Ever", "Former", "Current", "Ex-Smoker", "Recent Ex-Smoker", "Current Smoker"]
non_smoker = ["Never", "Never Smoked"]
col = ["smoker", "smoking_status", "smoking_history"]
mt_surv["smoking"] = combined_all[col].isin(smoker).any(axis=1).astype("boolean")
unknown = ~combined_all[col].isin(smoker + non_smoker).any(axis=1)
mt_surv.loc[unknown, "smoking"] = pd.NA


adeno = ["Adenocarcinoma", "Lung Adenocarcinoma (NOS)", "Lung Adenocarcinoma, Mixed Subtype", "Lung Adenocarcinoma", "Lung Papillary Adenocarcinoma", "Lung Acinar Adenocarcinoma", "Lung Solid Pattern Predominant Adenocarcinoma", "Lung Bronchioloalveolar Carcinoma, Mucinous", "Lung Micropapillary Adenocarcinoma", "Lung Mucinous Adenocarcinoma", "Lung Signet Ring Adenocarcinoma", "Invasive adenocarcinoma", "Adenosquamous carcinoma"]
scc = ["Squamous cell carcinoma", "Lung Squamous Cell Carcinoma (NOS)", "Lung Basaloid Squamous Cell Carcinoma", "Lung Papillary Squamous Cell Caricnoma", "Lung Squamous Cell Carcinoma", "Squamous cell carcinoma"]
col_his = ["tumor_type", "histology", "nsclc_subtype"]
mt_surv["subtype"]= pd.Series(pd.NA, index=combined_all.index, dtype="string")
mt_surv.loc[combined_all[col_his].isin(adeno).any(axis=1), "subtype"] = "adeno"
mt_surv.loc[combined_all[col_his].isin(scc).any(axis=1), "subtype"] = "scc"
other = ~combined_all[col_his].isin(adeno + scc).any(axis=1)
mt_surv.loc[other, "subtype"] = "other"

mt_surv = mt_surv.rename(columns={
    "tmb_(nonsynonymous)":"tmb",
    "progress_free_survival_(months)":"pfs"
})

mt_surv["tmb"]=pd.to_numeric(mt_surv["tmb"], errors="coerce")
mt_surv["pfs"]=pd.to_numeric(mt_surv["pfs"], errors="coerce")
mt_surv["event"]=mt_surv["pfs_status"].astype("Int64")

df_surv = mt_surv.copy()
df_surv = df_surv.dropna(
    subset=["tmb", "pfs", "event"]
)

median_tmb = df_surv["tmb"].median()

group = []
for value in df_surv["tmb"]:
    if value > median_tmb:
        group.append("high")
    else:
        group.append("low")

df_surv["tmb_group"] = group

kaplan = KaplanMeierFitter()

plt.figure(figsize=(9, 6))
for value in ["high", "low"]:
    subset=df_surv[df_surv["tmb_group"] == value]
    kaplan.fit(
        durations=subset["pfs"],
        event_observed=subset["event"],
        label=value
    )
    kaplan.plot_survival_function(ci_show=True)
    print(value, "Median PFS:", kaplan.median_survival_time_)

high = df_surv[df_surv["tmb_group"] == "high"]
low = df_surv[df_surv["tmb_group"] == "low"]

result = logrank_test(
    high["pfs"],
    low["pfs"],
    event_observed_A=high["event"],
    event_observed_B=low["event"]
)

print("p-value", result.p_value)

plt.title("Kaplan-Meier: PFS by TMB")
plt.xlabel("PFS (months)")
plt.ylabel("Progression-free probability")
plt.ylim(0, 1)
plt.grid(alpha=0.3)

plt.text(
    0.65, 0.85,
    f"log-rank p: {result.p_value:.3f}",
    transform=plt.gca().transAxes
)

plt.savefig("km_tmb.png", dpi=300, bbox_inches="tight")
plt.show()


cox_d = df_surv[["pfs", "event", "tmb"]].dropna()
cox = CoxPHFitter()
cox.fit(
    cox_d,
    duration_col="pfs",
    event_col="event"
)
cox.print_summary()
