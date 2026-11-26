#import basic packages
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

#basic setting of pandas
pd.set_option("display.max_columns", None, "display.max_rows", 100)

#useful function for investigation later
def shape(*dfs):
    list = []
    for df in dfs:
        list.append(df.shape)
    return print(list)

def value_count(*dfs):
    list = []
    for df in dfs:
        list.append(df.value_counts())
    return print(list)

#import raw data files
def read (*files):
    df = []
    for file in files:
        name = pd.read_csv(
            file,
            sep="\t"
        )
        df.append(name)
    return df

msk, tcga, tracer, mskcc, luad = (read(
    "nsclc_pd1_msk_2018_clinical_data.tsv",
    "lusc_tcga_pan_can_atlas_2018_clinical_data.tsv",
    "nsclc_tracerx_2017_clinical_data.tsv",
    "bm_nsclc_mskcc_2023_clinical_data.tsv",
    "luad_tcga_pan_can_atlas_2018_clinical_data.tsv"
))

#clean the column names
def clean_columns (*dfs):
    clean =[]
    for df in dfs:
        df.columns = (df.columns
                      .str.strip()
                      .str.lower()
                      .str.replace(" ", "_")
                      )
        clean.append(df)
    return clean

clean_columns(msk, tcga, tracer, mskcc, luad)

#combine all database together

def add_tracker (**dfs):
    tracker = []
    for name,df in dfs.items():
        df["tracker"]=name
        tracker.append(df)
    return tracker

(msk,
 tcga,
 tracer,
 mskcc,
 luad) = add_tracker(
    msk=msk,
    tcga=tcga,
    tracer=tracer,
    mskcc=mskcc,
    luad=luad
)

#initial investigation of the data
def audit (*dfs):
    audit_df = []
    for df in dfs:
        temp = pd.DataFrame({
            "dytpe": df.dtypes,
            "unique_n": df.nunique(dropna=True),
            "missing_n": df.isna().sum()
        })
        audit_df.append(temp)
    return audit_df

(msk_audit,
 tcga_audit,
 tracer_audit,
 mskcc_audit,
 luad_audit) = audit(
    msk,
    tcga,
    tracer,
    mskcc,
    luad)

#we quickily identified the columns of interest: patient_id, sample_id, cancer_type_detailed, progress_free_survival_(months), progression_free_status, sex, smoker, tmb_(nonsynonymous, tracker)
columns = [
    "patient_id",
    "sample_id",
    "cancer_type_detailed",
    "progress_free_survival_(months)",
    "progression_free_status",
    "sex",
    "smoker",
    "tmb_(nonsynonymous)",
    "tracker"
]

#combining all dataset
combined = pd.concat(
    [msk, tcga, tracer, mskcc, luad],
    ignore_index=True
)
combined = combined[columns]

#investigate the duplicated records
combined_duplicated = combined[combined["patient_id"].duplicated(keep=False)]
combined = combined.drop_duplicates()
combined_duplicated_audit = audit(combined_duplicated,)

#627 records are in combined_duplicated, by looking at combined_duplicated_audit, we knows that this is due to patients having multiple biopsy, the final survival analysis should be based on number of patients, not number of biopsy samples
#we should output the excel file and look at the source of the data one by one and decide which biopsy to be used, but as these are external data, we don't know the details of the trial, therefore we only choose the first record here to simplify the steps
combined = combined.groupby(
    "patient_id",
    as_index=False,
    sort=False
).first()
combined_duplicated = combined[combined["patient_id"].duplicated(keep=False)]
#now there are no more duplicated and 1617 records in total left

#we need to re-investigate the updated data
combined_audit = audit(combined,)

#few columns need further investigation and may need some cleaning: cancer_type_detailed, progression_free_status
value_count(combined["progression_free_status"], combined["cancer_type_detailed"])

#cleaning these two columns
combined["progression_free_status"] = combined["progression_free_status"].str.strip().str.lower()
progress = ["1:progression", "1:progressed"]
not_progress = ["0:not progressed", "0:censored"]
combined["pfs"] = combined["progression_free_status"].isin(progress).astype("boolean")
unknown = ~combined["progression_free_status"].isin(progress + not_progress)
combined.loc[unknown, "pfs"] = pd.NA

combined["cancer_type_detailed"] = combined["cancer_type_detailed"].str.strip().str.lower()
adeno = ["lung adenocarcinoma"]
scc = ["lung squamous cell carcinoma"]
combined["subtype"]= pd.Series(pd.NA, index=combined.index, dtype="string")
combined.loc[combined["cancer_type_detailed"].isin(adeno), "subtype"] = "adeno"
combined.loc[combined["cancer_type_detailed"].isin(scc), "subtype"] = "scc"
other = ~combined["cancer_type_detailed"].isin(adeno + scc)
combined.loc[other, "subtype"] = "other"

combined_audit = audit(combined,)

#now the data is ready and clean, we can perform the analysis
#kaplan meier analysis on tmb and pfs
combined_tmb = combined.dropna(
    subset=["tmb_(nonsynonymous)", "pfs", "progress_free_survival_(months)"]
).copy()

#spliting the group into high tmb and low tmb groups
median_tmb = combined_tmb["tmb_(nonsynonymous)"].median()

group = []
for value in combined_tmb["tmb_(nonsynonymous)"]:
    if value > median_tmb:
        group.append("high")
    else:
        group.append("low")

combined_tmb["tmb_group"] = group

#KaplanMeier analysis
kaplan = KaplanMeierFitter()

plt.figure(figsize=(9, 6))
for value in ["high", "low"]:
    subset=combined_tmb[combined_tmb["tmb_group"] == value]
    kaplan.fit(
        durations=subset["progress_free_survival_(months)"],
        event_observed=subset["pfs"],
        label=value
    )
    kaplan.plot_survival_function(ci_show=True)
    print(value, "Median PFS:", kaplan.median_survival_time_)

high = combined_tmb[combined_tmb["tmb_group"] == "high"]
low = combined_tmb[combined_tmb["tmb_group"] == "low"]

result = logrank_test(
    high["progress_free_survival_(months)"],
    low["progress_free_survival_(months)"],
    event_observed_A=high["pfs"],
    event_observed_B=low["pfs"]
)

print("p-value", result.p_value)

#plotting the curves
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

#saving the plot and show it here
plt.savefig("km_tmb.png", dpi=300, bbox_inches="tight")

#univariable cox regression analysis
cox = CoxPHFitter()

cox_tmb = combined_tmb[["pfs", "progress_free_survival_(months)", "tmb_(nonsynonymous)"]]

cox.fit(
    cox_tmb,
    duration_col="progress_free_survival_(months)",
    event_col="pfs"
)

cox.print_summary()

#multivariable cox regression analysis
multi_tmb = combined_tmb[["pfs",
                          "progress_free_survival_(months)",
                          "tmb_(nonsynonymous)",
                          "subtype",
                          "sex"
                          ]].dropna().copy()

multi_tmb["sex"] = multi_tmb["sex"].map({
    "Male": True,
    "Female": False
})
multi_tmb = pd.get_dummies(
    multi_tmb,
    columns = ["subtype"],
    drop_first = True,
    dtype = int
)

cox.fit(
    multi_tmb,
    duration_col="progress_free_survival_(months)",
    event_col="pfs"
)

cox.print_summary()

#conclusion:
#Total 1224 patients included with 542 PFS events
#Univariable Cox regression showed no significant association between TMB and PFS (HR = 1.00, p = 0.63)
#this finding remains non-significant after adjustment for sex and histological subtype (adjusted HR = 0.99, p = 0.28)
#Histology subtype significantly associated with PFS, potential explanation could be subtype "other" includes patient with missing histological info, patients with rapidly progressive disease may be less likely to undergo successful biopsy, patient without confirmed or other subtype of histology inherently do worse. Interstingly in this analysis, SCC patient do better than Adenocarcinoma patients, which is contrary to clinical observation.