
######### PREPARACIÓN DE DATOS #########

# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================
project_root = Path.cwd().parent
inputs_dir = project_root / "Data" / "inputs" / "dataset_covid.csv"
outputs_dir = project_root / "Data" / "outputs"
dir_plots_sponsor_key = outputs_dir / "plots" / "sponsor" / "key_relations"

# Se crean directorios de salida
dir_raw = outputs_dir / "raw"
dir_filtered = outputs_dir / "filtered"
dir_filtered_weekly = outputs_dir / "filtered_weekly"
tables_clu = outputs_dir / "tables" / "clustering"

for d in [dir_raw, dir_filtered, dir_filtered_weekly, tables_clu]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. CARGA Y DIVISIÓN INICIAL DE DATOS
# =============================================================================
print("Cargando y dividiendo los datos iniciales...")
df = pd.read_csv(inputs_dir, sep=",")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-12-31")]

code_str = df["code"].astype("string")
mask_countries = code_str.str.fullmatch(r"[A-Za-z]{3}", na=False)
mask_regions = code_str.str.fullmatch(r"OWID_[A-Za-z]{3}", na=False)

df_countries = df[mask_countries].copy()
df_regions = df[mask_regions].copy()
df_discard = df[~mask_regions & ~mask_countries].copy()


df_countries.to_csv(dir_raw / "df_countries.csv", sep=";", index=False)
df_regions.to_csv(dir_raw / "df_regions.csv", sep=";", index=False)
df_discard.to_csv(dir_raw / "df_discard.csv", sep=";", index=False)
print("Datos iniciales guardados en 'Data/outputs/raw'.")

# =============================================================================
# 3. ANÁLISIS DE VALORES NULOS (NA)
# =============================================================================
print("Generando gráfico de valores nulos...")
na_pct_graph = (df_countries.isna().mean().sort_values(ascending=False) * 100).reset_index()
na_pct_graph.columns = ["Variable", "Porcentaje_NA"]

plt.figure(figsize=(14, 12))
sns.barplot(data=na_pct_graph, x="Porcentaje_NA", y="Variable", color="steelblue")
plt.axvline(50, color="red", linestyle="--", label="Umbral 50%")
plt.title("Porcentaje de valores NA por variable en df_countries")
plt.xlabel("% de valores NA")
plt.ylabel("Variable")
plt.legend()
plt.tight_layout()
plt.savefig(dir_plots_sponsor_key / "valores_NA.png", dpi=200)
plt.show()

# =============================================================================
# 4. AGREGACIÓN SEMANAL PARA GRÁFICOS
# =============================================================================
print("Agregando datos a nivel semanal para gráficos...")
graph_vars = [
    "total_cases", "total_deaths", "total_cases_per_million", "total_deaths_per_million",
    "new_cases_per_million", "new_deaths_per_million", "new_cases_smoothed", "new_deaths_smoothed",
    "new_cases_smoothed_per_million", "new_deaths_smoothed_per_million", "population", "median_age",
    "gdp_per_capita", "diabetes_prevalence", "extreme_poverty", "hospital_beds_per_thousand",
    "stringency_index", "reproduction_rate", "people_vaccinated_per_hundred"
]
id_vars = ["country", "code", "continent", "date"]

df_countries_graph = df_countries[id_vars + [c for c in graph_vars if c in df_countries.columns]].copy()
df_regions_graph = df_regions[id_vars + [c for c in graph_vars if c in df_regions.columns]].copy()

def aggregate_to_weekly(df_in: pd.DataFrame, group_by_cols: list) -> pd.DataFrame:
    dfw = df_in.copy()
    dfw["week_start"] = dfw["date"] - pd.to_timedelta(dfw["date"].dt.dayofweek, unit="D")
    
    num_cols = [c for c in graph_vars if c in dfw.columns]
    sum_cols = [c for c in num_cols if c.startswith("new_")]
    max_cols = [c for c in num_cols if c.startswith("total_")] + ["people_vaccinated_per_hundred"]
    mean_cols = [c for c in num_cols if c in {"stringency_index", "reproduction_rate"}]
    static_cols = [
        "population", "median_age", "gdp_per_capita", "diabetes_prevalence",
        "extreme_poverty", "hospital_beds_per_thousand"
    ]
    
    agg_dict = {}
    for c in sum_cols: agg_dict[c] = "mean" if "smoothed" in c else "sum"
    for c in max_cols: agg_dict[c] = "max"
    for c in mean_cols: agg_dict[c] = "mean"
    for c in static_cols: agg_dict[c] = "first"
        
    df_weekly = dfw.groupby(group_by_cols + ["week_start"], as_index=False).agg(agg_dict)
    return df_weekly[df_weekly['week_start'].dt.year >= 2020]


df_regions_graph_renamed = df_regions_graph.drop(columns=['continent'])
df_regions_graph_renamed = df_regions_graph_renamed.rename(columns={'country': 'continent'})
df_countries_graph_weekly = aggregate_to_weekly(df_countries_graph, ["country", "continent"])
df_regions_graph_weekly = aggregate_to_weekly(df_regions_graph_renamed, ["continent"])

df_countries_graph_weekly.to_csv(dir_filtered_weekly / "df_countries_graph_weekly.csv", sep=";", index=False)
df_regions_graph_weekly.to_csv(dir_filtered_weekly / "df_regions_graph_weekly.csv", sep=";", index=False)
print("Ficheros semanales para gráficos guardados.")

# =============================================================================
# 5. CONSTRUCCIÓN DE DATASETS PARA MODELADO
# =============================================================================
print("Construyendo datasets robustos para modelado...")
id_vars = ["country", "code", "continent", "date"]
model_vars = [
    "total_cases_per_million", "total_deaths_per_million", "new_cases_per_million", "new_deaths_per_million",
    "reproduction_rate", "stringency_index", "people_vaccinated_per_hundred", "median_age", "gdp_per_capita",
    "diabetes_prevalence", "extreme_poverty", "hospital_beds_per_thousand"
]

dfm = df_countries[id_vars + [c for c in model_vars if c in df_countries.columns]].copy()
dfm = dfm.sort_values(["country", "date"])

# --- Imputación por Propagación hacia Adelante (Forward Fill) ---
ffill_cols = [
    "total_cases_per_million", "total_deaths_per_million", "reproduction_rate",
    "stringency_index", "people_vaccinated_per_hundred"
]
print("Aplicando imputación forward fill a variables de serie temporal...")
for c in [x for x in ffill_cols if x in dfm.columns]:
    dfm[c] = dfm.groupby("country")[c].ffill()

# --- Imputación Jerárquica para Variables Estáticas ---
statics = ["median_age", "gdp_per_capita", "diabetes_prevalence", "extreme_poverty", "hospital_beds_per_thousand"]
print("Aplicando imputación jerárquica a variables estáticas...")
for c in [x for x in statics if x in dfm.columns]:
    med_cont = dfm.groupby("continent")[c].transform("median")
    dfm[c] = dfm[c].fillna(med_cont).fillna(dfm[c].median())

dfm = dfm.dropna(subset=["country", "code", "continent", "date"])

last_by_cty = dfm.groupby("country")["date"].max()
good_countries = last_by_cty[last_by_cty >= pd.Timestamp("2024-10-01")].index
df_countries_model = dfm[dfm["country"].isin(good_countries)].copy()

daily_out = dir_filtered / "df_countries_model_imputed.csv"
df_countries_model.to_csv(daily_out, sep=";", index=False)
print("Diario canónico limpio guardado:", daily_out.name)


# --- Se guarda SEMANAL ROBUSTO para modelos ---
flow_cols = ["new_cases_per_million", "new_deaths_per_million"]
stock_cols = ["total_cases_per_million", "total_deaths_per_million"]
index_cols = ["reproduction_rate", "stringency_index"]
level_cols = ["people_vaccinated_per_hundred"]

def weekly_robust_fixed(g):
    idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    g = g.set_index("date").reindex(idx)
    for c in ["country", "code", "continent"] + statics:
        if c in g: g[c] = g[c].ffill().bfill()
    out = pd.DataFrame(index=idx)
    MIN_DAYS = 5
    for c in flow_cols:
        if c in g:
            sum_raw = g[c].resample("W-MON").sum(min_count=1)
            days_av = g[c].resample("W-MON").count()
            val = sum_raw.where(days_av >= MIN_DAYS) * (7 / days_av.clip(lower=1))
            out[c] = val
    for c in stock_cols + level_cols:
        if c in g: out[c] = g[c].resample("W-MON").max()
    for c in index_cols:
        if c in g:
            mean_w = g[c].resample("W-MON").mean()
            cnt_w = g[c].resample("W-MON").count()
            out[c] = mean_w.where(cnt_w >= MIN_DAYS)
    for c in ["country", "code", "continent"] + statics:
        if c in g: out[c] = g[c].resample("W-MON").last()
    if flow_cols:
        out = out[out[flow_cols].notna().any(axis=1)]
    return out.reset_index().rename(columns={"index": "week"})

weekly = df_countries_model.groupby("country", as_index=False, group_keys=False).apply(weekly_robust_fixed).reset_index(drop=True)
weekly_out = dir_filtered_weekly / "weekly_model_ready.csv"
weekly.to_csv(weekly_out, sep=";", index=False)
print("Semanal robusto guardado:", weekly_out.name)

# --- Se guarda SNAPSHOT para clustering ---
snapshot = df_countries_model.sort_values(["country", "date"]).groupby("country", as_index=False).tail(1).reset_index(drop=True)

# Se eliminan las columnas 'new_*' que no se usan en el clustering
cols_to_drop = ['new_cases_per_million', 'new_deaths_per_million']
snapshot_cleaned = snapshot.drop(columns=cols_to_drop, errors='ignore')

snap_out = tables_clu / "snapshot_2024_12_31_for_clustering.csv"
snapshot_cleaned.to_csv(snap_out, sep=";", index=False)
print("Snapshot clustering (limpio) guardado:", snap_out.name)

print("\n--- Proceso de preparación de datos finalizado. ---")