
######### CLUSTERING #########

# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import geopandas as gpd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# =============================================================================
# 1. CONFIGURACIÓN DE RUTAS
# =============================================================================

project_root = Path.cwd().parent
outputs_dir = project_root / "Data" / "outputs"
tables_clu = outputs_dir / "tables" / "clustering"
dir_plots_analyst_clu = outputs_dir / "plots" / "analyst" / "clustering"
dir_plots_maps_static = outputs_dir / "plots" / "maps" / "static"
geo_path = project_root / "Data" / "inputs" / "countries.geo.json"

for d in [tables_clu, dir_plots_analyst_clu, dir_plots_maps_static]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS PARA CLUSTERING
# =============================================================================
print("Cargando y preparando datos del snapshot para clustering...")
try:
    snapshot = pd.read_csv(tables_clu / "snapshot_2024_12_31_for_clustering.csv", sep=";")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo 'snapshot_2024_12_31_for_clustering.csv'. Asegúrate de haber ejecutado '01_data_preparation.py' primero.")
    exit()

model_vars = [
    "total_cases_per_million", 
    "total_deaths_per_million",
    "people_vaccinated_per_hundred", 
    "median_age", 
    "gdp_per_capita",
    "diabetes_prevalence", 
    "extreme_poverty", 
    "hospital_beds_per_thousand",
    "stringency_index", 
    "reproduction_rate"
]
id_cols = ["country", "code", "continent"]
X0 = snapshot[id_cols + [c for c in model_vars if c in snapshot.columns]].copy()

# Imputación jerárquica (continente -> global) y estandarización
numeric_cols = [c for c in model_vars if c in X0.columns]
for c in numeric_cols:
    if X0[c].isnull().any():
        med_cont = X0.groupby("continent")[c].transform("median")
        X0[c] = X0[c].fillna(med_cont).fillna(X0[c].median())

# Imputador simple como red de seguridad final
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X0[numeric_cols])

# Se estandarizan los datos, ya que K-Means es sensible a la escala
scaler = StandardScaler()
Xs = scaler.fit_transform(X_imp)

# =============================================================================
# 3. BÚSQUEDA DEL NÚMERO ÓPTIMO DE CLUSTERS (k)
# =============================================================================
print("Buscando el número óptimo de clusters (k) usando Elbow y Silhouette...")
ks = range(2, 9)
inertia, silh = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=50, random_state=27)
    labels_temp = km.fit_predict(Xs)
    inertia.append(km.inertia_)
    silh.append(silhouette_score(Xs, labels_temp))

best_k = ks[int(np.argmax(silh))]
print(f"Mejor k por Silhouette: {best_k} (score={max(silh):.3f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(list(ks), inertia, marker="o", linestyle='-')
ax[0].set_title("Método del Codo")
ax[0].set_xlabel("Número de clusters (k)")
ax[0].set_ylabel("Inertia")
ax[0].grid(True, alpha=0.3)

ax[1].plot(list(ks), silh, marker="o", linestyle='-')
ax[1].set_title("Método de la Silueta")
ax[1].set_xlabel("Número de clusters (k)")
ax[1].set_ylabel("Silhouette Score")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(dir_plots_analyst_clu / "kmeans_elbow_silhouette.png", dpi=200)
plt.show()

# =============================================================================
# 4. AJUSTE FINAL DEL MODELO Y GUARDADO DE RESULTADOS
# =============================================================================
print(f"Ajustando K-Means con k={best_k} y guardando resultados...")
kmeans = KMeans(n_clusters=best_k, n_init=100, random_state=27)
labels = kmeans.fit_predict(Xs)
centroids_std = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)

# Se Reordenan los clusters y se define paleta de colores
# Identificamos el cluster con "mejores" propiedades (mayor PIB per cápita)
good_cluster_label = centroids_std['gdp_per_capita'].idxmax()

# Si el cluster "bueno" no tiene la etiqueta 0, se invierten todas las etiquetas.
# Por ejemplo, si el bueno es el 1, esta operación convierte los 1s en 0s y los 0s en 1s.
if good_cluster_label != 0:
    labels = 1 - labels
    print("Se han invertido las etiquetas de los clusters para mantener la consistencia (0=Mejor, 1=Peor).")

# Se define una paleta de colores fija y con significado
palette = {
    0: '#1f77b4',  # Azul para el Cluster 0 ("mejores" propiedades)
    1: '#d62728'   # Rojo para el Cluster 1 ("peores" propiedades)
}


# Se guarda asignación de clusters por país
res = X0[id_cols].copy()
res["cluster"] = labels
res.to_csv(tables_clu / f"kmeans_clusters_k{best_k}.csv", sep=";", index=False)

# Se guardan centroides estandarizados
centroids_std = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
centroids_std.index.name = "cluster"
centroids_std.to_csv(tables_clu / f"kmeans_centroids_std_k{best_k}.csv", sep=";")

# Se guardan centroides en escala original
centroids_orig = pd.DataFrame(scaler.inverse_transform(centroids_std.values), columns=numeric_cols, index=centroids_std.index)
centroids_orig.index.name = "cluster"
centroids_orig.to_csv(tables_clu / f"kmeans_centroids_orig_k{best_k}.csv", sep=";")
print("Resultados del clustering (asignaciones y centroides) guardados.")

# =============================================================================
# 5. VISUALIZACIÓN Y ANÁLISIS DE CLUSTERS
# =============================================================================
print("Generando visualizaciones de los clusters...")

# --- a) PCA 2D para visualizar separación de clusters ---
pca = PCA(n_components=2, random_state=27)
Z = pca.fit_transform(Xs)
viz = pd.DataFrame(Z, columns=["PC1", "PC2"])
viz["cluster"] = labels

plt.figure(figsize=(9, 7))

sns.scatterplot(data=viz, x="PC1", y="PC2", hue="cluster", palette=palette, s=50, alpha=0.8, edgecolor="k", linewidth=0.5)
plt.title(f"Visualización de Clústeres con PCA (k={best_k})")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Clúster")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(dir_plots_analyst_clu / f"pca_kmeans_k{best_k}.png", dpi=200)
plt.show()

# --- b) Heatmap de centroides para interpretar clusters ---
centroids_std_sorted = pd.DataFrame(Xs).assign(cluster=labels).groupby('cluster').mean()
centroids_std_sorted.columns = numeric_cols

plt.figure(figsize=(14, 10))
sns.heatmap(centroids_std_sorted.T, cmap="vlag_r", center=0, annot=True, fmt=".2f", linewidths=0.5, linecolor="#f0f0f0")
plt.title(f"Perfil de Centroides Estandarizados (k={best_k})", fontsize=15)
plt.xlabel("Clúster", fontsize=12) 
plt.ylabel("Variable", fontsize=12)
plt.xticks(rotation=0) 
plt.yticks(rotation=0) 
plt.tight_layout()
plt.savefig(dir_plots_analyst_clu / f"heatmap_centroides_std_k{best_k}.png", dpi=200)
plt.show()

# --- c) Mapa Mundial de Clusters ---
world = gpd.read_file(geo_path)
map_gdf = world.merge(res, left_on="id", right_on="code", how="left")

fig, ax = plt.subplots(1, 1, figsize=(18, 9))
# Capa base para países sin datos
world[~world['id'].isin(map_gdf['code'])].plot(ax=ax, color='lightgrey', edgecolor="white", linewidth=0.5)
# Se grafica cada cluster con su color
for k, color in palette.items():
    map_gdf[map_gdf['cluster'] == k].plot(ax=ax, color=color, edgecolor="white", linewidth=0.5, label=f"Clúster {k}")

handles = [
    mpatches.Patch(color=palette[0], label='Clúster 0'),
    mpatches.Patch(color=palette[1], label='Clúster 1'),
    mpatches.Patch(color='lightgrey', label='Sin asignar')
]
ax.legend(handles=handles, title="Leyenda", loc='lower left', frameon=True, fontsize=12)
ax.set_title(f'Mapa Mundial — Clústeres COVID-19 (k={best_k})', fontdict={'fontsize': '18', 'fontweight': '3'})
ax.set_axis_off()
plt.tight_layout()
plt.savefig(dir_plots_maps_static / f"map_clusters_k{best_k}.png", dpi=300, bbox_inches="tight")
plt.show()

# --- d) Radar Chart para comparar perfiles de clusters ---
radar_vars = [
    "people_vaccinated_per_hundred", "gdp_per_capita", "median_age", "hospital_beds_per_thousand",
    "total_deaths_per_million", "stringency_index", "extreme_poverty"
]
summary_std_radar = centroids_std[[c for c in radar_vars if c in centroids_std.columns]]

labels_r = summary_std_radar.columns
num_vars = len(labels_r)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for k, row in summary_std_radar.iterrows():
    values = row.tolist()
    values += values[:1]
    color = palette.get(k)
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=f"Clúster {k}")
    ax.fill(angles, values, color=color, alpha=0.25)

ax.set_yticklabels([]); ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_r, size=10)
plt.title("Perfil Comparativo de Clústeres (Radar)", size=16, y=1.1)
ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1)); plt.tight_layout()
plt.savefig(dir_plots_analyst_clu / f"radar_clusters_k{best_k}.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n--- Proceso de clustering finalizado con colores consistentes. ---")
