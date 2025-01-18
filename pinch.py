import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Configuration de la page
st.set_page_config(page_title="PinchPro - Application Exemple", page_icon="📊", layout="wide")

# Titre principal
st.title("Outils Analyse Pinch")
st.markdown("""
L'outil d'analyse Pinch est un outil d'optimisation énergétique utilisé dans les procédés industriels pour réduire les consommations de chauffage et de refroidissement. 
Il repose sur une méthodologie scientifique permettant d'identifier les interactions thermiques optimales entre les courants chauds et froids afin de minimiser les besoins en énergie externe.
""")

# Barre latérale : Guide d'utilisation
st.sidebar.title("Guide d'utilisation")
st.sidebar.markdown("""
1. **Description du procédé** :  
   - Décrivez brièvement le système analysé.  
2. **Problématique et objectifs** :  
   - Définissez les défis énergétiques et les objectifs.  
3. **Extraction des données** :  
   - Entrez les flux thermiques avec températures et capacité calorifique (CP).  
4. **ΔTmin** :  
   - Ajustez la différence de température minimale.  
5. **Courbes composites** :  
   - Analysez les courbes affichées.  
""")

# Section 1 : Description du procédé
st.header("1. Description du procédé")
description = st.text_area("Décrivez le procédé étudié :", placeholder="Par exemple, le refroidissement des effluents dans une usine...")

# Section 2 : Problématique et objectifs
st.header("2. Problématique et objectifs")
problematique = st.text_area("Problématique :", placeholder="Décrivez les problèmes énergétiques rencontrés.")
objectifs = st.text_area("Objectifs :", placeholder="Quels sont les objectifs de l'analyse Pinch ?")

# Section 3 : Extraction des données
st.header("3. Extraction des données")
num_streams = st.number_input("Nombre de flux (courants)", min_value=2, max_value=10, value=4, step=1)

streams_data = []
for i in range(num_streams):
    st.subheader(f"Courant {i+1}")
    stream_type = st.selectbox(f"Type de courant {i+1}", ["Chaud", "Froid"], key=f"type_{i}")
    t_initial = st.number_input(f"Température initiale (°C) - Courant {i+1}", key=f"tinit_{i}")
    t_final = st.number_input(f"Température finale (°C) - Courant {i+1}", key=f"tfinal_{i}")
    cp = st.number_input(f"Capacité calorifique (CP, kW/°C) - Courant {i+1}", value=1.0, key=f"cp_{i}")
    streams_data.append({"Type": stream_type, "T_in": t_initial, "T_out": t_final, "CP": cp})

df_streams = pd.DataFrame(streams_data)

# Vérification des données
if df_streams.empty or len(df_streams[df_streams["Type"] == "Chaud"]) == 0 or len(df_streams[df_streams["Type"] == "Froid"]) == 0:
    st.error("Veuillez entrer au moins un courant chaud et un courant froid.")
    st.stop()

# Affichage des données
st.subheader("Données des flux")
st.write(df_streams)

# Section 4 : Détermination des cibles énergétiques
st.header("4. Détermination des cibles énergétiques")
delta_Tmin = st.slider("ΔTmin (°C)", min_value=5, max_value=20, value=10, step=1)

def adjust_temperatures_with_min_gap(row, delta_Tmin):
    if row["Type"] == "Chaud":
        return row["T_in"] - delta_Tmin / 2, row["T_out"] - delta_Tmin / 2
    else:
        return row["T_in"] + delta_Tmin / 2, row["T_out"] + delta_Tmin / 2

df_streams[["T_in_adj", "T_out_adj"]] = df_streams.apply(
    lambda row: adjust_temperatures_with_min_gap(row, delta_Tmin), axis=1, result_type="expand"
)

def construct_composite_curves(df, stream_type):
    df_filtered = df[df["Type"] == stream_type].copy()
    if df_filtered.empty:
        return [], []
    
    temperatures = []
    heat_loads = []
    cumulative_heat = 0
    
    if stream_type == "Chaud":
        df_filtered = df_filtered.sort_values("T_in_adj", ascending=False)
    else:
        df_filtered = df_filtered.sort_values("T_in_adj", ascending=True)
    
    for _, row in df_filtered.iterrows():
        delta_T = abs(row["T_out_adj"] - row["T_in_adj"])
        heat_load = delta_T * row["CP"]
        
        temperatures.extend([row["T_in_adj"], row["T_out_adj"]])
        heat_loads.extend([cumulative_heat, cumulative_heat + heat_load])
        cumulative_heat += heat_load
    
    return temperatures, heat_loads

def plot_composite_curves(T_hot, Q_hot, T_cold, Q_cold, delta_Tmin):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Créer des points d'interpolation pour un meilleur contrôle
    Q_interp = np.linspace(min(min(Q_hot), min(Q_cold)), 
                          max(max(Q_hot), max(Q_cold)), 1000)
    
    # Interpoler les températures
    T_hot_interp = np.interp(Q_interp, Q_hot, T_hot)
    T_cold_interp = np.interp(Q_interp, Q_cold, T_cold)
    
    # Ajouter un espacement vertical entre les courbes (delta_Tmin)
    T_cold_spaced = T_cold_interp + delta_Tmin
    
    # Recalculer les courbes avec l'espacement
    T_hot_new = np.interp(Q_hot, Q_interp, T_hot_interp)
    T_cold_new = np.interp(Q_cold, Q_interp, T_cold_spaced)
    
    # Tracer les courbes principales
    ax.plot(Q_hot, T_hot_new, 'r-', label='Composite Hot', linewidth=2)
    ax.plot(Q_cold, T_cold_new, 'b-', label='Composite Cold', linewidth=2)
    
    # Trouver le point de pincement
    temp_diff = T_hot_interp - T_cold_spaced
    pinch_idx = np.argmin(np.abs(temp_diff))
    
    # Coordonnées du point de pincement
    pinch_x = Q_interp[pinch_idx]
    pinch_y_hot = np.interp(pinch_x, Q_hot, T_hot_new)
    pinch_y_cold = np.interp(pinch_x, Q_cold, T_cold_new)
    
    # Définir les limites du graphique avec marges
    q_min = min(min(Q_hot), min(Q_cold))
    q_max = max(max(Q_hot), max(Q_cold))
    margin_q = (q_max - q_min) * 0.1
    
    t_min = min(min(T_cold_new), min(T_hot_new))
    t_max = max(max(T_cold_new), max(T_hot_new))
    margin_t = (t_max - t_min) * 0.1
    
    ax.set_xlim(q_min - margin_q, q_max + margin_q)
    ax.set_ylim(t_min - margin_t, t_max + margin_t)
    
    # Zone de récupération de chaleur (verte)
    ax.fill_between(Q_hot, T_hot_new, np.interp(Q_hot, Q_cold, T_cold_new), 
                   where=(np.interp(Q_hot, Q_cold, T_cold_new) <= T_hot_new),
                   color='lightgreen', alpha=0.3, label='Heat Recovery')
    
    # Zone de refroidissement (bleue)
    cooling_requirement = abs(min(Q_cold) - min(Q_hot))
    if cooling_requirement > 0:
        cooling_height = max(T_hot_new) - min(T_cold_new)
        cooling_base = min(T_cold_new)
        ax.fill_between([min(Q_hot) - margin_q/2, min(Q_hot) + cooling_requirement], 
                       [cooling_base, cooling_base + cooling_height],
                       color='lightblue', alpha=0.3,
                       label='Cooling Utility')
        
        # Libellé Cooling Utility
        ax.text(min(Q_hot) + cooling_requirement/2, 
                cooling_base + cooling_height/4,
                f'Cooling Utility\n{cooling_requirement:.0f} kW',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Zone de chauffage (rouge)
    heating_requirement = abs(max(Q_hot) - max(Q_cold))
    if heating_requirement > 0:
        heating_height = max(T_hot_new) - min(T_cold_new)
        heating_base = min(T_cold_new)
        ax.fill_between([max(Q_cold), max(Q_cold) + heating_requirement + margin_q/2],
                       [heating_base, heating_base + heating_height],
                       color='lightpink', alpha=0.3,
                       label='Heating Utility')
        
        # Libellé Heating Utility
        ax.text(max(Q_cold) + heating_requirement/2,
                heating_base + 3*heating_height/4,
                f'Heating Utility\n{heating_requirement:.0f} kW',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Point de pincement et annotations
    if not np.isnan(pinch_y_hot) and not np.isnan(pinch_y_cold):
        # Point de pincement
        arrow_props = dict(facecolor='green', shrink=0.05, width=1, headwidth=8)
        ax.annotate('Pinch\nPoint', 
                    xy=(pinch_x, pinch_y_hot),
                    xytext=(pinch_x - margin_q, pinch_y_hot + margin_t),
                    arrowprops=arrow_props,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # ΔTmin
        ax.annotate(f'ΔTmin = {delta_Tmin} °C',
                    xy=(pinch_x, (pinch_y_hot + pinch_y_cold)/2),
                    xytext=(pinch_x + margin_q/2, (pinch_y_hot + pinch_y_cold)/2),
                    ha='left',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    # Heat Recovery au centre de la zone verte
    recovery = abs(max(Q_cold) - min(Q_cold))
    recovery_x = (max(Q_cold) + min(Q_cold))/2
    recovery_y = (max(T_cold_new) + min(T_hot_new))/2
    
    ax.text(recovery_x, recovery_y,
            f'Heat Recovery\n{recovery:.0f} kW',
            ha='center', va='center',
            color='green',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Configuration finale du graphique
    ax.set_xlabel('Heat Flow (kW)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Composite Curves', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    return fig
# Génération et affichage des courbes composites
T_hot, Q_hot = construct_composite_curves(df_streams, "Chaud")
T_cold, Q_cold = construct_composite_curves(df_streams, "Froid")

if len(T_hot) > 0 and len(T_cold) > 0:
    fig = plot_composite_curves(T_hot, Q_hot, T_cold, Q_cold, delta_Tmin)
    st.pyplot(fig)
    
    # Section 5 : Réseaux d'échangeurs optimisés
    st.header("5. Réseaux d'échangeurs optimisés")
    
    G = nx.DiGraph()
    hot_streams = df_streams[df_streams["Type"] == "Chaud"]
    cold_streams = df_streams[df_streams["Type"] == "Froid"]
    
    for i, hot in hot_streams.iterrows():
        for j, cold in cold_streams.iterrows():
            if hot["T_in_adj"] > cold["T_out_adj"]:
                heat_load = min(
                    hot["CP"] * (hot["T_in"] - hot["T_out"]),
                    cold["CP"] * (cold["T_out"] - cold["T_in"])
                )
                G.add_edge(
                    f"Chaud {i+1}",
                    f"Froid {j+1}",
                    weight=round(heat_load, 2)
                )
    
    if len(G.edges()) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=2000, font_size=10, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        st.pyplot(fig)
    else:
        st.warning("Aucun échange de chaleur possible avec les données actuelles.")
else:
    st.error("Données insuffisantes pour générer les courbes composites.")