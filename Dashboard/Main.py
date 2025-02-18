import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk mengatur mode gelap
def set_dark_mode():
    st.markdown(
        """
        <style>
            body { background-color: #121212; color: white; }
            [data-testid="stSidebar"] { background-color: #1E1E1E; }
            .stApp { background-color: #121212; }
            .stButton>button { background-color: #BB86FC; color: white; }
            h1, h2, h3 { color: #BB86FC; }
        </style>
        """,
        unsafe_allow_html=True
    )

set_dark_mode()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data_with_clusters_clean.csv")  

df = load_data()

# Sidebar: Filter berdasarkan Artist (maksimal 5 artist)
selected_artists = st.sidebar.multiselect(
    "Pilih Artist (maksimal 5)",
    options=sorted(df["Artist"].unique()),
    default=sorted(df["Artist"].unique())[:5]
)

if len(selected_artists) > 5:
    st.sidebar.warning("Anda hanya bisa memilih maksimal 5 artist.")
    selected_artists = selected_artists[:5] 

# Sidebar: Filter berdasarkan Cluster
selected_clusters = st.sidebar.multiselect(
    "Pilih Cluster",
    options=sorted(df["Cluster"].unique()),
    default=sorted(df["Cluster"].unique())
)

# Sidebar: Filter berdasarkan Stream
min_stream, max_stream = st.sidebar.slider(
    "Pilih Rentang Jumlah Stream",
    int(df["Stream"].min()),
    int(df["Stream"].max()),
    (int(df["Stream"].min()), int(df["Stream"].max()))
)

# Sidebar: Filter berdasarkan Energy
min_energy, max_energy = st.sidebar.slider(
    "Pilih Rentang Energy",
    float(df["Energy"].min()),
    float(df["Energy"].max()),
    (float(df["Energy"].min()), float(df["Energy"].max()))
)

# Sidebar: Filter berdasarkan Views
min_views, max_views = st.sidebar.slider(
    "Pilih Rentang Jumlah Views",
    int(df["Views"].min()),
    int(df["Views"].max()),
    (int(df["Views"].min()), int(df["Views"].max()))
)

# Filter data berdasarkan input pengguna
filtered_df = df[
    (df["Artist"].isin(selected_artists)) &
    (df["Cluster"].isin(selected_clusters)) &
    (df["Stream"] >= min_stream) & (df["Stream"] <= max_stream) &
    (df["Energy"] >= min_energy) & (df["Energy"] <= max_energy) &
    (df["Views"] >= min_views) & (df["Views"] <= max_views)
]

# Tampilkan data yang difilter
st.title("Visualisasi Data Musik Berdasarkan Filtering")
st.subheader("Dataset Setelah Filter")
st.write(filtered_df)

# Visualisasi: Distribusi Stream per Cluster
st.subheader("Distribusi Jumlah Stream per Cluster")
fig, ax = plt.subplots()
sns.boxplot(x="Cluster", y="Stream", data=filtered_df, palette="magma", ax=ax)
st.pyplot(fig)

# Visualisasi: Scatter Plot Energy vs Danceability
st.subheader("Scatter Plot: Energy vs Danceability")
fig, ax = plt.subplots()
sns.scatterplot(
    x="Energy", y="Danceability", 
    hue="Cluster", palette="viridis", 
    data=filtered_df, ax=ax
)
st.pyplot(fig)

# Visualisasi: Barplot Jumlah Lagu per Artist
st.subheader("Jumlah Lagu per Artist")
artist_counts = filtered_df["Artist"].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=artist_counts.index, y=artist_counts.values, palette="coolwarm", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Visualisasi: Correlation Heatmap
# Checkbox untuk menampilkan atau menyembunyikan heatmap
hide_corr_heatmap = st.checkbox("Sembunyikan Korelasi Heatmap", value=False)

if not hide_corr_heatmap:
    st.subheader("Korelasi Fitur Numerik")
    if filtered_df.empty:
        st.warning("Tidak ada data yang tersedia untuk ditampilkan.")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_df = filtered_df.select_dtypes(include=["number"])  # Hanya kolom numerik
        if numeric_df.shape[1] < 2:
            st.warning("Tidak cukup fitur numerik untuk menghitung korelasi.")
        else:
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)


