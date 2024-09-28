import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import seaborn as sns
plt.style.use('dark_background')

df = pd.read_csv("data.csv")
 
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
df['Rain_Status'] = df['RAIN'].apply(lambda x: 'Rain' if x > 0 else 'No Rain')

st.title("Dashboard Visualisation Air Quality")
st.image('img.jpg', use_column_width=True)

def classify_air_quality(row):
    if row['PM2.5'] < 35 and row['PM10'] < 50:
        return 'Baik'
    elif 35 <= row['PM2.5'] < 75 or 50 <= row['PM10'] < 100:
        return 'Sedang'
    elif 75 <= row['PM2.5'] < 150 or 100 <= row['PM10'] < 200:
        return 'Tidak Sehat'
    else:
        return 'Berbahaya'

df['air_quality'] = df.apply(classify_air_quality, axis=1)

start_date, end_date = st.sidebar.date_input("Pilih rentang tanggal", value=[pd.to_datetime("2013-03-01"), pd.to_datetime("2017-02-28")])
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

station_filter = st.sidebar.multiselect(
    "Pilih station",
    options=df['station'].unique(),
    default=df['station'].unique()
)

filtered_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date) & (df['station'].isin(station_filter))]


if not filtered_df.empty:
    
    col1, col2 = st.columns(2)
    max_date = filtered_df['datetime'].max()
    frequency = filtered_df['datetime'].count()
    monetary = filtered_df['PM2.5'].mean()

    col1.metric(label="Frequency (Jumlah Pengukuran)",
               value=f"{frequency}",
               delta_color="normal")

    col2.metric(label="Rata-rata PM2.5",
               value=f"{monetary:.2f}",
               delta_color="normal")
else:
    st.warning("Tidak ada data yang tersedia untuk rentang tanggal dan station yang dipilih.")


demographic_df = filtered_df.groupby('station').agg(
        Total_Measurements=('datetime', 'count'),
        PM2_5=('PM2.5', 'mean'),
        PM10=('PM10', 'mean'),
        SO2=('SO2', 'mean'),
        NO2=('NO2', 'mean'),
        CO=('CO', 'mean'),
        O3=('O3', 'mean')
    ).reset_index()

st.subheader("Konsentrasi PM2.5 dan PM10")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x='Rain_Status', y='PM2.5', data=filtered_df)
plt.title('PM2.5 Concentration on Rainy vs Non-Rainy Days')
plt.xlabel('Rain Status')
plt.ylabel('PM2.5')

plt.subplot(1, 2, 2)
sns.boxplot(x='Rain_Status', y='PM10', data=filtered_df)
plt.title('PM10 Concentration on Rainy vs Non-Rainy Days')
plt.xlabel('Rain Status')
plt.ylabel('PM10')
st.pyplot(plt)



def plot_average_pollutant(demographic_df, pollutant):
    """Fungsi untuk membuat grafik rata-rata polutan."""
    fig = px.bar(demographic_df, x='station', y=pollutant,
                 labels={pollutant: f'Rata-rata {pollutant}', 'station': 'Station'})
    return fig





st.subheader("Rata-rata PM2.5 per Station")
st.plotly_chart(plot_average_pollutant(demographic_df, 'PM2_5'))



st.subheader("Scatter Plot PM2.5 vs PM10")
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PM2.5', 
    y='PM10', 
    hue='air_quality', 
    data=filtered_df, 
    palette={'Baik': 'green', 'Sedang': 'yellow', 'Tidak Sehat': 'orange', 'Berbahaya': 'red'},
    s=100
)

plt.xlabel('PM2.5 Concentration', fontsize=12)
plt.ylabel('PM10 Concentration', fontsize=12)
st.pyplot(plt)


