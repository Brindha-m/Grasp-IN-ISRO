# GRASP-IN (Ground And Satellite based Air PM Predictions in India)

> Bharatiya Antariksh Hackathon 2025

## Project Overview
- **Grasp-IN** turn what satellites see into what people need, actionable, ground-level air quality intelligence for every corner of the nation. The app collects satellite-derived Aerosol Optical Depth (AOD) data (from INSAT), meteorological parameters (from MERRA-2), and ground truth PM measurements (from CPCB stations).

## Key Features

- Using spatial and temporal data fusion, the system prepares a comprehensive dataset involving all steps from EDA to Feature Engineering. Machine learning models (Random Forest, XGBoost) are trained to learn the relationship between satellite signals and actual ground-level PM2.5/PM10. 
- Our model has (date range from 25 May – 2 June 2025 due to storage limits) achieved outstanding accuracy, with XGBoost explaining 98% of the variance in PM2.5/PM10 (R² = 0.98) and a low RMSE of 5.5 µg/m³ . The model then predicts PM values for any location - even in regions lacking ground stations enabling nationwide, high-resolution air quality mapping. 
- Unlike traditional systems that rely only on ground sensors or provide limited coverage, our solution fuses satellite AOD, meteorological reanalysis, and ground station data using advanced machine learning techniques delivers accurate, real-time PM2.5/PM10 predictions for any location
  
## Project Flow 
<img width="753" alt="{47426A0F-1457-44B4-9090-0AABA955ACCB}" src="https://github.com/user-attachments/assets/a2f8ee6d-9242-4189-8660-ecfa08f5c4a4" />


