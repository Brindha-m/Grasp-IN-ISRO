import h5py
import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime, timedelta

data_dir = 'C:/Users/Brindha/Downloads/INSAT 30DAYS'
output_csv = os.path.join(data_dir, 'insat3dr_aoddata_30days.csv')
files = sorted(glob.glob(os.path.join(data_dir, '*.h5')))

aod_records = []

for file in files:
    with h5py.File(file, 'r') as f:
        # Read datasets
        aod = f['AOD'][:][0, :, :]  # (time=1, lat, lon)
        lat = f['latitude'][:]      # (lat,)
        lon = f['longitude'][:]     # (lon,)

        # Handle fill values and invalid physical range
        aod = np.where((aod == -999.0) | (aod < 0) | (aod > 5), np.nan, aod)

        # Create 2D meshgrid for latitude and longitude
        lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')

        # Flatten arrays for tabular output
        flat_lat = lat2d.flatten()
        flat_lon = lon2d.flatten()
        flat_aod = aod.flatten()

        # Parse observation time from the file
        time_value = f['time'][0]  # Should be a single-element array
        # Get the units attribute and parse the base time
        time_units = f['time'].attrs['units']
        if isinstance(time_units, bytes):
            time_units = time_units.decode()
        base_time_str = time_units.split('since')[1].strip()
        base_time = datetime.strptime(base_time_str, "%Y-%m-%d %H:%M:%S")
        obs_time = base_time + timedelta(minutes=float(time_value))
        time_str = obs_time.strftime("%Y-%m-%d %H:%M:%S")

        # Only keep valid pixels
        valid = ~np.isnan(flat_aod)
        for lt, ln, ad in zip(flat_lat[valid], flat_lon[valid], flat_aod[valid]):
            aod_records.append({'time': time_str, 'lat': lt, 'lon': ln, 'aod': ad})

# Create DataFrame
df = pd.DataFrame(aod_records)
df.to_csv(output_csv, index=False)
print(f"AOD table with time saved to {output_csv} with {len(df)} rows.")