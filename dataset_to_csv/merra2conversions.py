import os
import glob
import xarray as xr
import pandas as pd

data_dir = r"C:\Users\Brindha\Downloads\Merra2 TN"  
output_csv = os.path.join(data_dir, 'merra2_TN_5days.csv')

variables = [
    "PBLH", "HFLUX", "EFLUX", "PRECTOT", "PRECSNO", "SPEEDMAX"
]

records = []
files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))

for file in files:
    print(f"Processing {os.path.basename(file)} ...")
    ds = xr.open_dataset(file)
    # No region slicing needed, already subset for India
    times = ds['time'].values
    lats = ds['lat'].values
    lons = ds['lon'].values
    for t_idx, t in enumerate(times):
        time_str = pd.to_datetime(str(t)).strftime('%Y-%m-%d %H:%M:%S')
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                row = {
                    "time": time_str,
                    "lat": float(lat),
                    "lon": float(lon)
                }
                for var in variables:
                    if var in ds:
                        val = ds[var].isel(time=t_idx, lat=i, lon=j).item()
                        row[var.lower()] = float(val)
                records.append(row)
    ds.close()

df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"Saved {len(df)} rows to {output_csv}")