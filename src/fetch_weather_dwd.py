"""Download DWD hourly observations and build data/weather_hourly.csv.

Source: DWD Climate Data Center (opendata.dwd.de), observed hourly values.
Variables: global radiation (W/m2), sunshine (min/h), cloud cover (1/8), temperature (degC),
averaged over a handful of stations spread over Germany. Output is on the SMARD local
wall-clock hourly grid (Europe/Berlin, duplicate October hour dropped).

Run:  python src/fetch_weather_dwd.py
"""
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly"
DATA = Path(__file__).resolve().parents[1] / "data"
CACHE = DATA / "weather_raw"

# station id -> name (spread across Germany; all have solar records covering 2019-2026)
STATIONS = {
    "01975": "Hamburg", "03987": "Potsdam", "00662": "Braunschweig", "07365": "Bochum",
    "00853": "Chemnitz", "05705": "Wuerzburg", "04928": "Stuttgart", "05404": "Weihenstephan",
    "04336": "Saarbruecken", "03668": "Nuernberg",
}


def _get(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def _list(folder: str) -> str:
    return _get(f"{BASE}/{folder}/").decode("utf-8", "ignore")


def _cached_zip(name: str, url: str) -> zipfile.ZipFile:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        p.write_bytes(_get(url))
    return zipfile.ZipFile(io.BytesIO(p.read_bytes()))


def _read_product(z: zipfile.ZipFile) -> pd.DataFrame:
    name = next(n for n in z.namelist() if n.startswith("produkt_"))
    return pd.read_csv(z.open(name), sep=";", skipinitialspace=True, na_values=["-999", "-999.0", "-9999"])


def _station_files(folder: str, prefix: str, sid: str) -> list[str]:
    """('historical' + 'recent') zip file names for one station, or the single 'row' file for solar."""
    out = []
    for sub in ("historical", "recent"):
        try:
            html = _list(f"{folder}/{sub}")
        except requests.HTTPError:
            continue
        out += [f"{sub}/{m}" for m in re.findall(rf"stundenwerte_{prefix}_{sid}_[^\"]*?\.zip", html)]
    return sorted(set(out))


def load_variable(folder: str, prefix: str, sid: str, column: str, hour_fmt: str) -> pd.Series | None:
    if folder == "solar":
        files = [f"stundenwerte_ST_{sid}_row.zip"]
    else:
        files = _station_files(folder, prefix, sid)
    parts = []
    for f in files:
        try:
            z = _cached_zip(f"{folder}_{Path(f).name}", f"{BASE}/{folder}/{f}")
        except requests.HTTPError:
            continue
        df = _read_product(z)
        if column not in df:
            continue
        ts = pd.to_datetime(df["MESS_DATUM"].astype(str).str[:10], format=hour_fmt)  # floor to hour, UTC (interval END for solar)
        vals = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if column == "V_N":
            vals[vals < 0] = float("nan")  # -1 = missing
        parts.append(pd.Series(vals, index=ts))
    if not parts:
        return None
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")]  # recent overrides historical overlap
    return s.sort_index()


def station_mean(folder, prefix, column, hour_fmt) -> pd.Series:
    series = []
    for sid, name in STATIONS.items():
        s = load_variable(folder, prefix, sid, column, hour_fmt)
        if s is None:
            print(f"  skip {folder}/{name} ({sid})")
            continue
        series.append(s.rename(sid))
        print(f"  {folder}/{name}: {s.index.min()} -> {s.index.max()}, NaN {s.isna().mean():.1%}")
    return pd.concat(series, axis=1).mean(axis=1)


def to_local(s: pd.Series) -> pd.Series:
    """UTC hour-start index -> Europe/Berlin wall clock, no tz, first duplicate kept (matches SMARD)."""
    s = s.copy()
    s.index = s.index.tz_localize("UTC").tz_convert("Europe/Berlin").tz_localize(None)
    return s[~s.index.duplicated(keep="first")]


def build() -> pd.DataFrame:
    fmt = "%Y%m%d%H"
    rad = station_mean("solar", "ST", "FG_LBERG", fmt) * 10000 / 3600  # J/cm2 per hour -> W/m2
    sun = station_mean("solar", "ST", "SD_LBERG", fmt)  # minutes of sunshine per hour
    cloud = station_mean("cloudiness", "N", "V_N", fmt)  # eighths (0-8); DWD uses -1 for missing
    temp = station_mean("air_temperature", "TU", "TT_TU", fmt)
    # radiation / sunshine are sums over the hour that ENDS at MESS_DATUM -> label with the interval start
    # (like SMARD). Checked: corr(solar(t), radiation(t+1 h)) was the maximum before this shift.
    one_h = pd.Timedelta(hours=1)
    rad.index, sun.index = rad.index - one_h, sun.index - one_h
    df = pd.concat({"radiation": to_local(rad), "sunshine": to_local(sun),
                    "cloud_cover": to_local(cloud), "temp": to_local(temp)}, axis=1)
    df = df.loc["2019-01-01":]
    return df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h", name="timestamp"))


if __name__ == "__main__":
    w = build()
    w.to_csv(DATA / "weather_hourly.csv")
    print(w.index.min(), "->", w.index.max(), w.shape)
    print(w.isna().mean().round(3).to_dict())
    print(w.describe().round(1))
