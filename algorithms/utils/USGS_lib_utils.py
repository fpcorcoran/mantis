import json
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def LoadSpecLib(year="2009",path="../../spectral_libs/AVIRIS_Spectral_Library.json"):
    """Load USGS Spectral Library"""
    with open(path,'r') as obj:
        splib = json.load(obj)

    return splib[year]

def MakeDataframe(splib, types=['Vegetation','SoilsAndMixtures','Liquids','Minerals',"Coatings","OrganicCompounds","ArtificialMaterial"]):
    """
    Create Pandas DataFrame from spectral library

    types:
        'Vegetation'
        'SoilsAndMixtures'
        'Liquids'
        'Minerals'
        'Coatings'
        'OrganicCompounds'
        'ArtificialMaterial'
    """
    df = {}

    wv = splib['metadata']['Wavelength']

    for t in types:
        for key, items in splib['library'][t].items():
            df[key] = items['Data']

    df = pd.DataFrame.from_dict(df)
    df.index = wv

    return df

def ChangeSensor(df, new_wv):
    """Interpolate spectra to new band centers"""
    new_df = {}

    wv = df.index.values.astype(np.float64)

    for entry in df.columns:
        intensity = df[entry].values

        cs = interp1d(wv, intensity, kind='cubic')

        new_intensity = cs(new_wv)

        new_df[entry] = new_intensity

        new_df = pd.DataFrame.from_dict(new_df)
        new_df.index = new_wv

    return new_df

def synthetic_data(spectra,abundances,dims=(100,100),pure_pix=True):
    """generate test data via Dirichlet distribution"""

    s = np.random.dirichlet(abundances,np.product(dims))

    if pure_pix:
        pix = np.product(dims)
        for i in spectra:
            idx = np.random.choice(pix)
            s[idx,:] = np.ones_like(s[idx,:])

    synthetic_hsi = s@spectra

    return (synthetic_hsi.reshape(((-1,)+dims)), s)
