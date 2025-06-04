#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:03:37 2025

@author: cdeval
"""

import duckdb
import requests
import os

outdir = './data/'
os.makedirs(outdir, exist_ok=True)


# URL of the Parquet file to download
claims_url = "https://www.fema.gov/about/reports-and-data/openfema/FimaNfipClaims.parquet"
policies_url = "https://www.fema.gov/about/reports-and-data/openfema/FimaNfipPolicies.parquet"


# path of saved file
claims_file_path = os.path.join(outdir, "FimaNfipClaims.parquet")
policies_file_path = os.path.join(outdir, "FimaNfipPolicies.parquet")
# send an HTTP GET request to the URL


with requests.get(claims_url, stream=True) as r:
    r.raise_for_status()
    with open(claims_file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)




with requests.get(policies_url, stream=True) as r:
    r.raise_for_status()
    with open(policies_file_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
                
   