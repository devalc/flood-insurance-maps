#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import duckdb
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import numpy as np


def configure_paths(base_dir='./'):
    """
    Configures and returns file paths and directories.
    """
    outdir = os.path.join(base_dir, 'figures')
    os.makedirs(outdir, exist_ok=True)
    print(f"Output directory created: {outdir}")
    return {
        'outdir': outdir,
        'claims_fp': os.path.join(base_dir, 'data', 'FimaNfipClaims.parquet'),
        'policies_fp': os.path.join(base_dir, 'data', 'FimaNfipPolicies.parquet'),
        'counties_shp': os.path.join(base_dir, 'data', 'us_census_county', 'tl_2024_us_county', 'tl_2024_us_county.shp')
    }

def load_data(claims_fp, policies_fp):
    """
    Loads claims and policies data using DuckDB.
    """
    print("Loading data...")
    try:
        con = duckdb.connect()
        claims_df = con.execute(
            f"SELECT countyCode, yearOfLoss, netBuildingPaymentAmount FROM read_parquet('{claims_fp}')"
        ).df()
        policies_df = con.execute(
            f"SELECT countyCode, policyCount, totalBuildingInsuranceCoverage FROM read_parquet('{policies_fp}')"
        ).df()
        con.close()
        print("Data loaded successfully.")
        return claims_df, policies_df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure your .parquet files exist in the specified 'data' directory.")
        exit() # Exit if data loading fails

def aggregate_data(claims_df, policies_df):
    """
    Aggregates claims and policies data by countyCode.
    """
    print("Aggregating data...")
    claims_agg = (
        claims_df.dropna(subset=['countyCode'])
        .groupby('countyCode')
        .agg(avg_claim_amt=('netBuildingPaymentAmount', 'sum'))
        .reset_index()
    )

    policies_agg = (
        policies_df.dropna(subset=['countyCode'])
        .groupby('countyCode')
        .agg(total_policies=('policyCount', 'sum'),
             avg_coverage=('totalBuildingInsuranceCoverage', 'sum'))
        .reset_index()
    )

    agg_df = pd.merge(claims_agg, policies_agg, on='countyCode', how='outer')
    print("Data aggregated.")
    return agg_df

def perform_bivariate_classification(agg_df):
    """
    Performs bivariate classification and assigns colors based on claim and policy bins.
    """
    print("Performing bivariate classification...")
    agg_df['claim_bin'] = pd.Series(dtype='str')
    agg_df['policy_bin'] = pd.Series(dtype='str')

    valid_claims = agg_df['avg_claim_amt'].notna()
    valid_policies = agg_df['total_policies'].notna()

    # Use pd.qcut only on valid data to avoid errors with NaNs
    agg_df.loc[valid_claims, 'claim_bin'] = pd.qcut(
        agg_df.loc[valid_claims, 'avg_claim_amt'], 3, labels=["1", "2", "3"], duplicates='drop'
    ).astype(str)

    agg_df.loc[valid_policies, 'policy_bin'] = pd.qcut(
        agg_df.loc[valid_policies, 'total_policies'], 3, labels=["A", "B", "C"], duplicates='drop'
    ).astype(str)

    agg_df['claim_bin'] = agg_df['claim_bin'].fillna('')
    agg_df['policy_bin'] = agg_df['policy_bin'].fillna('')
    agg_df['Bi_Class'] = agg_df['claim_bin'] + agg_df['policy_bin']
    agg_df.loc[agg_df['Bi_Class'] == '', 'Bi_Class'] = 'NoData'

    # NEW COLOR MATRIX: Purple-Blue Bivariate Scheme
    # This matrix provides a distinct bivariate color scheme.
    # The rows (claims) and columns (policies) represent increasing values.
    # As you move RIGHT (increasing policies), the color gets more CYAN/BLUE.
    # As you move DOWN (increasing claims), the color gets more MAGENTA/PURPLE.
    darker_colors_matrix = [
        # Policies: Low (A)  Med (B)  High (C)
        ["#e8e8e8", "#ace4e4", "#5ac8c8"],   # Claims: 1 (Low)
        ["#dfb0d6", "#a5add3", "#5694c0"],   # Claims: 2 (Medium)
        ["#be64ac", "#8c62aa", "#3b4994"],   # Claims: 3 (High)
    ]

    color_dict = {}
    claim_labels = ["1", "2", "3"]
    policy_labels = ["A", "B", "C"]
    claim_to_row = {"1": 0, "2": 1, "3": 2} 

    for c in claim_labels:
        for p in policy_labels:
            row = claim_to_row[c]
            col = policy_labels.index(p)
            color_dict[c + p] = darker_colors_matrix[row][col]

    color_dict["NoData"] = "#eddcd2" 
    agg_df['color'] = agg_df['Bi_Class'].map(color_dict)
    print("Bivariate classification complete.")
    return agg_df, color_dict, darker_colors_matrix, claim_labels, policy_labels

def merge_and_filter_counties(agg_df, counties_shp, color_dict):
    """
    Merges aggregated data with county shapefile and filters out non-contiguous states.
    Fills 'color' column with 'NoData' color for counties not found in agg_df.
    """
    print("Merging with county shapefile...")
    try:
        counties = gpd.read_file(counties_shp)
    except Exception as e:
        print(f"Error loading county shapefile: {e}")
        print("Please ensure the 'tl_2024_us_county.shp' file and its companions (.dbf, .shx, etc.) are in the specified directory.")
        exit() 

    agg_df['countyCode'] = agg_df['countyCode'].astype(str).str.zfill(5)

    # Get the set of county codes from the counties GeoDataFrame
    counties_geoids = set(counties['GEOID'].unique())

    # Get the set of county codes from your aggregated data
    agg_df_countycodes = set(agg_df['countyCode'].unique())

    # Find county codes present in agg_df_countycodes but NOT in counties_geoids
    missing_in_shapefile = agg_df_countycodes - counties_geoids

    print(f"\n--- Counties in agg_df but NOT in the shapefile ({len(missing_in_shapefile)} found) ---")
    if missing_in_shapefile:
        missing_df = agg_df[agg_df['countyCode'].isin(list(missing_in_shapefile))]
        print("Information for these counties from agg_df:")
        print(missing_df[['countyCode', 'Bi_Class', 'avg_claim_amt', 'total_policies']])
    else:
        print("All county codes in agg_df are present in the shapefile's GEOID column.")

    merged = counties.merge(agg_df, left_on='GEOID', right_on='countyCode', how='left')

    # IMPORTANT FIX: Fill NaN colors resulting from the left merge
    # This happens when a county exists in the shapefile but not in agg_df
    merged['color'] = merged['color'].fillna(color_dict["NoData"])
    merged['Bi_Class'] = merged['Bi_Class'].fillna('NoData') # Also fill Bi_Class for consistency

    excluded_fps = ['02', '15', '72', '66', '60', '69', '78']
    merged_conus = merged[~merged['STATEFP'].isin(excluded_fps)]
    merged_nonconus = merged[merged['STATEFP'].isin(excluded_fps)]
    print("County merge and filter complete.")
    return merged_conus, merged_nonconus

def plot_maps(merged_conus, merged_nonconus, inset_defs):
    """
    Plots the main CONUS map and inset maps for non-contiguous states,
    with labels positioned right on top of each inset's geometry.
    """
    print("Generating maps...")
    fig = plt.figure(figsize=(16, 11))
    main_ax = fig.add_subplot(1, 1, 1)

    # Plot CONUS states
    merged_conus.plot(ax=main_ax, color=merged_conus['color'], edgecolor='white', linewidth=0.4)
    main_ax.axis("off") # Turn off axis for cleaner map

    # Plot insets for non-CONUS states/territories
    for statefp, meta in inset_defs.items():
        ax_inset = fig.add_axes(meta['bounds'])
        subset = merged_nonconus[merged_nonconus['STATEFP'] == statefp]

        if not subset.empty:
            subset.plot(ax=ax_inset, color=subset['color'], edgecolor='white', linewidth=0.3)
            ax_inset.axis("off") # Turn off axis for cleaner inset map

            # Get the bounding box of the actual geometries in data coordinates
            minx, miny, maxx, maxy = subset.total_bounds

            # Calculate the top-center point of this geometry in data coordinates
            data_x_center = (minx + maxx) / 2
            data_y_top = maxy

            # Transform this data point to the 'display' (pixel) coordinates of the figure.
            display_x, display_y = ax_inset.transData.transform((data_x_center, data_y_top))

            # Transform the display (pixel) coordinates to 'figure' coordinates (0-1 range).
            figure_x, figure_y = fig.transFigure.inverted().transform((display_x, display_y))

            # Retrieve custom offsets for this inset, defaulting if not provided
            # This allows you to fine-tune label positions for specific insets
            x_offset = meta.get('x_offset', 0.0) # Default to 0.0 if not specified
            y_offset = meta.get('y_offset', 0.005) # Default to a small vertical offset if not specified

            # Place the text using figure coordinates and custom offsets
            fig.text(figure_x + x_offset,
                     figure_y + y_offset,
                     meta['title'],
                     ha='center', va='bottom', fontsize=9)

    print("Maps generated.")
    return fig

def create_bivariate_legend(fig, darker_colors_matrix, claim_labels, policy_labels):
    """
    Creates and adds the bivariate legend to the figure.
    """
    # Position the legend more precisely
    # [left, bottom, width, height] in figure coordinates
    legend_ax = fig.add_axes([0.85, 0.2, 0.12, 0.15]) # Adjusted position for better fit

    # Draw colored squares for each bin combination
    for i in range(len(claim_labels)):
        for j in range(len(policy_labels)):
            legend_ax.axvspan(j / len(policy_labels), (j + 1) / len(policy_labels),
                              ymin=i / len(claim_labels), ymax=(i + 1) / len(claim_labels),
                              color=darker_colors_matrix[i][j])

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off') # Hide default axes ticks and labels

    # Grid lines for the legend
    for j in range(len(policy_labels) + 1):
        x = j / len(policy_labels)
        legend_ax.plot([x, x], [0, 1], color='black', lw=1)
    for i in range(len(claim_labels) + 1):
        y = i / len(claim_labels)
        legend_ax.plot([0, 1], [y, y], color='black', lw=1)

    # Axis labels for the legend (Low/Med/High)
    coverage_labels = ['Low', 'Med', 'High']
    for j, label in enumerate(coverage_labels):
        x = (j + 0.5) / len(policy_labels)
        legend_ax.text(x, -0.05, label, ha='center', va='top', fontsize=10) # Policy labels

    claims_labels_txt = ['Low', 'Med', 'High']
    for i, label in enumerate(claims_labels_txt):
        y = (i + 0.5) / len(claim_labels)
        legend_ax.text(-0.06, y, label, ha='right', va='center', fontsize=10, rotation=90) # Claims labels

    # Overall legend titles
    legend_ax.text(0.5, -0.18, '→ Total Building\nInsurance Coverage', ha='center', va='top', fontsize=11)
    legend_ax.text(-0.2, 0.5, '→ Net Building\nPayment Amount', ha='right', va='center', fontsize=11, rotation=90)
    print("Bivariate legend created.")

def create_nodata_legend(fig, color_dict):
    """
    Creates and adds the "No Data" legend to the figure.
    """
    no_data_patch = mpatches.Patch(color=color_dict["NoData"], label="No Data (missing \nclaims or policies)")
    nodata_legend_ax = fig.add_axes([0.8, 0.05, 0.1, 0.1]) # Position below the main legend
    nodata_legend_ax.axis('off') # Hide default axes ticks and labels
    nodata_legend_ax.legend(handles=[no_data_patch], loc='center left', fontsize=14, frameon=False)
    print("No Data legend created.")


if __name__ == "__main__":
    print("Starting script...")

    # Configure paths and directories
    paths = configure_paths()

    # Define inset map parameters (bounds are [left, bottom, width, height] in figure coordinates)
    # Added 'x_offset' and 'y_offset' for fine-tuning label positions
    inset_defs = {
        # Adjusted x_offset for Alaska to move its label slightly left
        '02': dict(bounds=[0.0, 0.00, 0.5, 0.5], title="Alaska", x_offset=-0.19, y_offset=-0.18),
        '15': dict(bounds=[0.1, 0.02, 0.2, 0.4], title="Hawaii", x_offset=0.08, y_offset=-0.2),
        '72': dict(bounds=[0.3, 0.05, 0.1, 0.1], title="Puerto Rico", x_offset=0.0, y_offset=-0.005),
        '66': dict(bounds=[0.4, 0.05, 0.08, 0.08], title="Guam", x_offset=0.0, y_offset=0.005),
        '60': dict(bounds=[0.5, 0.02, 0.1, 0.2], title="American Samoa", x_offset=0.0, y_offset=-0.1),
        '69': dict(bounds=[0.9, 0.35, 0.1, 0.25], title="N. Mariana \nIslands", x_offset=-0.05, y_offset=-0.15),
        '78': dict(bounds=[0.6, 0.07, 0.08, 0.08], title="US Virgin Islands", x_offset=0.0, y_offset=0.005),
    }

    # Load data from parquet files
    claims_df, policies_df = load_data(paths['claims_fp'], paths['policies_fp'])

    # Aggregate claims and policies data by countyCode
    agg_df = aggregate_data(claims_df, policies_df)

    # Perform bivariate classification and get color mapping
    agg_df, color_dict, darker_colors_matrix, claim_labels, policy_labels = perform_bivariate_classification(agg_df)

    # Merge aggregated data with county shapefile and filter out non-contiguous states
    merged_conus, merged_nonconus = merge_and_filter_counties(agg_df, paths['counties_shp'], color_dict)

    # Plot the main CONUS map and all inset maps
    fig = plot_maps(merged_conus, merged_nonconus, inset_defs)

    # Create the bivariate legend
    create_bivariate_legend(fig, darker_colors_matrix, claim_labels, policy_labels)

    # Create the "No Data" legend
    create_nodata_legend(fig, color_dict)

    # Adjust layout to prevent elements from overlapping and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(paths['outdir'], "coverage_vs_payments_figure.png"), dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {os.path.join(paths['outdir'], 'coverage_vs_payments_figure.png')}")