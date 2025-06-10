#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 03 18:00:00 2025
@author: cdeval
"""
import duckdb
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import numpy as np
from datetime import date
import cpi

# --- Helper Functions (Copied for Self-Containment) ---

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
    Includes all relevant payment amounts from claims.
    """
    print("Loading data...")
    try:
        con = duckdb.connect()
        claims_df = con.execute(
            f"""
            SELECT
                id, countyCode, yearOfLoss, floodEvent,
                amountPaidOnBuildingClaim,
                amountPaidOnContentsClaim,
                amountPaidOnIncreasedCostOfComplianceClaim,
                netBuildingPaymentAmount
            FROM read_parquet('{claims_fp}')
            """
        ).df()
        policies_df = con.execute(
            f"SELECT countyCode, policyCount, totalBuildingInsuranceCoverage FROM read_parquet('{policies_fp}')"
        ).df()
        con.close()
        print("Data loaded successfully.")
        return claims_df, policies_df
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure your .parquet files exist in the specified 'data' directory and contain the necessary columns.")
        exit()

def aggregate_data(claims_df, policies_df):
    """
    Aggregates claims and policies data by countyCode.
    Note: 'claims_df' here should already be filtered for the events you want to plot.
    """
    claims_agg = (
        claims_df.dropna(subset=['countyCode'])
        .groupby('countyCode')
        .agg(
            amountPaidOnBuildingClaim_sum=('amountPaidOnBuildingClaim', 'sum'),
            amountPaidOnContentsClaim_sum=('amountPaidOnContentsClaim', 'sum'),
            amountPaidOnIncreasedCostOfComplianceClaim_sum=('amountPaidOnIncreasedCostOfComplianceClaim', 'sum')
        )
        .reset_index()
    )
    claims_agg['avg_claim_amt'] = (
        claims_agg['amountPaidOnBuildingClaim_sum'] +
        claims_agg['amountPaidOnContentsClaim_sum'] +
        claims_agg['amountPaidOnIncreasedCostOfComplianceClaim_sum']
    )
    claims_agg = claims_agg.drop(columns=[
        'amountPaidOnBuildingClaim_sum',
        'amountPaidOnContentsClaim_sum',
        'amountPaidOnIncreasedCostOfComplianceClaim_sum'
    ])

    policies_agg = (
        policies_df.dropna(subset=['countyCode'])
        .groupby('countyCode')
        .agg(total_policies=('policyCount', 'sum'),
             avg_coverage=('totalBuildingInsuranceCoverage', 'sum'))
        .reset_index()
    )

    agg_df = pd.merge(claims_agg, policies_agg, on='countyCode', how='outer')
    return agg_df

def perform_bivariate_classification(agg_df):
    """
    Performs bivariate classification and assigns colors based on claim and policy bins.
    """
    agg_df['claim_bin'] = pd.Series(dtype='str')
    agg_df['policy_bin'] = pd.Series(dtype='str')

    valid_claims = agg_df['avg_claim_amt'].notna()
    valid_policies = agg_df['avg_coverage'].notna()

    if agg_df.loc[valid_claims, 'avg_claim_amt'].nunique() >= 3:
        agg_df.loc[valid_claims, 'claim_bin'] = pd.qcut(
            agg_df.loc[valid_claims, 'avg_claim_amt'], 3, labels=["1", "2", "3"], duplicates='drop'
        ).astype(str)
    elif agg_df.loc[valid_claims, 'avg_claim_amt'].nunique() >= 2:
        agg_df.loc[valid_claims, 'claim_bin'] = pd.qcut(
            agg_df.loc[valid_claims, 'avg_claim_amt'], 2, labels=["1", "2"], duplicates='drop'
        ).astype(str)
    else:
        agg_df.loc[valid_claims, 'claim_bin'] = "1"

    if agg_df.loc[valid_policies, 'avg_coverage'].nunique() >= 3:
        agg_df.loc[valid_policies, 'policy_bin'] = pd.qcut(
            agg_df.loc[valid_policies, 'avg_coverage'], 3, labels=["A", "B", "C"], duplicates='drop'
        ).astype(str)
    elif agg_df.loc[valid_policies, 'avg_coverage'].nunique() >= 2:
        agg_df.loc[valid_policies, 'policy_bin'] = pd.qcut(
            agg_df.loc[valid_policies, 'avg_coverage'], 2, labels=["A", "B"], duplicates='drop'
        ).astype(str)
    else:
        agg_df.loc[valid_policies, 'policy_bin'] = "A"

    agg_df['claim_bin'] = agg_df['claim_bin'].fillna('')
    agg_df['policy_bin'] = agg_df['policy_bin'].fillna('')
    agg_df['Bi_Class'] = agg_df['claim_bin'] + agg_df['policy_bin']
    agg_df.loc[agg_df['Bi_Class'] == '', 'Bi_Class'] = 'NoData'

    claim_labels = sorted(agg_df['claim_bin'].loc[agg_df['claim_bin'] != ''].unique())
    policy_labels = sorted(agg_df['policy_bin'].loc[agg_df['policy_bin'] != ''].unique())

    all_possible_colors = [
        ["#e8e8e8", "#ace4e4", "#5ac8c8"],
        ["#dfb0d6", "#a5add3", "#5694c0"],
        ["#be64ac", "#8c62aa", "#3b4994"],
    ]

    claim_map = {"1": 0, "2": 1, "3": 2}
    policy_map = {"A": 0, "B": 1, "C": 2}

    color_dict = {}
    used_colors_matrix = []

    for c_label_idx, c_label in enumerate(claim_labels):
        row_colors = []
        for p_label_idx, p_label in enumerate(policy_labels):
            try:
                row_idx_in_full_matrix = claim_map[c_label]
                col_idx_in_full_matrix = policy_map[p_label]
                color = all_possible_colors[row_idx_in_full_matrix][col_idx_in_full_matrix]
                color_dict[c_label + p_label] = color
                row_colors.append(color)
            except KeyError:
                color_dict[c_label + p_label] = "#D3D3D3"
                row_colors.append("#D3D3D3")
        if row_colors:
            used_colors_matrix.append(row_colors)

    color_dict["NoData"] = "#eddcd2"
    agg_df['color'] = agg_df['Bi_Class'].map(color_dict)
    return agg_df, color_dict, used_colors_matrix, claim_labels, policy_labels


def merge_and_filter_counties(agg_df, counties_shp, color_dict):
    """
    Merges aggregated data with county shapefile and filters out non-contiguous states.
    Fills 'color' column with 'NoData' color for counties not found in agg_df.
    """
    try:
        counties = gpd.read_file(counties_shp)
    except Exception as e:
        print(f"Error loading county shapefile: {e}")
        print("Please ensure the 'tl_2024_us_county.shp' file and its companions (.dbf, .shx, etc.) are in the specified directory.")
        exit()

    agg_df['countyCode'] = agg_df['countyCode'].astype(str).str.zfill(5)

    merged = counties.merge(agg_df, left_on='GEOID', right_on='countyCode', how='left')

    merged['color'] = merged['color'].fillna(color_dict["NoData"])
    merged['Bi_Class'] = merged['Bi_Class'].fillna('NoData')

    excluded_fps = ['02', '15', '72', '66', '60', '69', '78']
    merged_conus = merged[~merged['STATEFP'].isin(excluded_fps)]
    merged_nonconus = merged[merged['STATEFP'].isin(excluded_fps)]
    return merged_conus, merged_nonconus

def plot_single_frame(ax_main, ax_insets, merged_conus, merged_nonconus, inset_defs, title_suffix, color_dict_no_data_key):
    """
    Plots a single frame of the map.
    `color_dict_no_data_key` is passed instead of `color_dict` to ensure the 'NoData' color is accessible.
    """
    ax_main.clear()
    for ax_inset in ax_insets.values():
        ax_inset.clear()

    merged_conus.plot(ax=ax_main, color=merged_conus['color'], edgecolor='white', linewidth=0.4)
    ax_main.axis("off")

    for statefp, meta in inset_defs.items():
        ax_inset = ax_insets[statefp]
        subset = merged_nonconus[merged_nonconus['STATEFP'] == statefp]

        if not subset.empty:
            subset.plot(ax=ax_inset, color=subset['color'], edgecolor='white', linewidth=0.3)
            ax_inset.axis("off")

            x_offset = meta.get('x_offset', 0.0)
            y_offset = meta.get('y_offset', 0.005)
            ax_inset.text(0.5 + x_offset, 1.0 + y_offset, meta['title'],
                          ha='center', va='bottom', fontsize=9, transform=ax_inset.transAxes)

    ax_main.set_title(title_suffix, fontsize=16) # Adjusted y for title position


def create_bivariate_legend(fig, darker_colors_matrix, claim_labels, policy_labels):
    """
    Creates and adds the bivariate legend to the figure.
    """
    legend_ax = fig.add_axes([0.85, 0.2, 0.12, 0.15])

    for i in range(len(claim_labels)):
        for j in range(len(policy_labels)):
            row_idx = i
            col_idx = j

            color_to_use = '#D3D3D3'
            if darker_colors_matrix and row_idx < len(darker_colors_matrix) and col_idx < len(darker_colors_matrix[row_idx]):
                color_to_use = darker_colors_matrix[row_idx][col_idx]
            else:
                pass

            legend_ax.axvspan(j / len(policy_labels), (j + 1) / len(policy_labels),
                              ymin=i / len(claim_labels), ymax=(i + 1) / len(claim_labels),
                              color=color_to_use)

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')

    for j in range(len(policy_labels) + 1):
        x = j / len(policy_labels)
        legend_ax.plot([x, x], [0, 1], color='black', lw=1)
    for i in range(len(claim_labels) + 1):
        y = i / len(claim_labels)
        legend_ax.plot([0, 1], [y, y], color='black', lw=1)

    coverage_labels_map = {1: [''], 2: ['Low', 'High'], 3: ['Low', 'Med', 'High']}
    claims_labels_map = {1: [''], 2: ['Low', 'High'], 3: ['Low', 'Med', 'High']}

    coverage_labels_to_use = coverage_labels_map.get(len(policy_labels), [''] * len(policy_labels))
    claims_labels_to_use = claims_labels_map.get(len(claim_labels), [''] * len(claim_labels))


    for j, label in enumerate(coverage_labels_to_use):
        x = (j + 0.5) / len(policy_labels)
        legend_ax.text(x, -0.05, label, ha='center', va='top', fontsize=10)
    for i, label in enumerate(claims_labels_to_use):
        y = (i + 0.5) / len(claim_labels)
        legend_ax.text(-0.06, y, label, ha='right', va='center', fontsize=10, rotation=90)

    legend_ax.text(0.5, -0.18, '→ Total Building\nInsurance Coverage', ha='center', va='top', fontsize=11)
    legend_ax.text(-0.2, 0.5, '→ Total Claim\nPayment Amount', ha='right', va='center', fontsize=11, rotation=90)

def create_nodata_legend(fig, color_dict):
    """
    Creates and adds the "No Data" legend to the figure.
    """
    no_data_patch = mpatches.Patch(color=color_dict["NoData"], label="No Data (missing \nclaims or policies)")
    nodata_legend_ax = fig.add_axes([0.8, 0.05, 0.1, 0.1])
    nodata_legend_ax.axis('off')
    nodata_legend_ax.legend(handles=[no_data_patch], loc='center left', fontsize=14, frameon=False)

def filter_claims_for_plotting(claims_df, events_to_plot=None):
    """
    Filters the claims DataFrame to include only specified flood events for plotting.
    If events_to_plot is None or empty, it means no specific filtering other than floodEvent NOT NULL.
    events_to_plot should be a list of tuples: [(year, 'Event Name'), (year, 'Another Event')]
    """
    if events_to_plot:
        claims_df['floodEvent'] = claims_df['floodEvent'].astype(str)
        event_tuples_set = set(events_to_plot)
        filtered_claims = claims_df[
            claims_df.apply(lambda row: (row['yearOfLoss'], row['floodEvent']) in event_tuples_set, axis=1)
        ].copy()
    else:
        filtered_claims = claims_df[claims_df['floodEvent'].notna()].copy()

    for col in ['amountPaidOnBuildingClaim', 'amountPaidOnContentsClaim', 'amountPaidOnIncreasedCostOfComplianceClaim']:
        filtered_claims[col] = pd.to_numeric(filtered_claims[col], errors='coerce').fillna(0)

    return filtered_claims

# --- Main Script for Static Maps ---

if __name__ == "__main__":
    print("Starting static map generation script...")

    paths = configure_paths()
    claims_raw_df, policies_df = load_data(paths['claims_fp'], paths['policies_fp'])

    # Identifying Top 10 Costliest Flood Events
    print("\n--- Identifying Top 10 Costliest Flood Events ---")
    con = duckdb.connect()
    con.register('claims', claims_raw_df)

    base_event_df = con.sql("""
        SELECT
            yearOfLoss,
            floodEvent,
            COUNT(id) AS countClaims,
            ROUND(
                SUM(amountPaidOnBuildingClaim)
                + SUM(amountPaidOnContentsClaim)
                + SUM(amountPaidOnIncreasedCostOfComplianceClaim), 0
            )::BIGINT AS paidTotalClaim
        FROM claims
        WHERE floodEvent IS NOT NULL
        GROUP BY yearOfLoss, floodEvent
    """).df()
    con.unregister('claims')
    con.close()

    event_df = base_event_df.copy()

    try:
        cpi.update()
        timestamp_target = pd.Timestamp(date(2025, 1, 1))

        event_df['yearOfLossFormatted'] = pd.to_datetime(event_df['yearOfLoss'].astype(str) + "-01-01")
        event_df["paidTotalClaim2025"] = event_df.apply(
            lambda x: cpi.inflate(
                x.paidTotalClaim,
                x.yearOfLossFormatted,
                to=timestamp_target),
            axis=1)

        event_df = (
            event_df
            .assign(
                paidTotalClaimM = event_df['paidTotalClaim'] / 1_000_000,
                paidTotalClaimM2025 = event_df['paidTotalClaim2025'] / 1_000_000,
                averagePaidClaim2025 = event_df['paidTotalClaim2025'] / event_df['countClaims']
            )
            .astype({'averagePaidClaim2025':int})
            .sort_values(by='paidTotalClaim2025', ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        event_df['rank'] = range(1, 11)
        cols = ['rank', 'yearOfLoss', 'floodEvent', 'countClaims', 'paidTotalClaimM', 'paidTotalClaimM2025', 'averagePaidClaim2025']
        event_df = event_df.loc[:, cols]
        print("Top 10 Costliest Flood Events (adjusted to 2025 dollars):")
        try:
            from tabulate import tabulate
            print(tabulate(event_df, headers='keys', tablefmt='pipe', floatfmt=",.2f"))
        except ImportError:
            print("Warning: 'tabulate' library not found. Printing table without markdown formatting.")
            print(event_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    except Exception as e:
        print(f"Warning: CPI adjustment failed ({e}). Skipping inflation adjustment and using raw 'paidTotalClaim' for ranking.")
        event_df = (
            base_event_df
            .assign(
                paidTotalClaimM = base_event_df['paidTotalClaim'] / 1_000_000,
            )
            .sort_values(by='paidTotalClaim', ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        event_df['rank'] = range(1, 11)
        event_df['paidTotalClaimM2025'] = event_df['paidTotalClaimM']
        cols = ['rank', 'yearOfLoss', 'floodEvent', 'countClaims', 'paidTotalClaimM']
        event_df = event_df.loc[:, cols]
        print("Top 10 Costliest Flood Events (raw dollars, CPI adjustment failed):")
        print(event_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        event_df['averagePaidClaim2025'] = 0

    # Define inset map parameters
    inset_defs = {
        '02': dict(bounds=[0.0, 0.00, 0.5, 0.5], title="Alaska", x_offset=-0.19, y_offset=-0.18),
        '15': dict(bounds=[0.1, 0.02, 0.2, 0.4], title="Hawaii", x_offset=0.08, y_offset=-0.2),
        '72': dict(bounds=[0.3, 0.05, 0.1, 0.1], title="Puerto Rico", x_offset=0.0, y_offset=-0.005),
        '66': dict(bounds=[0.4, 0.05, 0.08, 0.08], title="Guam", x_offset=0.0, y_offset=0.005),
        '60': dict(bounds=[0.5, 0.02, 0.1, 0.2], title="American Samoa", x_offset=0.0, y_offset=-0.1),
        '69': dict(bounds=[0.9, 0.35, 0.1, 0.25], title="N. Mariana \nIslands", x_offset=-0.05, y_offset=-0.15),
        '78': dict(bounds=[0.6, 0.07, 0.08, 0.08], title="US Virgin Islands", x_offset=0.0, y_offset=0.005),
    }

    ### 1. Map for All Top 10 Events Combined
    print("\nCreating map for ALL Top 10 Events combined...")
    events_for_plotting_all = list(event_df[['yearOfLoss', 'floodEvent']].itertuples(index=False, name=None))
    plot_title_all = "All Top 10 Costliest Flood Events (Adjusted to 2025 Dollars)"
    output_file_name_all = "top_10_all_events_bivariate_map.png"

    filtered_claims_df_all = filter_claims_for_plotting(claims_raw_df, events_for_plotting_all)
    agg_df_all = aggregate_data(filtered_claims_df_all, policies_df)
    agg_df_all, color_dict_all, darker_colors_matrix_all, claim_labels_all, policy_labels_all = \
        perform_bivariate_classification(agg_df_all)
    merged_conus_all, merged_nonconus_all = \
        merge_and_filter_counties(agg_df_all, paths['counties_shp'], color_dict_all)

    fig_all = plt.figure(figsize=(16, 11))
    ax_main_all = fig_all.add_subplot(1,1,1)
    ax_insets_all = {statefp: fig_all.add_axes(meta['bounds']) for statefp, meta in inset_defs.items()}

    plot_single_frame(ax_main_all, ax_insets_all, merged_conus_all, merged_nonconus_all, inset_defs,
                      plot_title_all, color_dict_all["NoData"])
    create_bivariate_legend(fig_all, darker_colors_matrix_all, claim_labels_all, policy_labels_all)
    create_nodata_legend(fig_all, color_dict_all)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(paths['outdir'], output_file_name_all), dpi=300, bbox_inches='tight')
    plt.close(fig_all)
    print(f"Figure saved to: {os.path.join(paths['outdir'], output_file_name_all)}")

    ### 2. Individual Maps for Each of the Top 10 Events (Static PNGs)
    print("\nCreating individual maps for each of the Top 10 Events...")
    for index, row in event_df.iterrows():
        event_year = row['yearOfLoss']
        event_name = row['floodEvent']
        event_rank = row['rank']

        print(f"  Processing event {event_rank}: {event_name} ({event_year})...")

        events_for_plotting_single = [(event_year, event_name)]
        plot_title_single = f"{event_name} ({event_year})"
        sanitized_event_name = "".join(x for x in event_name if x.isalnum() or x.isspace()).strip().replace(" ", "_")
        output_file_name_single = f"event_{event_rank}_{event_year}_{sanitized_event_name}_bivariate_map.png"

        filtered_claims_df_single = filter_claims_for_plotting(claims_raw_df, events_for_plotting_single)

        if not filtered_claims_df_single.empty:
            agg_df_single = aggregate_data(filtered_claims_df_single, policies_df)

            if agg_df_single['avg_claim_amt'].sum() == 0 and agg_df_single['avg_coverage'].sum() == 0:
                print(f"    No significant claim or policy data for {event_name} ({event_year}). Skipping static map.")
                continue

            agg_df_single, color_dict_single, darker_colors_matrix_single, claim_labels_single, policy_labels_single = \
                perform_bivariate_classification(agg_df_single)
            merged_conus_single, merged_nonconus_single = \
                merge_and_filter_counties(agg_df_single, paths['counties_shp'], color_dict_single)

            fig_single = plt.figure(figsize=(16, 11))
            ax_main_single = fig_single.add_subplot(1,1,1)
            ax_insets_single = {statefp: fig_single.add_axes(meta['bounds']) for statefp, meta in inset_defs.items()}

            plot_single_frame(ax_main_single, ax_insets_single, merged_conus_single, merged_nonconus_single, inset_defs,
                              plot_title_single, color_dict_single["NoData"])
            create_bivariate_legend(fig_single, darker_colors_matrix_single, claim_labels_single, policy_labels_single)
            create_nodata_legend(fig_single, color_dict_single)

            plt.tight_layout()
            plt.savefig(os.path.join(paths['outdir'], output_file_name_single), dpi=300, bbox_inches='tight')
            plt.close(fig_single)
            print(f"    Figure saved to: {os.path.join(paths['outdir'], output_file_name_single)}")
        else:
            print(f"    No claims data found for {event_name} ({event_year}). Skipping static map.")

    print("\nStatic map generation finished.")