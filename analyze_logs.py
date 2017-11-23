import sqlite3
import sys
import os
import pandas as pd
import numpy as np
from dateutil import parser
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import shapely.geometry as sgeom
import matplotlib.colorbar
import urllib.request
import pathlib

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

def get_n1mm_logs_in_path(input_path):
    """
    Get all N1MM logs in all subfolders, subsubfolders, ... of the input path.

    \param station_call Station callsign
    \param input_path Root path for search
    \\return Logs as a pandas dataframe
    """
    qso_data = pd.DataFrame()

    #collect all .s3db-files containing ham logs into a large pandas dataframe
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            if filename.endswith(('.s3db')):
                conn = sqlite3.connect(os.path.join(root, filename))

                #check whether this is a valid log by checking for the DXLOG table
                if 'DXLOG' in ';;;'.join([t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")]):
                    table_data = pd.read_sql_query("SELECT * FROM DXLOG", conn)
                    qso_data = qso_data.append(table_data, ignore_index=True)
    qso_data = qso_data[qso_data.IsOriginal == 1]
    return qso_data

def qsos_per_operator_pie_chart(qso_data, filename):
    """
    Plot the number of QSOs per operator as a pie diagram.

    \param qso_data QSO log data
    \param filename Output filename of plot
    """
    #get number of QSOs per operator
    operators_all, counts_all = np.unique(qso_data.Operator, return_counts=True)

    #squash operators that have a small percentage of the QSO total to a single count
    percentages = counts_all/np.sum(counts_all)
    perc_thresh = 0.01
    counts = counts_all[percentages > perc_thresh]
    operators = operators_all[percentages > perc_thresh]
    counts = np.append(counts, np.sum(counts_all[percentages <= perc_thresh]))
    operators = np.append(operators, 'Others')
    operator_others = 'Others:\n' + ', '.join([str(n) + " (" + str(m) + ")" for m,n in zip(counts_all[percentages <= perc_thresh], operators_all[percentages <= perc_thresh])])
    percentages = counts/np.sum(counts)

    explode = 0.1*np.exp(percentages)
    explode[percentages > 0.1] = 0

    #plot
    plt.pie(counts, labels=operators, explode=explode, autopct=lambda p: '{:.0f} qsos'.format(p * np.sum(counts) / 100) if p > 1 else '')
    plt.text(-0.04, 0.0,operator_others, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes, wrap=True)
    plt.savefig(filename)
    plt.clf()

def qso_frequency_per_hour(qso_data, filename):
    """
    Plot the number of QSOs as a function of the time of the day.

    \param qso_data QSO log data
    \param filename Output filename of plot
    """
    timestamps = pd.DatetimeIndex(qso_data.TS)
    plt.hist(timestamps.hour + timestamps.minute/60.0, bins=100)
    plt.xlabel('Hour of day')
    plt.ylabel('Number of QSOs')
    plt.savefig(filename)
    plt.clf()

def qso_frequency_per_hour_per_operator(qso_data, filename):
    """
    Plot the number of QSOs as a function of the time of the day for the various operators in the log.

    \param qso_data QSO log data
    \param filename Output filename of plot
    """
    operators, counts = np.unique(qso_data.Operator, return_counts=True)
    operators = operators[counts/np.sum(counts) > 0.05]
    for operator in operators:
        timestamps = pd.DatetimeIndex(qso_data[qso_data.Operator == operator].TS)
        timestamps = timestamps.hour + timestamps.minute/60.0
        binwidth = 0.25
        plt.hist(timestamps, bins=np.arange(min(timestamps), max(timestamps) + binwidth, binwidth), label=operator, alpha=0.8)
    plt.legend()
    plt.xlabel('Hour of day')
    plt.ylabel('Number of QSOs')
    plt.savefig(filename)
    plt.clf()

def qso_frequency_per_day(qso_data, filename):
    """
    Plot the number of QSOs as a function of the day since the first log entry.
    Mark with start of UKA and CQWWRTTY.

    \param qso_data QSO log data
    \param filename Output filename of plot
    """
    timestamps = pd.DatetimeIndex(qso_data.TS)
    start_date = np.min(timestamps)
    timestamps = timestamps - start_date
    timestamps = timestamps.components
    timestamps = timestamps.days + timestamps.hours/24.0 + timestamps.minutes/(60.0*24.0)

    plt.hist(timestamps, bins=50)
    plt.xlabel('Day number')
    plt.ylabel('Number of QSOs')

    #annotate with start day of UKA
    uka_start_day = (parser.parse('2017-10-04') - start_date).days
    ax = plt.axes()
    ax.annotate('UKA began', xy=(uka_start_day, 0), xytext=(uka_start_day+10, 210), arrowprops=dict(facecolor='black', shrink=0.05))

    #annotate with date for CQWWRTTY
    cqwwrtty_day = (parser.parse('2017-09-23') - start_date).days
    ax.annotate('CQWWRTTY', xy=(cqwwrtty_day, 0), xytext=(cqwwrtty_day+10, 300), arrowprops=dict(facecolor='black', shrink=0.05))

    plt.savefig(filename)
    plt.clf()

def qso_frequency_per_day_per_operator(qso_data, filename):
    """
    Plot the number of QSOs as a function of the day since the first log entry,
    for each operator in the log.

    \param qso_data QSO log data
    \param filename Output filename of plot
    """
    operators, counts = np.unique(qso_data.Operator, return_counts=True)
    operators = operators[counts/np.sum(counts) > 0.01]
    for operator in operators:
        timestamps = pd.DatetimeIndex(qso_data[qso_data.Operator == operator].TS)
        start_date = np.min(timestamps)
        timestamps = timestamps - start_date
        timestamps = timestamps.components
        timestamps = timestamps.days + timestamps.hours/24.0 + timestamps.minutes/(60.0*24.0)
        binwidth = 1
        plt.hist(timestamps, bins=np.arange(min(timestamps), max(timestamps) + binwidth, binwidth), label=operator, alpha=0.8)

    plt.legend()
    plt.xlabel('Day number')
    plt.ylabel('Number of QSOs')
    plt.savefig(filename)
    plt.clf()

def qso_frequency_per_hour_per_band(qso_data, filename):
    """
    Plot the number of QSOs as a function of the time of the day, for each frequency band in the log.

    \param qso_data QSO log data
    \param filename Output filename of plot
    """
    for band in qso_data.Band.unique():
        timestamps = pd.DatetimeIndex(qso_data[qso_data.Band == band].TS)
        timestamps = timestamps.hour + timestamps.minute/60.0
        binwidth = 0.25
        plt.hist(timestamps, bins=np.arange(min(timestamps), max(timestamps) + binwidth, binwidth), label=str(band) + ' MHz', alpha=0.8)

    plt.legend()
    plt.xlabel('Hour of day')
    plt.ylabel('Number of QSOs')
    plt.savefig(filename)
    plt.clf()

def get_itu_prefix_country_mapping():
    """
    Get ITU mapping of callsign prefix to country name. Will, unless file exists,
    obtain the current table from ITU website (URL could be subject to change,
    could fail) and download it to the current directory.

    \\return ITU mapping from callsign prefix to country name as a pandas dataframe
    """
    #check if ITU table has been downloaded, download if not
    itu_mapping_filename = 'CallSigns'
    if not pathlib.Path(itu_mapping_filename).is_file():
        itu_url = 'https://www.itu.int/gladapp/Allocation/CallSigns'
        urllib.request.urlretrieve(itu_url, itu_mapping_filename)

    #read ITU HTML table into pandas dataframe and do some massaging
    with open(itu_mapping_filename, "r") as itu_mapping_file:
        itu_mapping = itu_mapping_file.read()
    itu_mapping = pd.read_html(itu_mapping)[0]
    itu_mapping.columns = ['raw_prefix', 'country']
    itu_mapping.country = itu_mapping.country.str.replace('(.*)\\s+\((.*)\)', '\\2 \\1') #switch from e.g. `Netherlands (Kingdom of the)` to `Kingdom of the Netherlands`

    #split prefix column to multiple prefixes for each country
    prefix = itu_mapping.raw_prefix.str.replace('(\S*) - (\S*)', '\\1-\\2').str.split('  ').apply(pd.Series, 1).stack()
    prefix.name = 'prefix'
    itu_mapping = itu_mapping.reindex(index=prefix.index, level=0).join(prefix, how='inner')[['country', 'prefix']].reset_index(drop=True)
    return itu_mapping

def country_from_callsign(itu_mapping, callsign):
    """
    Convert callsign to a country name.

    \param itu_mapping ITU mapping of prefix to country
    \param callsign Callsign
    \\return Country name
    """
    if len(callsign) <= 0:
        return str(None)

    #should be able to get a unique match from the first two letters of the callsign only
    country_candidates = np.unique(itu_mapping.iloc[itu_mapping.prefix[itu_mapping.prefix.str.match(callsign[0:2])].index].country.values)
    if (len(country_candidates) != 1):
        return str(None)
    return country_candidates[0]

def qso_country_map(qso_data, filename):
    """
    Mark on a world map which countries are present in the log.

    \param qso_data QSO log data
    \param filename Output filename of plot
    """
    itu_mapping = get_itu_prefix_country_mapping()

    #find country names from the prefixes in the logs
    run_countries = qso_data.CountryPrefix.unique()
    i=0
    for prefix in run_countries:
        run_countries[i] = country_from_callsign(itu_mapping, prefix)
        i += 1
    run_countries = np.unique(run_countries)

    #prepare world map and country data
    ax = plt.axes([0, 0, 1, 1],
                  projection=ccrs.Mercator())
    shapefilename = shpreader.natural_earth(resolution='110m',
                                         category='cultural', name='admin_0_countries')
    reader = shpreader.Reader(shapefilename)
    countries = list(reader.records())
    for country in countries:
        formal_name = country.attributes['FORMAL_EN']
        long_name = country.attributes['NAME_LONG']
        geometry = country.geometry
        facecolor = [0.9375, 0.9375, 0.859375]
        edgecolor = 'black'
        run_country_color = [0.5375, 0.9375, 0.659375]

        #check if country exists in the logs, fill country with different color if it does
        if formal_name in run_countries or long_name in run_countries:
            facecolor = run_country_color

        #add country to map
        ax.add_geometries([geometry], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor=edgecolor)

    plt.savefig(filename)


def main():
    if len(sys.argv) < 2:
        print('Import all .s3db-files containing N1MM ham radio logs into a pandas DataFrame and plot statistics. Will walk through entire folder structure contained in input path.')
        print('')
        print('Usage: ' + sys.argv[0] + ' [path to folder] [Optional: [station call sign]]')
        exit()

    input_path = sys.argv[1]
    station_callsign = ''
    if len(sys.argv) > 2:
        station_callsign = sys.argv[2]

    #get ham radio logs
    qso_data = get_n1mm_logs_in_path(input_path)

    #filter on input station callsign
    if len(station_callsign) > 0:
        qso_data = qso_data[qso_data.StationPrefix == 'LM100UKA']

    #generate plots
    qsos_per_operator_pie_chart(qso_data, 'operator_qsos.png')
    qso_frequency_per_hour(qso_data, 'hour_of_day.png')
    qso_frequency_per_hour_per_operator(qso_data, 'hour_of_day_split_on_operator.png')
    qso_frequency_per_day(qso_data, 'day_of_uka.png')
    qso_frequency_per_day_per_operator(qso_data, 'day_of_uka_split_on_operator.png')
    qso_frequency_per_hour_per_band(qso_data, 'hour_of_day_split_on_band.png')
    qso_country_map(qso_data, 'country_map.png')

if __name__ == "__main__":
    main()
