u'''General informations'''
__author__ = "Mousavi S.M."
r"""Load STEAD data updated dataset"""
u'''Required modules'''
import os
# data selection part
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
# convert waveforms part
from convert_waveforms import make_stream
from convert_waveforms import make_plot
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client

from fractions import gcd

def data_selector(csv_file,file_name):

    # reading the csv file into a dataframe:
    df = pd.read_csv(csv_file)
    print(f'total events in csv file: {len(df)}')
    # filterering the dataframe
    df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 15) & (df.source_magnitude < 4.5) & (4 < df.source_magnitude)]
    print(f'total events selected: {len(df)}')

    # making a list of trace names for the selected data
    ev_list = df['trace_name'].to_list()

    # retrieving selected waveforms from the hdf5 file: 
    dtfl = h5py.File(file_name, 'r')
    for c, evi in enumerate(ev_list):
        dataset = dtfl.get('data/'+str(evi)) 
        # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel 
        data = np.array(dataset)

        # convert waveforms part ##################################################################################################################
        # downloading the instrument response of the station from IRIS
        client = Client("IRIS")
        inventory = client.get_stations(network=dataset.attrs['network_code'],
                                        station=dataset.attrs['receiver_code'],
                                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                                        loc="*",
                                        channel="*",
                                        level="response")

        # # converting into displacement
        # st = make_stream(dataset)
        # st = st.remove_response(inventory=inventory, output="DISP", plot=False)
        # # converting into velocity
        # st = make_stream(dataset)
        # st = st.remove_response(inventory=inventory, output='VEL', plot=False)
        # converting into acceleration
        st = make_stream(dataset)
        st.remove_response(inventory=inventory, output="ACC", plot=False)
        acc_for_loading_1 = st[2].data
        acc_for_loading_1 = np.float32(acc_for_loading_1)

        acc_for_loading_1  = np.expand_dims(acc_for_loading_1, axis=1)

        if c == 0:
            acc_for_loading  = acc_for_loading_1
        else:
            acc_for_loading  = np.concatenate((acc_for_loading, acc_for_loading_1), axis=1)
        # end convert waveforms part ##############################################################################################################
    return acc_for_loading
        



def data_selector_with_plot(csv_file,file_name):

    # reading the csv file into a dataframe:
    df = pd.read_csv(csv_file)
    print(f'total events in csv file: {len(df)}')
    # filterering the dataframe
    df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
    print(f'total events selected: {len(df)}')

    # making a list of trace names for the selected data
    ev_list = df['trace_name'].to_list()

    # retrieving selected waveforms from the hdf5 file: 
    dtfl = h5py.File(file_name, 'r')
    for c, evi in enumerate(ev_list):
        dataset = dtfl.get('data/'+str(evi)) 
        # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel 
        data = np.array(dataset)

        fig = plt.figure()
        ax = fig.add_subplot(311)         
        plt.plot(data[:,0], 'k')
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight':'bold'}    
        plt.tight_layout()
        ymin, ymax = ax.get_ylim()
        pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
        sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
        cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
        plt.legend(handles=[pl, sl, cl], loc = 'upper right', borderaxespad=0., prop=legend_properties)        
        plt.ylabel('Amplitude counts', fontsize=12) 
        ax.set_xticklabels([])

        ax = fig.add_subplot(312)         
        plt.plot(data[:,1], 'k')
        plt.rcParams["figure.figsize"] = (8, 5)
        legend_properties = {'weight':'bold'}    
        plt.tight_layout()
        ymin, ymax = ax.get_ylim()
        pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
        sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
        cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
        plt.legend(handles=[pl, sl, cl], loc = 'upper right', borderaxespad=0., prop=legend_properties)        
        plt.ylabel('Amplitude counts', fontsize=12) 
        ax.set_xticklabels([])

        ax = fig.add_subplot(313)         
        plt.plot(data[:,2], 'k')
        plt.rcParams["figure.figsize"] = (8,5)
        legend_properties = {'weight':'bold'}    
        plt.tight_layout()
        ymin, ymax = ax.get_ylim()
        pl = plt.vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
        sl = plt.vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
        cl = plt.vlines(dataset.attrs['coda_end_sample'], ymin, ymax, color='aqua', linewidth=2, label='Coda End')
        plt.legend(handles=[pl, sl, cl], loc = 'upper right', borderaxespad=0., prop=legend_properties)        
        plt.ylabel('Amplitude counts', fontsize=12) 
        ax.set_xticklabels([])
        plt.show() 

        for at in dataset.attrs:
            print(at, dataset.attrs[at])    


        # convert waveforms part ##################################################################################################################
        # downloading the instrument response of the station from IRIS
        client = Client("IRIS")
        inventory = client.get_stations(network=dataset.attrs['network_code'],
                                        station=dataset.attrs['receiver_code'],
                                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                                        loc="*", 
                                        channel="*",
                                        level="response")  

        # converting into displacement
        st = make_stream(dataset)
        st = st.remove_response(inventory=inventory, output="DISP", plot=False)
        # ploting the verical component
        make_plot(st[2], title='Displacement', ylab='meters')
        # converting into velocity
        st = make_stream(dataset)
        st = st.remove_response(inventory=inventory, output='VEL', plot=False)
        # ploting the verical component
        make_plot(st[2], title='Velocity', ylab='meters/second')
        # converting into acceleration
        st = make_stream(dataset)
        st.remove_response(inventory=inventory, output="ACC", plot=False) 
        # ploting the verical component
        make_plot(st[2], title='Acceleration', ylab='meters/second**2')

        inp = input("Press a key to plot the next waveform!")
        if inp == "r":
            continue
        # end convert waveforms part ##############################################################################################################