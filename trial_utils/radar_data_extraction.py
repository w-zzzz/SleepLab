'''
Author: Xuelin Kong
Date: 2024-04-25 13:50:32
LastEditors: Do not edit
LastEditTime: 2024-04-26 16:02:05
FilePath: \2243dataprocessing\trial_utils\radar_data_extraction.py
'''
import numpy as np
import os
import pandas as pd
import scipy
from datetime import datetime
from mmwave.dsp.cfar import ca
from mmwave.dsp.doppler_processing import doppler_processing,doppler_resolution
from mmwave.dsp.range_processing import range_processing,range_resolution
from mmwave.dsp.noise_removal import prune_to_peaks,range_based_pruning,peak_grouping_along_doppler
from mmwave.IF_proc import IFSignalProcessor
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class RadarDataProcessor:
    def __init__(self, radar_config, csv_path,output_path=None):
        self.config=radar_config    
        self.csv_path = csv_path
        self.bin_file_info = self.read_csv_info(csv_path)
        self.rangeMin=0.6 # m
        self.rangeMax=0.9 # m
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            self.output_path=output_path

    @staticmethod
    def readDCA1000(fileName):
        if not os.path.isfile(fileName):
            raise ValueError('File not found')
        numADCBits = 16 
        numLanes = 4     
        isReal = False 
        with open(fileName, 'rb') as fid:
            adcData = np.fromfile(fid, dtype=np.int16)
            if numADCBits != 16:
                l_max = 2**(numADCBits-1)-1
                adcData[adcData>l_max]=2**numADCBits
        if isReal:
            adcData = np.reshape(adcData, (numLanes, -1))
        else:
            adcData = np.reshape(adcData, (numLanes*2, -1))
            adcData = adcData[:numLanes, :]+1j*adcData[numLanes:, :]
        return adcData
        
    def read_csv_info(self, csv_path):
        # Read and process the CSV file to get metadata for each bin file
        df = pd.read_csv(csv_path)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        return df
    
    def find_relevant_bin(self, start_datetime, end_datetime):
        # Filter DataFrame to find the bin file that covers the requested time span
        relevant_bins = self.bin_file_info[
            (self.bin_file_info['start_time'] <= start_datetime) & 
            (self.bin_file_info['end_time'] >= end_datetime)]
        return relevant_bins
   
    def extract_data_by_timestamp(self, start_datetime, end_datetime):
        relevant_bin = self.find_relevant_bin(start_datetime, end_datetime)
        print(relevant_bin)
        if relevant_bin.empty:
            raise ValueError("No bin file covers the provided time span")
        
        # Read the bin data from the relevant file
        if len(relevant_bin)==1:
            bin_data = self.readDCA1000(relevant_bin.iloc[0,0])
            
            # Calculate indexes based on data points and timestamps
            # total_seconds = (end_datetime - start_datetime).total_seconds()
            total_seconds=relevant_bin.iloc[0]['duration_seconds']
            start_idx = int(((start_datetime - relevant_bin.iloc[0]['start_time']).total_seconds() / total_seconds) * relevant_bin.iloc[0]['data_points'])
            end_idx = int(((end_datetime - relevant_bin.iloc[0]['start_time']).total_seconds() / total_seconds) * relevant_bin.iloc[0]['data_points'])
            
            return bin_data[:, start_idx:end_idx]
        
    
    def target_detection_by_frame(self,rawDataCube, method='peak',rx=0):
        
        range_re,bandwidth=range_resolution(self.config.Nadc, self.config.sample_rate, self.config.slope)
        
        # doppler_re=doppler_resolution(bandwidth, self.config['f0'], self.config['ramp_end_time'], 
        #                               self.config['idle_time'], self.config['Nchirp'], self.config['Tx'])
        
        rawDataCube=np.reshape(rawDataCube,(self.config.Tx*self.config.Rx,self.config.Nadc,self.config.Nchirp)) #(x,y,z)
        rawDataCube=np.transpose(rawDataCube, (2, 0, 1))  # (z, x, y)(128,4,256)
        range_FFT=range_processing(rawDataCube)
        dopplermap, aoa_input=doppler_processing(range_FFT,clutter_removal_enabled=True,num_tx_antennas=1,accumulate=False)        

        if method == 'CFAR':
            detected_indices=ca(dopplermap, l_bound=10, guard_len=10, noise_len=8)
            target_candidates=np.abs(dopplermap)*detected_indices
            # ignore data out of range
            if self.rangeMin is not None and self.rangeMax is not None:
                target_candidates[:int(self.rangeMin/range_re),:] = 0  
                target_candidates[int(self.rangeMax/range_re):,:] = 0 
            peak = np.max(target_candidates)
            indices = np.where(dopplermap== peak)
            row=indices[0] 
            col=indices[1]

        elif method == 'peak':
            dopplermap=np.abs(dopplermap)
            if self.rangeMin is not None and self.rangeMax is not None:
                dopplermap[:int(self.rangeMin/range_re),:] = 0  
                dopplermap[int(self.rangeMax/range_re):,:] = 0  
            peak = np.max(dopplermap)
            indices = np.where(dopplermap== peak)
            if indices[0].size > 1:
                row = indices[0][0]
                col = indices[0][1]
            else:
                row = indices[0]
                col = indices[1]
        else:
            raise ValueError("Unsupported method")
        idata=rawDataCube[col,rx,row].real
        qdata=rawDataCube[col,rx,row].imag
        return idata, qdata, [col, row]
        
    
    def target_detection(self, raw_data, method='peak', tracking=False):
        """
        Apply target detection algorithms to radar data.
        
        Args:
        iq_data (numpy.ndarray): The IQ data array from radar.
        method (str): The detection method to use ('CFAR', 'peak').
        
        Returns:
        targets (numpy.ndarray): Detected targets' information.
        """
        # Trimming the data in order to keep only the complete frame
        Nframe = int(np.floor(raw_data.shape[1] / (self.config.Nadc * self.config.Nchirp)))
        NDataPoints = Nframe * self.config.Nadc * self.config.Nchirp
        trimmed_rawData = raw_data[:, :NDataPoints]
        # rawData_reshaped = np.reshape(trimmed_rawData, (self.config['Rx']*self.config['Tx'], self.config['Nadc'],-1))
        Idata=np.array([])
        Qdata=np.array([])
        if tracking:
            for i in range(0,Nframe):
                cdata = trimmed_rawData[:,  i*self.config.Nchirp*self.config.Nadc:(i+1)*self.config.Nchirp*self.config.Nadc].flatten()
                i_data,q_data,_=self.target_detection_by_frame(cdata,method)
                Idata=np.append(Idata,i_data)
                Qdata=np.append(Qdata,q_data)
        else:
            cdata = trimmed_rawData[:,  :self.config.Nchirp*self.config.Nadc].flatten() #only 1st frame
            i_data,q_data,idx=self.target_detection_by_frame(cdata,method)
            Idata=np.append(Idata,i_data)
            Qdata=np.append(Qdata,q_data)
            for i in range(1,Nframe): 
                cdata = trimmed_rawData[:, i*self.config.Nchirp*self.config.Nadc:(i+1)*self.config.Nchirp*self.config.Nadc].flatten()
                rawDataCube=np.reshape(cdata,(self.config.Tx*self.config.Rx,self.config.Nadc,self.config.Nchirp)) #(x,y,z)
                rawDataCube=np.transpose(rawDataCube, (2, 0, 1))  # (z, x, y)(128,4,256)
                Idata=np.append(Idata,rawDataCube[idx[0],0,idx[1]].real)
                Qdata=np.append(Qdata,rawDataCube[idx[0],0,idx[1]].imag)



        def butter_lowpass(cutoff, fs, order=5):
            print('fs=',fs)
            nyq = 0.5 * fs  # 奈奎斯特频率
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a
        
        def lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = filtfilt(b, a, data)
            return y

        # # 设置滤波器参数
        # cutoff = 10  # 截止频率 (Hz)
        # order = 5    # 滤波器阶数
        # # 对信号进行低通滤波
        # Idata = lowpass_filter(Idata, cutoff, 1e3/self.config.Period, order)
        # Qdata = lowpass_filter(Qdata, cutoff, 1e3/self.config.Period, order)

        print(self.output_path )
        if self.output_path is not None:
            # Stack the arrays horizontally
            combined_data = np.column_stack((Idata, Qdata))
            # Write to a text file
            np.savetxt(os.path.join(self.output_path,'IFdata.txt'), combined_data, header='Idata Qdata', fmt='%s', comments='')

        return Idata,Qdata

def main():
    # Example usage:
    Args = {
        'f0':77,                    # Hz
        'ADCStarttime': 6,    # us
        'slope': 29.982,            # MHz/us
        'idle_time': 100,       # us
        'ramp_end_time': 60,    # us
        'Nadc': 256,             # samples per chirp
        'sample_rate': 10000,    # samples per second (ksps)
        'Rx': 4,                      # Number of RX channels
        'Tx': 1,                      # Number of TX channels
        'Nchirp': 128,                # Number of chirps per frame
        'Period': 40            # ms
    }# All transferred to s
    output_path='../2024-01-11/trial1/bin_files_time_info.csv'
    processor = RadarDataProcessor(Args,output_path,output_path='../2024-01-11/trial1/')
    start_time = datetime(2024, 1, 11, 14, 31, 55)
    end_time = datetime(2024, 1, 11, 14, 32, 15)
    radar_rawdata = processor.extract_data_by_timestamp(start_time, end_time)
    print(radar_rawdata.shape)
    Idata,Qdata=processor.target_detection(raw_data=radar_rawdata)
    plt.subplot(211)
    plt.plot(Idata)
    plt.subplot(212)
    plt.plot(Qdata)
    plt.show()
    
    iq_data=Idata + 1j*Qdata
    IFprocessor = IFSignalProcessor(iq_data, period=40e-3, sampling_interval=1/0.04, plot_enabled=True)
    phase_data = IFprocessor.phase_unwrapping()
    frequency, magnitude = IFprocessor.fft_of_signal(phase_data)
    filtered_signal = IFprocessor.lowpass_filter(phase_data)
    smoothed_signal = IFprocessor.smooth_signal(filtered_signal)

if __name__ == "__main__":
    main()