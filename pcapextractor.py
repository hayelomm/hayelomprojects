import dpkt
from scapy.all import *
import csv
import numpy as np
import pandas as pd

Path = "Benign/"

def extract_pcaps_files(pcap_dir):
    
    ## Analyze the PCAP files
    for root, dirs, files in os.walk(pcap_dir):
        
        ## for each file in the directory
        for filename in files:

            ## calculate the pcap file including its path
            pcap_file = pcap_dir  + filename

            ## analyze the pcap file
            pcapToCsv(pcap_file)

    return 0

def pcapToCsv(fname):
    #f = open('SynFlood_Sample.pcap')
    pkts=rdpcap(fname)
    #pcap = dpkt.pcap.Reader(f)
    outfile=fname+'.csv'
    csvfile=fname+'.csv'
    with open(outfile, 'a', newline='') as csvfile:
        fieldnames = ['IP src', 'IP dst','sport', 'dport', 'len', 'Iflags', 'frag',
                      'ttl','Ichksum','seq', 'ack', 'dataofs', 
                      'Tflags','window', 'Tchksum','Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
    
        for pkt in pkts:
            #pkt.show()
            if pkt.haslayer(TCP):
                #print( "dst: " +  str(pkt.getlayer(IP).chksum))
                writer.writerow({'IP src': str(pkt.getlayer(IP).src), 'IP dst': str(pkt.getlayer(IP).dst), 
                            'sport': str(pkt.getlayer(TCP).sport), 'dport': str(pkt.getlayer(TCP).dport), 
                            'len': int(pkt.getlayer(IP).len), 'Iflags': int(pkt.getlayer(IP).flags), 
                            'frag': int(pkt.getlayer(IP).frag), 'ttl': int(pkt.getlayer(IP).ttl), 
                            'Ichksum': int(pkt.getlayer(IP).chksum),
                            'seq': int(pkt.getlayer(TCP).seq), 'ack': int(pkt.getlayer(TCP).ack), 
                            'dataofs': int(pkt.getlayer(TCP).dataofs), 'Tflags': int(pkt.getlayer(TCP).flags),
                            'window': int(pkt.getlayer(TCP).window), 'Tchksum': int(pkt.getlayer(TCP).chksum),'Label': "Normal"})
    
    print("Done with conversion",fname)
    return outfile



if __name__ == "__main__":

    ## Analyze the 2016 PCAP files
    print("Starting pcap to csv conversion...")
    extract_pcaps_files(Path)
    ##fname = 'benign-dec.pcap'
    ##pcapToCsv(fname)
#csvfile=pcapToCsv(fname)
    ##csvfile='ACKFloodingMerged.csv'
    ##print("Output csv file name "+csvfile )

    ##data = pd.read_csv(csvfile)
#print(data.head())

    print("Pcap to CSV conversion completed!")
    print("")
