# CS 4793 Fall 2016
# Written by : Doug Heisterkamp
# last modified: 09/14/2016
"""
helper program to generate plots from training and validation error rate log file

Expects data file format to be whitespace separated with one epoch rates per line:

epoch  trTLU0 trTLU1 ... trTLUk trTLUall valTLU0 valTLU1 ... valTLUk valTLUall 

where tr stands for the training error rates and val the validation error rates.
"""

#imports from __future__ to make python version 2.7 behave like 3
#Also renames raw_input to input
#
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import numpy
import sys

if __name__ == "__main__":
   if len(sys.argv) != 2 :
      print("usage: python3 pa1plot.py  pa1ErrorLog")
      sys.exit(-1)

   d = numpy.loadtxt(sys.argv[1])
   k = (d.shape[1] -1)//2
   epoch = d[:,0]
   trError = d[:,1:k+1]
   valError = d[:,k+1:]

   #plt.style.use('ggplot')
   with PdfPages('pa1plots_'+ sys.argv[1][:-4]+ '.pdf') as pdf:
      # We can also set the file's metadata via the PdfPages object:
      pd = pdf.infodict()
      pd['Title'] = 'CS 4793 PA1'
      #pd['Author'] = ''
      pd['Subject'] = 'TLU Error Rates'
      pd['Keywords'] = ''
      #pd['CreationDate'] = datetime.datetime(2016, 9, 13)
      pd['CreationDate'] = datetime.datetime.today()
      #pd['ModDate'] = datetime.datetime.today()

      for p in [0,1]:
         if p == 0 :
            rates = trError
            titlestr = "PA1 Training from "+sys.argv[1]
         else:
            rates = valError
            titlestr = "PA1 Validation from "+sys.argv[1]
         # place 4 plots per page
         t = 0
         while t+3 < k:
            fig, ax1 = plt.subplots(figsize=(8,5))
            plt.plot(epoch,rates[:,t],label="TLU {}".format(t))
            plt.plot(epoch,rates[:,t+1],label="TLU {}".format(t+1))
            plt.plot(epoch,rates[:,t+2],label="TLU {}".format(t+2))
            tl = "TLU {}".format(t+3) if t+3 < k-1 else "ALL TLU"
            plt.plot(epoch,rates[:,t+3],label=tl)
            plt.xlabel('Epoch')
            plt.ylabel('Error Rate')
            plt.title(titlestr)
            plt.grid(True)
            plt.legend()
            ax1.margins(0.05)
            pdf.savefig(fig)
            plt.close()
            t += 4
         if t < k :
            fig, ax1 = plt.subplots(figsize=(8,5))
            tl = "TLU {}".format(t) if t < k-1 else "ALL TLU"
            plt.plot(epoch,rates[:,t],label=tl)
            if t +1 < k :
               tl = "TLU {}".format(t+1) if t+1 < k-1 else "ALL TLU"
               plt.plot(epoch,rates[:,t+1],label=tl)
            if t +2 < k :
               tl = "TLU {}".format(t+2) if t+2 < k-1 else "ALL TLU"
               plt.plot(epoch,rates[:,t+2],label=tl)
            if t +3 < k :
               tl = "TLU {}".format(t+3) if t+3 < k-1 else "ALL TLU"
               plt.plot(epoch,rates[:,t+3],label=tl)
            plt.xlabel('Epoch')
            plt.ylabel('Error Rate')
            plt.title(titlestr)
            plt.grid(True)
            plt.legend()
            ax1.margins(0.05)
            pdf.savefig(fig)
            plt.close()
    


