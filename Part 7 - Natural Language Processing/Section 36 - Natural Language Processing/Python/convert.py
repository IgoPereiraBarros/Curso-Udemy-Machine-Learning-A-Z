# -*- coding: utf-8 -*-

# Exemplo

import csv
 
with open('D: file.csv','r') as csvin, open('D:/NAD.txt', 'w') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')
 
    for row in csvin:
        tsvout.writerow(row)