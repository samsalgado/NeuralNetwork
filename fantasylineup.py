import torch
import os
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import sys
#Convert Defense Stat Data into a Tensor
Teams = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022]
Def = [15, 14, 21.5, 14.4, 19.6, 18.5, 15.6, 18.4, 20.3, 19.2, 22.2, 21.9, 21.7]
data = torch.tensor(Def)
print(data)
print(data[0])

#Note that Tensor and Pytorch is for Deep Learning
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x,y)
s = x.view(4)
print(s)





dfs_data = pd.read_csv('weekly_points_data.csv')
print(dfs_data)
dfs_data.describe()
QB = [18.4,
24.2,
21.9,
13.2,
15.3,
20.7,
21.1,
19.4,
19.2,
17.8,
16.7,
16.7,
18.7,
16.4,
17.3,
17.7,
17,
14.7,
18.6,
18.4,
15.8,
16.2,
12.8,
12.6,
10.4,
19.4,
16.8,
18.9
]
x = numpy.median(QB)
print(x)
#Create Tensor for above 17.5
DFS_QB = {
"Names": ['Patrick Mahomes II', 'Josh Allen', 'Jalen Hurts', 'Dak Prescott', 'Lamar Jackson', 'Brock Purdy', 'CJ Stroud', 'Jordan Love', 'Kyler Murray', 
         'Justin Fields', 'Jared Goff', 'Russell Wilson', 'Justin Herbert', 'Kirk Cousins'],
"AboveAVG": [18.4, 24.2, 21.9, 20.7, 21.1, 19.2, 18.7, 19.4, 18.9, 18.4, 17.8, 17.7, 18.6, 19.4]
}
QB_Above_AVG = [18.4, 24.2, 21.9, 20.7, 21.1, 19.2, 18.7, 19.4, 18.9, 18.4, 17.8, 17.7, 18.6, 19.4]
Above_avg = torch.tensor(QB_Above_AVG)
print(Above_avg)
df = pd.DataFrame(DFS_QB)
print(df)
x = ['Mahomes II', 'Allen', 'Hurts', 'Prescott', 'Jackson', 'Purdy', 'Stroud', 'Love', 'Murray', 
         'Fields', 'Goff', 'Wilson', 'Herbert', 'Cousins']
y = [18.4, 24.2, 21.9, 20.7, 21.1, 19.2, 18.7, 19.4, 18.9, 18.4, 17.8, 17.7, 18.6, 19.4]
plt.figure(figsize=(14, 6))    
plt.scatter(x, y)
plt.subplots_adjust(left=0.1, bottom=0.1, 
                    top=0.9, wspace=6.5,hspace=7.5)
plt.show()
Elite_QB = [18.4, 24.2, 21.9, 20.7, 21.1, 19.2, 18.7, 19.4, 18.9, 18.4, 17.8, 17.7, 18.6, 19.4]
e = numpy.median(Elite_QB)
print(e)
#Elite = 19.05
sys.stdout.flush()
#Do the Same for Pass Catchers
PassCatchers = [ 23.7,23.5,20.7,17.6,17,16.9,16.6,21.5,16.1,15.6,16.4,17.4,14.1,13.5,14.6,14.6,12.7,12.6,10.7,10.7,10.4,10,9.4,9,8.1,9.7]
Aboveavg = numpy.median(PassCatchers)
print(Aboveavg)
#Print above 17.2
DFS_wideouts = {
    "Names":['Lamb', 'Hill', 'St Brown', 'Nacua', 'Allen', 'Collins'],
    "AboveAvg":[23.7,23.5,20.7,17.6,21.5,17.4]
}

df2 = pd.DataFrame(DFS_wideouts)
print(df2)
Elite_WR = [23.7,23.5,20.7,17.6,21.5,17.4]
elite = numpy.median(Elite_WR)
print(elite)
#Plot is for above 21.1
TE_DFS = [14.1,
13.5,
14.6,
14.6,
12.7,
12.6,
10.7,
10.7,
10.4,
10,
9.4,
9,
8.1,
9.7
]
TE = numpy.median(TE_DFS)
print(TE)
#Above 10.7
te_DFS = {
    "Names":['LaPorta', 'Engram', 'Kelce', 'Hockenson', 'Kittle', 'Njoku'],
    "Stats":[14.1,13.5,14.6,14.6,12.7,12.6]
}
df3 = pd.DataFrame(te_DFS)
print(df3)
elite_te = [14.1,13.5,14.6,14.6,12.7,12.6]
TE_ELITE = numpy.median(elite_te)
print(TE_ELITE)
#Get above avg of pass catchers
ALL_PassCatchers = [23.7,23.5,20.7,17.6,21.5,17.4,14.1,13.5,14.6,14.6,12.7,12.6]
ALL_receivers = numpy.median(ALL_PassCatchers)
print(ALL_receivers)
RB_DFS = [24.5,
17.1,
16.6,
15.8,
17.8,
15.7,
21.3,
14.5,
14.5,
16.1,
17.9,
13.7,
15.9,
13.1,
15.3,
12.4,
15.5,
13.3
]
RB = numpy.median(RB_DFS)
print(RB)
#DF for above 15.75
RBDFS = {
    "Names":['McCaffrey', 'Hall', 'Etienne Jr', 'White', 'Mostert', 'Kamara', 'Barkley', 'Gibbs', 'Williams'],
    "Stats":[24.5,17.1,16.6,15.8,17.8,17.9,15.9, 16.1, 21.3]
}

df = pd.DataFrame(RBDFS)
print(df)
Elite_RB = [24.5,17.1,16.6,15.8,17.8,17.9,15.9, 16.1, 21.3]
elite_backs = numpy.median(Elite_RB)
print(elite_backs)
#PLot ABOVE Elite Players: QB:19.05, RB:17.1, TE:13.8, WR:21.1

Elite_DFS = {
 "Names": [ 'Allen', 'Hurts', 'Prescott', 'Jackson', 'Purdy','Love', 'Cousins', 'McCaffrey',
     'Williams', 'Mostert', 'Kamara', 'LaPorta', 'Kelce', 'Hockenson', 'Lamb', 'Ty Hill', 'K.Allen'],
 "Stats":[24.2,21.9,20.7,21.1,19.2,19.4,19.4,24.5,21.3,17.8,17.9,14.1,14.6,14.6,23.7,23.5,21.5]
}
df = pd.DataFrame(Elite_DFS)
print(df)
Names = [ 'Allen', 'Hurts', 'Prescott', 'Jackson', 'Purdy','Love', 'Cousins', 'McCaffrey',
     'Williams', 'Mostert', 'Kamara', 'LaPorta', 'Kelce', 'Hockenson', 'Lamb', 'Ty Hill', 'K.Allen']
Stats = [24.2,21.9,20.7,21.1,19.2,19.4,19.4,24.5,21.3,17.8,17.9,14.1,14.6,14.6,23.7,23.5,21.5]
Jersey = [17, 1, 4, 8, 13, 10, 8, 23, 23, 31, 41, 87, 87, 87, 88, 10, 13]
plt.figure(figsize=(14, 6))    
plt.scatter(Names,Stats)
plt.subplots_adjust(left=0.1, bottom=0.1, 
                    top=0.9, wspace=6.5,hspace=7.5)
plt.show()
#DEFENSE and Kickers outlined in the Notebook
Ultimate_Lineup_2024 = {
    "QB":['Josh Allen', 'Jalen Hurts', 'Dak Prescott', 'Brock Purdy', 'Jordan Love'],
    "RB":['Christian McCaffrey', 'Kyren Williams', 'Raheem Mostert', 'Alvin Kamara'],
    "WR":['CeeDee Lamb', 'Tyreek Hill', 'Keenan Allen'],
    "TE":['Sam LaPorta', 'Travis Kelce', 'TJ Hockenson']
}
print(Ultimate_Lineup_2024)
