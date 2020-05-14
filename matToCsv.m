clear,clc,close all

% load in .mat file containing 'forGPFA' data
% dump all trials in forGPFA.dat into separate csv files (trial<x>.csv)

load('data/14October2013.mat')

data = forGPFA.dat;

numTrials = size(data,2);

for i=1:numTrials
    csvwrite(strcat('data/trial',int2str(i),'.csv'),data(i).spikes);
end