clear,clc,close all

% load in .mat file containing 'forGPFA' data
% dump all trials in forGPFA.dat into separate csv files (trial<x>.csv)

load('data/14October2013.mat')

data = forGPFA.dat;

numTrials = size(data,2);

% for i=1:numTrials
%     csvwrite(strcat('data/trial',int2str(i),'.csv'),data(i).spikes);
% end

RT = [];
cue = [];
choice = [];
for i=1:numTrials
    RT(i) = data(i).RT;
    cue(i) = data(i).Cue;
    choice(i) = data(i).choice;
end

RT = RT';
cue = cue';
choice = choice';

exData = [RT, cue, choice];

csvwrite('data/RT_cue_choice.csv',exData);







