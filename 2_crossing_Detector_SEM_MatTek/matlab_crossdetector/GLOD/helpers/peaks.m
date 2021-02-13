function [npeaks]= getPeaks(R)

%% select right peaks, then scan in upper and lower parts of the signal to
% get new weaker peaks
ipeaks = [];
fprintf('Finding peaks by non maximum supression \n');
peaks = findpeaks2(R,12,'Threshold',0.05);
ipeaks = discardwrongpeaks(peaks); % based on the grid definition
fprintf('%d peaks found.\n',length(ipeaks));
npeaks = refinePeaks(R,ipeaks); % Refine, again, based on the grid definition
