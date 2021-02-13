function [npeaks,error]= getPeaks(R,gridsize)

%% select right peaks, then scan in upper and lower parts of the signal to
% get new weaker peaks
npeaks = [];
fangles = [];
ipeaks = 0;
error = 0;
fprintf('Finding peaks by non maximum supression \n');
in_peaks = (length(gridsize)*4)*2;
peaks = detectPeaksNMS(R,in_peaks,'Threshold',0.05);
[ipeaks,error] = discardwrongpeaks(R,peaks,gridsize); % based on the grid definition
if(error>0)
    fprintf('ERROR II: not enough peaks found. Bad image quality. \n');
    error = 2;
    return;
end
fprintf('%d peaks found.\n',length(ipeaks));
npeaks = alternativePeaks(R,ipeaks); % Refine, again, based on the grid definition
fprintf('A total of %d peaks found during refinement.\n',length(npeaks)-length(ipeaks));
npeaks(:,2) = npeaks(:,2)-1;

