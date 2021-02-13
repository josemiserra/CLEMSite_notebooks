function [npeaks,error]= getPeaks(R,gridsize)

%% select right peaks, then scan in upper and lower parts of the signal to
% get new weaker peaks
error = 0;
fprintf('Finding peaks by non maximum supression \n');
peaks = detectPeaksNMS(R,12,'Threshold',0.05);
npeaks = discardwrongpeaks(R,peaks,gridsize); % based on the grid definition
if(isempty(npeaks))
    error = 3;
    return
end
fprintf('%d peaks found.\n',length(npeaks));
npeaks(:,2) = npeaks(:,2)-1;