
function r = m_chamfer(imTarget,imTemplate)
% distance transform and matching

dist = bwdist(imTarget, 'euclidean');
distT = bwdist(imTemplate,'euclidean');
r = normxcorr2(distT, dist);
%map = imfilter(dist, double(imTemplate), 'corr', 'same');
%add the local minima pixels
%threshold = 4*min(min(map));
%val = map(find(map < threshold)); 
%r = 1/length(val);