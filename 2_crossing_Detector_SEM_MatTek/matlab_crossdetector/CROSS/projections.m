% projections.m

%% This MATLAB function takes an image matrix and vector of angles and then 
%% finds the 1D projection (Radon transform) at each of the angles.  It returns
%% a matrix whose columns are the projections at each angle.


function PR = projections(swt,orientim,K,inc,aspace,range)

% pad the image with zeros so we don't lose anything when we rotate.
[iLength, iWidth] = size(swt);
iDiag = sqrt(iLength^2 + iWidth^2);
LengthPad = ceil(iDiag - iLength) + 2;
WidthPad = ceil(iDiag - iWidth) + 2;
padIMG = zeros(iLength+LengthPad, iWidth+WidthPad);
padIMG(ceil(LengthPad/2):(ceil(LengthPad/2)+iLength-1), ...
       ceil(WidthPad/2):(ceil(WidthPad/2)+iWidth-1)) = swt;

padIMGOR = zeros(iLength+LengthPad, iWidth+WidthPad);
padIMGOR(ceil(LengthPad/2):(ceil(LengthPad/2)+iLength-1), ...
       ceil(WidthPad/2):(ceil(WidthPad/2)+iWidth-1)) = orientim;   
   
   
% loop over the number of angles, rotate 90-theta (because we can easily sum
% if we look at stuff from the top), and then add up.  Don't perform any
% interpolation on the rotating.

 % -90 and 90 are the same, we must remove 90
 THETA = (-90:inc:89);
 th = zeros(size(THETA))+inf;
 if(range>0)
    for i = 1:length(aspace)
        k = aspace(i)+90;
        kplus = k+range;
        kminus = k-range;
        if(kplus>180)  kplus = 180; end;
        if(kminus<0)   kminus = 1; end;
        th(k:kplus) = THETA(k:kplus);
        th(kminus:k) = THETA(kminus:k);
    end;
 else
     th = THETA;
 end;
 th = th*pi/180;   
    
% THETA = (0:inc:180);
% th =(0:inc:180)*pi/180;
 n = length(THETA);
 PR = zeros(size(padIMG,2), n);

M = padIMG > 0;
% h = waitbar(0,'Please wait...');
max_seg = length(padIMG);

for i = 1:n
   % tic
   if(th(i)~=inf)
    % tmpimg = imrotate(padIMG, THETA(i), 'bilinear', 'crop');
    final = coft(M,K, padIMGOR,th(i));
    % final = gpuArray(final);
    tmpimg = imrotate(final,THETA(i), 'bilinear', 'crop');
    % if(mod(i,30)==0) show(tmpimg); end
    PR(:,i) = (sum(tmpimg,2));
    % waitbar(i / n)
    % toc
   else
    PR(:,i)=zeros(1,max_seg);
   end
end

PR(PR<0)=0.0;
PR = PR/iDiag;
PR = PR*10;
PR = PR.^2;
PR = PR*0.1;
PR = PR/max(max(PR));
% close(h);

