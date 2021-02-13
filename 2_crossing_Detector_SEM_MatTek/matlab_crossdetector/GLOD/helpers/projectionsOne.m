function PR = projectionsOne(swt,orientim,K,alpha)

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
   
 
M = padIMG > 0;

final = soft(M,K, padIMGOR,alpha);
% final = gpuArray(final);
tmpimg = imrotate(final, alpha, 'bilinear', 'crop');
% if(mod(i,30)==0) show(tmpimg); end
pr = (sum(tmpimg'));
pr(find(pr<0))=0.0;
pr = pr/iDiag;
pr = pr.^2;
PR = pr/max(pr);