function PR = projectionsTwo(edges,alpha) 


% pad the image with zeros so we don't lose anything when we rotate.
[iLength, iWidth] = size(edges);
iDiag = sqrt(iLength^2 + iWidth^2);
LengthPad = ceil(iDiag - iLength) + 2;
WidthPad = ceil(iDiag - iWidth) + 2;
padIMG = zeros(iLength+LengthPad, iWidth+WidthPad);
padIMG(ceil(LengthPad/2):(ceil(LengthPad/2)+iLength-1), ...
       ceil(WidthPad/2):(ceil(WidthPad/2)+iWidth-1)) = edges;



tmpimg = imrotate(padIMG, alpha, 'bilinear', 'crop');
se = strel('line',5,90);
tmpimg = imdilate(tmpimg,se);

%show(tmpimg)
pr = (sum(tmpimg'));
pr(find(pr<0))=0.0;
pr = pr/iDiag;
PR = pr.^2;
%PR = pr/max(pr);