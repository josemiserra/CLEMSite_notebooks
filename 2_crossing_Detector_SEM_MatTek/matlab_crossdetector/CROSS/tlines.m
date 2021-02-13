function [rlines] = tlines(iimg,ipeaks,fname)
%% This MATLAB function takes an image and a set of peaks  
%% It locates the peaks inside the function and it translate them
%% to lines in the image (slope and intercept) 
%% (0,0) up left corner

% Get image size, value of diagonal,value of padding, and angle relation
% between Lenght and Width.
[iLength, iWidth] = size(iimg);
iDiag = sqrt(iLength^2 + iWidth^2);
LengthPad = ceil(iDiag - iLength) + 2;
WidthPad = ceil(iDiag - iWidth) + 2;

iimgc = (iimg>-1);

padIMG = zeros(iLength+LengthPad, iWidth+WidthPad);
padIMG(:) = -1;
padIMG(ceil(LengthPad/2):(ceil(LengthPad/2)+iLength-1), ...
       ceil(WidthPad/2):(ceil(WidthPad/2)+iWidth-1)) = iimgc;

 top = max(iWidth+WidthPad,iLength+LengthPad);
   

lines = ipeaks(:,1);
i=1;
m=1;
posx = zeros(2*length(lines),1);
for k = 1:length(lines)   
   tmpimg = imrotate(padIMG, ipeaks(k,2), 'bilinear', 'crop');  
   j=1;
   posx(i) = 0;
   posx(i+1) = 0;
   found = false;
   while(j<top)
     if(tmpimg(uint32(ipeaks(k,1)),j)>0) 
      posx(i)=j;
      found = true;
      break;
     end;
      j = j + 1;
   end;
   if(~found)
       posx(i)=0;
   end;
   while(j<top)
     if(tmpimg(uint32(ipeaks(k,1)),j)<0) 
      posx(i+1)=j-1;
      break;
     end;
      j = j + 1;
   end;
   if(j>top)
       posx(i+1)=top;
   end;
   i=i+2;
   
   angle = ipeaks(k,2);
   rotmat =[ cosd(angle) -sind(angle); sind(angle) cosd(angle) ];
   p1 = [(posx(i-1)-1-(WidthPad/2+iWidth/2)) ipeaks(k,1)-1-(LengthPad/2+iLength/2)];
   p2 = [(posx(i-2)-1-(WidthPad/2+iWidth/2)) ipeaks(k,1)-1-(LengthPad/2+iLength/2)];
   p1 = rotmat*p1';
   p2 = rotmat*p2';
   p1(1) = ceil(p1(1)+iWidth/2+1);
   p1(2) = ceil(p1(2)+iLength/2+1);
   p2(1) = ceil(p2(1)+iWidth/2+1);
   p2(2) = ceil(p2(2)+iLength/2+1);
   rlines(m,:) = p1;
   rlines(m+1,:) = p2;
   m=m+2;
%     show(padIMG);
%     hold on;
%     line( [p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',3,'Color','red');
   
%     show(tmpimg);
%     hold on;
%     line( [posx(i-1) posx(i-2)],[ipeaks(k,1) ipeaks(k,1)],'LineWidth',3,'Color','red');
%     plot(posx(i-1),ipeaks(k,1),'o','LineWidth',2,'Color','yellow');
%     plot(posx(i-2),ipeaks(k,1),'o','LineWidth',2,'Color','yellow');
   
end;
i=1;
 
% figure('Visible','off');
% axis('image'); axis('off');
% subplot('position',[0 0 1 1]);
% imshow(iimg); hold on;
show(iimg); hold on;

for k=1:2:length(rlines)
%% NO!
%    if(p1(1)<1) p1(1) = 1; end;
%    if(p1(2)<1) p1(2) = 1; end;
%    if(p2(1)<1) p2(1) = 1; end;
%    if(p2(2)<1) p2(2) = 1; end;
%    if(p1(2)>iLength) p1(2) = iLength; end;
%    if(p1(1)>iWidth) p1(1) = iWidth; end;
%    if(p2(2)>iLength) p2(2) = iLength; end;
%    if(p2(1)>iWidth) p2(1) = iWidth; end;
    p1 = rlines(k,:);
    p2 = rlines(i+1,:);
    line( [p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'Color','red');
    i=i+2;
end;
 fname = strcat(fname,'.tif');
 saveas(gcf,fname,'tif');

 
 
