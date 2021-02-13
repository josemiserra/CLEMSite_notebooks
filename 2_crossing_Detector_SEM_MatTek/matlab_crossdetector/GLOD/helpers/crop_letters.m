function [myimg]= crop_letters(iimg,all_cutpoints,angneg,distsq,folder)

[iLength, iWidth] = size(iimg);
[rows cols] = size(all_cutpoints);
k = 1;
for j=1:4:rows
    myimg=imrotate(iimg,angneg,'crop');
    % Rotate inner point and move to center of coordinates
    minxp = 1;
    minyp = 1;
    distmin = 100000;
    show(myimg);
    hold on;    
    for i = 1:4
        xp = all_cutpoints(j+i-1,1);
        yp = all_cutpoints(j+i-1,2);
        minpoint = [xp-iWidth/2 yp-iLength/2];
        irotmat =[ cosd(angneg) sind(angneg); -sind(angneg) cosd(angneg) ];
        minpoint = irotmat*minpoint';
        minpoint = ceil([iWidth/2+minpoint(1)+1 iLength/2+minpoint(2)+1]);
        plot(minpoint(1),minpoint(2),'.','LineWidth',2,'Color','blue');
        distp = round(sqrt(minpoint(1)^2 + minpoint(2)^2));
        if(distp<distmin) 
            minxp =minpoint(1); minyp = minpoint(2); 
            distmin = distp;
        end       
    end
    myimg = imcrop(myimg,[minxp minyp distsq distsq]);
    hold off
    show(myimg);
    
    fname = strcat(folder,'\letter_');
    fname = strcat(fname,num2str(k));
    k = k+1;
    fname = strcat(fname,'.tif');
    imwrite(myimg,fname);  
end
