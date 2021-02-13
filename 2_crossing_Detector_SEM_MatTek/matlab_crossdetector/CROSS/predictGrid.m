%% This MATLAB function takes a group of peaks and tries
%% to predict where the complementary angle peaks are
%% The grid image is then reconstructed

function [goodpeaks,error] = predictGrid(bpeaks,R,gridsize)
     error=0;
     goodpeaks = [];
    
    if(length(bpeaks)<2)
        error = 3;
        return
    end
    
    angle = mode(bpeaks(:,2));
    bpeaks(:,2)=angle; 
    cangle = 0;
    if(angle>0)
        cangle = angle - 90;
    else
        cangle = angle+90;
    end
    c180angle = cangle +90+1;
   
    signal = R(:,c180angle); % get complementary
    [speaks] = nonmaxsup1D(signal,6,round(gridsize*0.5));
    ms_dim = size(speaks);
    if(ms_dim(1)<2)
        goodpeaks = [];
        error =3;
        return;
    end
    fpeaks = zeros(6,3);
    
    for i =1:ms_dim(1)
        fpeaks(i,1) = speaks(i,2);
    end
    for i=1:6
        fpeaks(i,2) = cangle+1;
        fpeaks(i,3) = 0.99;
    end
    
    goodpeaks = findCrossSequence(fpeaks,gridsize);
    
    if(isempty(goodpeaks))
        fpeaks = sortrows(fpeaks,2); % sort by amplitude of signal
        origin = fpeaks(1,1);
        max_l = size(R);
        max_l = max_l(1);
        goodpeaks = zeros(2,3);
        goodpeaks(1,1) = origin;
        goodpeaks(2,1) = origin+gridsize; 
        for i =1:2
              if(goodpeaks(i,1)>max_l)
                  goodpeaks(1,1) = fpeaks(i+1,1);
                  goodpeaks(2,1) = fpeaks(i+1,1)+gridsize; 
              end
            goodpeaks(i,3) = 0.99;
            goodpeaks(i,2) = cangle;
         end
         if(goodpeaks(i,1)>max_l)
            goodpeaks = [];
            error =3;
            return; 
         end
         
    end
    
    goodpeaks = [ goodpeaks; bpeaks];

end
        
    



%%
function [gpeaks,gaps] = findGaps(total_p)
    total_peaks = length(total_p);
    gpeaks = [];
    gaps = []; 
    gap2 = 0;
    gap3 = 0;
   
    total_p = sortrows(total_p, 1);
    gap1 = abs(total_p(1,1)- total_p(2,1));
   
    if(total_peaks>2)
       gap2 = abs(total_p(2,1)- total_p(3,1));
    end
    if(total_peaks>3)
        gap3 = abs(total_p(3,1)- total_p(4,1));         
    end       
        
    gpeaks = total_p;
    gaps = [gap1,gap2,gap3];

end


%% 

