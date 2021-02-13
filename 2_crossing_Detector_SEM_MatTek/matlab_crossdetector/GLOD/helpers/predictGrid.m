%% This MATLAB function takes a group of peaks and tries
%% to predict where the complementary angle peaks are
%% The grid image is then reconstructed

function [goodpeaks,error] = predictGrid(ipeaks,R,gridsize)
     error=0;
     goodpeaks = [];
    % Purge to grid
    [bpeaks,gaps] = findGridSequence(ipeaks,gridsize);
    
    if(length(bpeaks)<4)
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
    [speaks] = nonmaxsup1D(signal,6,round(gaps(1)/2));
    ms_dim = size(speaks);
    if(ms_dim(1)<4)
          % We could evaluate 2 consecutive peaks within  gap1 
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
    
    [goodpeaks,~] = findGridSequence(fpeaks,gridsize);
    
    if(isempty(goodpeaks))
        [fpeaks,ngaps] = findGridSequence(fpeaks,min(gridsize));
        if(isempty(fpeaks))
            goodpeaks = [];
            error = 3;
            return;
        end
        fpeaks = sortrows(fpeaks,-1);
        origin = fpeaks(1,1);
        max_l = size(R);
        max_l = max_l(1);
        goodpeaks = zeros(4,3);
        goodpeaks(1,1) = origin;
        goodpeaks(2,1) = fpeaks(2,1);
        if(origin<max_l/2)
           goodpeaks(3,1) = origin+gaps(1)+gaps(2);
           goodpeaks(4,1) = origin+gaps(1)+gaps(2)+gaps(1);
        else
           goodpeaks(3,1) = origin-gaps(1)-gaps(2);
           goodpeaks(4,1) = origin-gaps(1)-gaps(2)-gaps(1);
        end
           
        for i =1:4
              if(goodpeaks(i,1)<0 || goodpeaks(i,1)>max_l)
                  goodpeaks = [];
                  error =3;
                  return
              end
            goodpeaks(i,3) = 0.99;
            goodpeaks(i,2) = cangle;
         end
           
    end
    
    goodpeaks = [ goodpeaks; bpeaks];
   % If is grided by default, we use the default and we enhance the peaks
    % at that point
   
    % From peaks get lines : evaluate 
   %  rlines = tlines(fswt,goodpeaks,'E:\f'); 
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

