function [mSmall mBig] = patternDistances(ipeaks)
    %% Find average distance between peaks
    % Take positive
    pospeaks = ipeaks(find(ipeaks(:,2)>-1),:);
    % Take negative
    negpeaks  = ipeaks(find(ipeaks(:,2)<0),:);
    
    %% If everything is correct, compute differences between consecutives points
    [srows scols] = size(pospeaks);
    sortrows(pospeaks,1); % just in case
    s1 = [];
    s2 = [];
    i = 1;
    j = 1;
    for k = 1:(srows-1)
       dist =  pospeaks(k+1,1)-pospeaks(k,1);
       if(mod(k,2) == 0) 
                s2(j) = dist;
                j = j+1;
       else
          s1(i) = dist;
          i = i+1;
       end
    end
   [srows scols] = size(negpeaks);
   sortrows(negpeaks,1);
   for k = 1:(srows-1)
       dist =  negpeaks(k+1,1)-negpeaks(k,1);
       if(mod(k,2) == 0) 
                s2(j) = dist;
                j = j+1;
       else
          s1(i) = dist;
          i = i+1;
       end
   end
   mSmall = median(s1);
   mBig = median(s2);
   
   