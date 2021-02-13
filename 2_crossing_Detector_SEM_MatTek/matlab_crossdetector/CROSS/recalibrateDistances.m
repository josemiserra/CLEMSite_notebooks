function [npeaks] = recalibrateDistances(ipeaks,mSmall, mBig)
    %% Find average distance between peaks
    % Take positive
    pospeaks = ipeaks(find(ipeaks(:,2)>-1),:);
    % Take negative
    negpeaks  = ipeaks(find(ipeaks(:,2)<0),:);
    
    %% Compute difference between consecutive points.
    [prows pcols] = size(pospeaks);
    sortrows(pospeaks,1); % just in case
    s1 = [];
    i = 1;
    j = 1;
    for k = 1:(prows-1)
       dist =  pospeaks(k+1,1)-pospeaks(k,1);
       if(mod(k,2) ~= 0) 
          s1(i,1) = dist;
          s1(i,2) = k;
          i = i+1;
       end
    end
   
   % Now we proceed to find the first closest value for positive and for negative. 
   % Value that is equal to msmall is marked as fixed (assumed,
   % correct)
   [srows scols] = size(s1);
   sortrows(s1,1);
   for k = 2:srows
       if (s1(k,1)>mSmall)
           if((s1(k,1)-mSmall)>(mSmall-s1(k-1,1)))
                fixed = k-1;
           else 
               fixed = k;
           end
           break;
       end
   end
  
   for k = fixed:prows-1
        if(mod(k,2) == 0)
            pospeaks(k+1,1) = pospeaks(k,1)+mBig;
        else
            pospeaks(k+1,1) = pospeaks(k,1)+mSmall;
        end
   end
       
   
   [nrows ncols] = size(negpeaks);
   sortrows(negpeaks,1);
   for k = 1:(nrows-1)
       dist =  negpeaks(k+1,1)-negpeaks(k,1);
       if(mod(k,2) ~= 0) 
                s2(j,1) = dist;
                s2(j,2) = k;
                j = j+1;
       end
   end
   
   [srows scols] = size(s2);
   sortrows(s2,1);
   for k = 2:srows
       if (s2(k,1)>mSmall)
           if((s2(k,1)-mSmall)>(mSmall-s2(k-1,1)))
                fixed = k-1;
           else 
               fixed = k;
           end
           break;
       end
   end
  
   for k = fixed:nrows-1
        if(mod(k,2) == 0)
            negpeaks(k+1,1) = negpeaks(k,1)+mBig;
        else
            negpeaks(k+1,1) = negpeaks(k,1)+mSmall;
        end
   end
   
    npeaks = [ pospeaks; negpeaks];
        