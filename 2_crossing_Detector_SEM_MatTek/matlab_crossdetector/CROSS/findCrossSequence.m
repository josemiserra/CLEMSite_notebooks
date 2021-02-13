function [gpeaks] = findCrossSequence(ipeaks,gridsize)
    % The allowed grid sequence can be one strike after another 
    % or just one line.
    total_peaks = size(ipeaks);
    gpeaks = [];
   
    possible_pairs = combnk(1:total_peaks(1),2);
    dim_pairs = size(possible_pairs);
    
    j = 1;
    error = [];
    saved_group = {};
    for m = 1:dim_pairs(1)
        comb_indxs = possible_pairs(m,:);
        total_p = ipeaks(comb_indxs,:);      
        total_p = sortrows(total_p,-1);
        [rows,~] = size(total_p);
        terror= 0;
        dist_p = abs(total_p(1,1)- total_p(2,1));
        good = 1;
        dif_error = (abs(dist_p - gridsize)/gridsize);
        terror= terror+dif_error;
        if(dif_error>0.3) %first test, fit grids conditions 
                good = 0;
        end           
        if(good==1)
            if(abs(total_p(1,2)-total_p(2,2))>2) % second test, difference between angles not bigger than 3 deg
                good = 0;
            end
        end
        if(good==1)
           saved_group{j} = total_p;
           error(j) = terror;
           j = j+1;
        end
    end
   
   if(isempty(saved_group))
       return; 
   end
   % For each group check that they are not competing in the same distances
   k = 1;
   good_group = {};
   if(length(saved_group) == 1)
    good_group = saved_group;
   end
   for i = 1:length(saved_group)
            for j = i+1:length(saved_group)
                if(i~=j)
                    g1 = saved_group{i};
                    g2 = saved_group{j};
                    err_dif = abs(error(i)-error(j));
                    if(err_dif>0.1) % We keep the set with minimum error
                        if(error(i)<error(j)) 
                            good_group{k}=g1;
                        else
                            good_group{k}=g2;
                        end
                        k = k+1;
                    else %potential candidates, leave them
                        good_group{k} = g1;
                        k=k+1;
                        good_group{k} = g2;
                        k = k+1;
                    end
                end
            end
   end   
   if(isempty(good_group))
       return; 
   end
   
   % Now is time to select the BEST candidate
   % Is going to be the one that sums up the most
   sg = [];
   for i=1:length(good_group)
       g = good_group{i};
       sg(i) = sum(g(:,3));
   end
   gpeaks = good_group{find(sg==max(sg))};
 
end