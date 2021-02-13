function [gpeaks,gaps] = findGridSequence(ipeaks,gridsize)
    % The allowed grid sequence can be one strike after another 
    % or just one line.
    total_peaks = size(ipeaks);
    gpeaks = [];
    gaps = [];
    gsize = length(gridsize);
    if(gsize==2)
        n_combinations = 4;
    else
        n_combinations = 2;
    end
    possible_pairs = combnk(1:total_peaks(1),n_combinations);
    dim_pairs = size(possible_pairs);
    
    j = 1;
    error = [];
    saved_group = {};
    for m = 1:dim_pairs(1)
        comb_indxs = possible_pairs(m,:);
        total_p = ipeaks(comb_indxs,:);      
        total_p = sortrows(total_p,-1);
        [rows,~] = size(total_p);
        k =1;
        terror= 0;
        for i=1:rows-1
            dist_p(k) = abs(total_p(i,1)- total_p(i+1,1));
            k = k+1;
        end
        good = 1;
        for i=1:gsize
            dif_error = (abs(dist_p(i) - gridsize(i))/gridsize(i));
            terror= terror+dif_error;
            if(dif_error>0.3) %first test, fit grids conditions 
                good = 0;
            end
            
        end    
        for i=1:gsize % check again symmetrically
            dif_error = (abs(dist_p(length(dist_p)-i+1) - gridsize(i))/gridsize(i));
            terror = terror+dif_error;
            if(dif_error>0.3) % fit grids conditions 
                good = 0;
            end
        end 
        if(good==1)
            if(max(total_p(:,2))-min(total_p(:,2))>3) % second test, difference between angles not bigger than 3 deg
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
                    if(abs(mode(g1(:,2))-mode(g2(:,2)))<2) % angle is the same
                        r_group = union(g1(:,1),g2(:,1)); % check what is similar
                        max_distance = max(m_pdist(r_group)); % check the remaining and how far are one each other
                        if(max_distance<1.5*max(gridsize)) % one of the two groups must disappear (if they are not far enough)
                            if(error(i)<error(j)) % We keep the set with minimum error
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
   
   gaps = [];
   [rows,~]=size(gpeaks);
   for i=1:rows-1
       gaps(i) = abs(gpeaks(i,1)- gpeaks(i+1,1));
   end   
end