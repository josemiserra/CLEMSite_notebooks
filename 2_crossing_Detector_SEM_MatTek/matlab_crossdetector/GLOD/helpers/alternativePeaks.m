function [npeaks]=alternativePeaks(R,ipeaks)
% ipeaks are supossed to be good peaks. We can try now to get the weak
% peaks based on the distance.

%  - Again we divide between negative and positive angles
%  - Supress all peaks already found.
% We get only the line corresponding to the main degrees positive and
% negative
     
      angles = unique(ipeaks(:,2)); 
      npeaks = ipeaks;
      for i=1:length(angles)
          newpeaks = searchNewPeaksForAngle(R,ipeaks,angles(i));
          npeaks = [npeaks; newpeaks];
      end
      

end
%% 
% peaks - total peaks
% anglepos - angle of search
function [npeaks] = searchNewPeaksForAngle(R,ipeaks,anglepos)
        angleposind = find(ipeaks(:,2)==anglepos);
        signal_pos = R(:,90+anglepos);
        npeaks = [];
        if(isempty(angleposind)) % if we have dont have peaks in that angle
            return
        end
        pos_peaks = ipeaks(angleposind,1); % get positions
        pos_peaks = sortrows(pos_peaks,1); % order them by distance

        %% take the distances between the found lines to known where to search up and down.
        %% this could be also provided, but we calculate them based on the pattern.
        distances = pdist(pos_peaks);
        mindist = min(distances); % Minimum is the distance between two close peaks ^_ mindist__^____distances_____^__^
        distances = distances(distances>3*mindist);
        if(isempty(distances)) %% assuming a squared pattern we could try with the complementary
            comp = 90-anglepos;
            anglecompind = ipeaks(:,2)==comp;
            comp_peaks = ipeaks(anglecompind,1);
            comp_peaks = sortrows(comp_peaks,1);
            distances = pdist(comp_peaks); % repeat
            mindist = min(distances);
            distances = distances(distances>3*min(distances));
            if(isempty(distances)) % if no luck, return
                npeaks = [];
                return; 
            end;
        end;
        distances = sort(distances);
        md_pos = distances(1)+ mindist;
        %% supress the peaks already found and 100 px around.
        window = 100;
        for j=1:length(pos_peaks)
            indx = pos_peaks(j,1);
            signal_pos = supress(signal_pos,indx,window);
        end
        
        
        %% Start searching up
        %% I take the first point
        pos_1 = pos_peaks(1,1);
        % suppress everything that is not up
        % Original : ____^__^________^___^________^___^____
        % After    : ____^__^______________________________
        window = round(mindist*2.25);
        signal_1= supress_comp(signal_pos,pos_1-md_pos,window);
        % Take 2 possible peaks by NMS
        peaks_pos = nonmaxsup1Dt(signal_1,2,20,0.05*max(ipeaks(angleposind,3)));
        tpeaks_pos = [];
       
        if(peaks_pos(1)>0 && peaks_pos(2)>0)
          peaks_pos(:,3)=anglepos;
          tpeaks_pos = peaks_pos;
        end
        if(~isempty(tpeaks_pos))
            npeaks =[ npeaks; tpeaks_pos(:,2) tpeaks_pos(:,3)  tpeaks_pos(:,1)];
        end
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % I take the second point
         if(~isempty(angleposind) && length(angleposind)>3)
            pos_1 = pos_peaks(4,1);
            % in case we already have
            % ____^__^________^1___^2________^3___^4______^__^
         else
             pos_1 = pos_peaks(2,1);
         end
        % suppress everything that is not down
        % Original : ____^__^________^___^________^___^____
        % After    : _____________________________^___^_____
        
        window = round(mindist*2.25);
        signal_1= supress_comp(signal_pos,pos_1+md_pos,window);
        % Take 2 possible peaks 
        peaks_pos = nonmaxsup1Dt(signal_1,2,20,0.05*max(ipeaks(angleposind,3)));
        tpeaks_pos = [];
        if(peaks_pos(1)>0 && peaks_pos(2)>0)
          peaks_pos(:,3)=anglepos;
          tpeaks_pos = peaks_pos;
        end  
      if(~isempty(tpeaks_pos))
        npeaks =[ npeaks;  tpeaks_pos(:,2) tpeaks_pos(:,3) tpeaks_pos(:,1)];
      end
end

%%
function signal= supress_comp(signal,indx,window)
    for i=1:indx-window
            if(i>1 && i<length(signal)) 
                signal(i) = 0;
            end
    end
    for i=indx+window:length(signal)
            if(i>1 && i<length(signal)) 
                signal(i) = 0;
            end
    end
end
 
  
 function signal = supress(signal,indx,window)
    for i=indx-window:indx+window
            if(i>1 && i<length(signal)) 
                signal(i) = 0;
            end
    end
 
 end
 
 function [peaks] = nonmaxsup1Dt(signal,npeaks,window,thresh)
     
    k = 1;
    peaks = zeros(npeaks,2);
    while(npeaks>0)
          [num indx] = max(signal);
          if(num<thresh) break; end;
          peaks(k,1) = num;
          peaks(k,2) = indx;
          for i=indx-window:indx+window
            if(i>1 && i<length(signal)) 
                signal(i) = 0;
            end
          end
          npeaks = npeaks-1;
          k=k+1;
    end
 end