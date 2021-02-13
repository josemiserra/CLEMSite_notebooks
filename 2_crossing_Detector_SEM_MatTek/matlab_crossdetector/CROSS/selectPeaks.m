function [npeaks]=selectPeaks(R)
%% select right peaks from the sinogram
%  It starts finding the peaks by NMS. This function is the same that is
%  used in the Hough transform sinogram.
%  The most 12 promiment peaks bigger than 0.5*max are found.
ipeaks = 0;
peaks = detectPeaksNMS(R,12);  
threshold = min(peaks(:,3));

fprintf('Finding peaks by non maximum supression \n');
while(threshold>0.05)
    peaks = findpeaks2(R,12,'Threshold',threshold); % We need margin for noise!
    peaks(:,2) = peaks(:,2)-1;
    if(length(peaks)<8)
        threshold = threshold - 0.025;
        continue;
    end;
    ipeaks = discardwrongpeaks(peaks);
    if(~((length(find(ipeaks(:,2)>-1))>=4 && length(find(ipeaks(:,2)<0))>=4))) % + and - add 8 or more
        threshold = threshold - 0.025;
    else
         break; %% Avoid inequilibrium between + and -
    end;
    fprintf('%d peaks found.\n',length(ipeaks));
end
  fprintf('Initially, %d peaks from 8(max) found.\n',length(ipeaks));
  if(length(ipeaks)<8)
      fprintf('Insufficient peaks found.\n Try to improve your image SNR, reorient your image 45 degrees or adjust B&C.');
      return;
  end

  % This are supossed to be good peaks. We can try now to get the weak
  % peaks based on the distance 
  
  % We get only the line corresponding to the main degrees positive and
  % negative
     
      angleposind = find(ipeaks(:,2)>-1);
      
      if(isempty(angleposind))
          error('No positive angles found. Reorient your image 45 degrees and adjust B&C.'); %should never get here
      end
      angleposind = angleposind(1:4);
      anglepos = ipeaks(angleposind(1),2);
      
      anglenegind = find(ipeaks(:,2)<0);
      if(isempty(anglenegind))
          error('No negative angles found. Reorient your image 45 degrees and adjust B&C.'); %should never get here
      end
      anglenegind = anglenegind(1:4);
      angleneg = ipeaks(anglenegind(1),2);
      
      signal_pos = R(:,90+anglepos);
      signal_neg = R(:,90+angleneg);
      
      pos_peaks = ipeaks(angleposind,1);
      pos_peaks = sortrows(pos_peaks,1);
      neg_peaks = ipeaks(anglenegind,1);
      neg_peaks = sortrows(neg_peaks,1);
      % supress the 8 already found and 100 px around.
      window = 100;
      for j=1:length(pos_peaks)
        indx = pos_peaks(j,1);
        signal_pos = supress(signal_pos,indx,window);
      end
      for j=1:length(neg_peaks)
        indx = neg_peaks(j,1);
        signal_neg = supress(signal_neg,indx,window);
      end
      % take the max distance between the 4 lines
      md_pos = max(pdist(pos_peaks));
      md_neg = max(pdist(neg_peaks));
      tpeaks_pos = [];
      tpeaks_neg = [];
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % I take the first point
      pos_1 = pos_peaks(1,1);
      % suppress everything that is not up
      window = round(md_pos*0.333);
      signal_1= supress_comp(signal_pos,pos_1-md_pos,window);
      % Take 2 possible peaks 
      peaks_pos = nonmaxsup1Dt(signal_1,2,20,0.05*max(ipeaks(angleposind,3)));
       if(peaks_pos(1)>0 && peaks_pos(2)>0)
          peaks_pos(:,3)=anglepos;
          tpeaks_pos = [ tpeaks_pos; peaks_pos];
      end          
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % I take the second point
      pos_1 = pos_peaks(4,1);
      % suppress everything that is not up
      window = round(md_pos*0.333);
      signal_1= supress_comp(signal_pos,pos_1+md_pos,window);
      % Take 2 possible peaks 
      peaks_pos = nonmaxsup1Dt(signal_1,2,20,0.05*max(ipeaks(angleposind,3)));
       if(peaks_pos(1)>0 && peaks_pos(2)>0)
          peaks_pos(:,3)=anglepos;
          tpeaks_pos = [ tpeaks_pos; peaks_pos];
      end          
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      % I take negatives
      pos_1 = neg_peaks(1,1);
      % suppress everything that is not up
      window = round(md_neg*0.333);
      signal_1= supress_comp(signal_neg,pos_1-md_neg,window);
      % Take 2 possible peaks 
      peaks_neg = nonmaxsup1Dt(signal_1,2,20,0.05*max(ipeaks(anglenegind,3)));
      if(peaks_neg(1)>0 && peaks_neg(2)>0)
          peaks_neg(:,3)=angleneg;
          tpeaks_neg = [ tpeaks_neg; peaks_neg];
      end          
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % I take negatives
      pos_1 = neg_peaks(4,1);
      % suppress everything that is not up
      window = round(md_neg*0.333);
      signal_1= supress_comp(signal_neg,pos_1+md_neg,window);
      % Take 2 possible peaks 
      peaks_neg = nonmaxsup1Dt(signal_1,2,20,0.05*max(ipeaks(anglenegind,3)));
       if(peaks_neg(1)>0 && peaks_neg(2)>0)
          peaks_neg(:,3)=angleneg;
          tpeaks_neg = [ tpeaks_neg; peaks_neg];
      end          
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      npeaks = [];
      if(~isempty(tpeaks_pos))
        npeaks =[ npeaks; tpeaks_pos(:,2) tpeaks_pos(:,3)];
      end
      if(~isempty(tpeaks_neg))
        npeaks =[ npeaks; tpeaks_neg(:,2) tpeaks_neg(:,3)];
      end
        
        npeaks =[ npeaks; ipeaks(angleposind,1) ipeaks(angleposind,2); ipeaks(anglenegind,1) ipeaks(anglenegind,2)];       
        fprintf('A total of %d peaks found.\n',length(npeaks));
end

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