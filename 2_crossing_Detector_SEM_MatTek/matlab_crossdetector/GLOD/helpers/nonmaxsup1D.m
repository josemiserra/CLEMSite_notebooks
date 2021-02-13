function [peaks] = nonmaxsup1D(signal,npeaks,window)
     
    k = 1;
    peaks = zeros(npeaks,2);
    while(npeaks>0)
          [num indx] = max(signal);
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
    