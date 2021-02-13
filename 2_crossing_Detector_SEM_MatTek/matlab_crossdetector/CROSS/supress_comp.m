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