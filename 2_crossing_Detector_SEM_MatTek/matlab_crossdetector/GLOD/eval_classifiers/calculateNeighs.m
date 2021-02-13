function [copylabels] = calculateNeighs(cletter,nneig)
    %% Generate closest set of points, given a letter, which letters are around mine
    xlabels='0123456789abcdefghijk+';
    ylabels='ABCDEFGHIJKLMNOPQRSTUVWXYZ+';

    poslminus =  find(xlabels==cletter(1));
    poslplus  =  find(ylabels==cletter(2));
    k = 1;
    for i=poslminus-nneig:poslminus+nneig
        if(i<length(xlabels)&& i>0)
            minusl = xlabels(i);
            for j=poslplus-nneig:poslplus+nneig
                if(j<length(ylabels)&& j>0)
                    plusl = ylabels(j);
                    copylabels{k} = strcat(minusl,plusl);
                    k=k+1;
                end
            end
        end
    end
end