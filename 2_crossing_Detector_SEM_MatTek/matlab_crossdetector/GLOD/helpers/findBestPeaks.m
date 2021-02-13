function [npeaks,Rhigh,error] = findBestPeaks(FSWT,ORIENTIM,K,delta,gridsize,dispim,p_angles)
    error = 0;
    Rhigh = [];
    s_angles = [];
    if(~isempty(p_angles))
         for k=1:length(p_angles)
            if(p_angles(k)>0)
                newang = p_angles(k)-90;
            else
                newang = 90+p_angles(k);
            end
            s_angles = [s_angles; p_angles(k); newang];
         end
         Rhigh = projections(FSWT,ORIENTIM,K,delta,s_angles,5);
    else    
        [iLength, iWidth] = size(FSWT);
        red_im = imresize(FSWT,[iLength/4 iWidth/4],'nearest'); % important to keep values using NN
        red_or = imresize(ORIENTIM,[iLength/4 iWidth/4],'nearest'); % 

        R = projections(red_im,red_or,8,1,0,0); % 
        fprintf('Projections at low resolution. \n');   
        if(dispim)  
            THETA = (-90:1:89);
            n =length(THETA);
            [iLength, iWidth] = size(red_im);
            iDiag = sqrt(iLength^2 + iWidth^2);
            figure, imagesc(THETA,[1:iDiag],R); colormap(pink); colorbar;
            xlabel('\theta'); ylabel('x\prime');
        end; 
        % Select the most prevalent angles at low resolution
        npeaks = [];
        npeaks = detectPeaksNMS(R,8,'Threshold',0.25);
        sangles = unique(npeaks(:,2));
        if(isempty(sangles))
            Rhigh = projections(FSWT,ORIENTIM,K,delta,[],0);
        else
            % take all angles, but also the complementary ones
             for k=1:length(sangles)
                if(sangles(k)>0)
                    newang = sangles(k)-90;
                else
                    newang = 90+sangles(k);
                end
                sangles = [sangles; newang];
             end
            sangles = unique(sangles);
            fprintf('Projections at high resolution with angles: \n');
            fprintf('%d. \t',sangles);
            Rhigh = projections(FSWT,ORIENTIM,K,delta,sangles,5);
        end
    end   
    if(dispim)
        THETA = (-90:1:89);
        n =length(THETA);
        [iLength, iWidth] = size(FSWT);
        iDiag = sqrt(iLength^2 + iWidth^2);
        figure, imagesc(THETA,[1:iDiag],Rhigh); colormap(hot); colorbar;
        xlabel('\theta'); ylabel('x\prime');
    end;
    [npeaks,error] = getPeaks(Rhigh,gridsize); 
end