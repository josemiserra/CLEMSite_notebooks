function [npeaks,ic] = calibrateIntersections(iim,goodlines,ipeaks,folder)
%% Peak recalibration 
% Find intercept and slope
% same order as peaks pairs, for each peak, one line
ipoints= findIntersectionPoints(goodlines);
angles = unique(ipeaks(:,2));
mytree = KDTreeSearcher(ipoints(:,1:2));
ipoints_copy = ipoints;
[rows cols]=size(ipoints);
total = floor(rows/4);
pc = 1;
for k = 1:total 
    for m=1:rows
        if(ipoints_copy(m,1)>-inf) break;
        end
    end
    pNN = knnsearch(mytree,ipoints_copy(m,1:2),'K',4);
    ipoints_copy(pNN) = -inf;
    %average x and y
    avx = mean(ipoints(pNN,1));
    avy = mean(ipoints(pNN,2));
     
    xp = ipoints(pNN(1),1);
    yp = ipoints(pNN(1),2);
    xpi = ipoints(pNN(2),1);
    ypi = ipoints(pNN(2),2);
    
    is_in = checkpoints(iim,ipoints(pNN,:));
    add_original =~is_in; % if is_in, add_original is false
    
    if(is_in)
        wo = round(sqrt((xp-xpi)^2 + (yp-ypi)^2));
        wd = wo*2;
        w = wo*5;
        m_ipoints= adjustLines(w,wd,avx,avy,iim,angles,folder); 
        xp = m_ipoints(1,1);
        yp = m_ipoints(1,2);        
        for j = 1:length(m_ipoints)
            xpi =  m_ipoints(j,1);
            ypi =  m_ipoints(j,2);
            distp(j,1) = round(sqrt((xp-xpi)^2 + (yp-ypi)^2));
            distp(j,2) = j;
        end;
        % Sort by distance 
        distp = sortrows(distp,1);
        % Now we need to check that these points, if by chance the
        % distance between them is much more LESS or much MORE than the original, we
        % cancel the adjustment
        
        % No matter the choice, original points cannot be ]0.5*wo,1.5*wo[ far away
        % from the original centroid of points, since it is a fine calibration
        if(testSquareFails(m_ipoints,wo,w,avx,avy,iim,distp))
            fprintf('First test failed for %d %d .\n',avx,avy);
            w = wo*7;
            m_ipoints= adjustLines(w,wd,avx,avy,iim,angles,folder); 
            xp = m_ipoints(1,1);
            yp = m_ipoints(1,2);        
            for j = 1:length(m_ipoints)
               xpi =  m_ipoints(j,1);
               ypi =  m_ipoints(j,2);
               distp(j,1) = round(sqrt((xp-xpi)^2 + (yp-ypi)^2));
               distp(j,2) = j;
            end;
            if(testSquareFails(m_ipoints,wo,w,avx,avy,iim,distp))
                fprintf('Second test failed for %d %d .\n',avx,avy);
                add_original = true;
            end
        end
      
    end;
        %%
        
        if(~add_original)
            cpoints = [0 0];
            indp = distp(1:4,2);
            % Last and first are opposite
            p1 = m_ipoints(indp(1),:);
            p2 = m_ipoints(indp(4),:);
            p3 = m_ipoints(indp(2),:);
            p4 = m_ipoints(indp(3),:);
            hold on;
            plot(p1(1),p1(2),'x','Color','yellow');
            plot(p2(1),p2(2),'x','Color','yellow');
            plot(p3(1),p3(2),'x','Color','yellow');
            plot(p4(1),p4(2),'x','Color','yellow');
            cpoints(1) = p1(1)+round((p2(1)-p1(1))/2);
            cpoints(2) = p1(2)+round((p2(2)-p1(2))/2);
            plot(cpoints(1),cpoints(2),'.','LineWidth',2,'Color','green');
            fname = strcat(folder,'cross_');
            numr = clock;
            fname = strcat(fname,num2str(numr(6)));
            fname = strcat(fname,'.tif');
            saveas(gcf,fname,'tif');
            hold off;
            hw = w*0.5;
            tx = round(avx-hw-1);
            ty = round(avy-hw-1);
            if(tx<0) tx = 0; end;
            if(ty<0) ty = 0; end;
            
            fpoints(pc,1) = p1(1)+tx;
            fpoints(pc,2) = p1(2)+ty;
            fpoints(pc,3) = k;
            fpoints(pc+1,1) = p2(1)+tx;
            fpoints(pc+1,2) = p2(2)+ty;
            fpoints(pc+1,3) = k;
            fpoints(pc+2,1) = p3(1)+tx;
            fpoints(pc+2,2) = p3(2)+ty;
            fpoints(pc+2,3) = k;
            fpoints(pc+3,1) = p4(1)+tx;
            fpoints(pc+3,2) = p4(2)+ty;
            fpoints(pc+3,3) = k;
            fpoints(pc+4,1) = cpoints(1)+tx;
            fpoints(pc+4,2) = cpoints(2)+ty;
            fpoints(pc+4,3) = k;
            pc = pc+5;
        end
    if(add_original)
        fpoints(pc,1) = ipoints(pNN(1),1);
        fpoints(pc,2) = ipoints(pNN(1),2);
        fpoints(pc,3) = k;
        fpoints(pc+1,1) = ipoints(pNN(2),1);
        fpoints(pc+1,2) = ipoints(pNN(2),2);
        fpoints(pc+1,3) = k;
        fpoints(pc+2,1) = ipoints(pNN(3),1);
        fpoints(pc+2,2) = ipoints(pNN(3),2);
        fpoints(pc+2,3) = k;
        fpoints(pc+3,1) = ipoints(pNN(4),1);
        fpoints(pc+3,2) = ipoints(pNN(4),2);
        fpoints(pc+3,3) = k;
        fpoints(pc+4,1) = mean(ipoints(pNN,1));
        fpoints(pc+4,2) = mean(ipoints(pNN,2));
        fpoints(pc+4,3) = k;
        pc = pc+5;
    end
end

show(iim);
hold on;

for k = 1:length(fpoints)
    plot(round(fpoints(k,1)),round(fpoints(k,2)),'.','Color','yellow');
end


for k=5:5:length(fpoints)
    plot(fpoints(k,1),fpoints(k,2),'.','LineWidth',2,'Color','blue');
end

fname = strcat(folder,'/sketch.tif');
set(gca,'position',[0 0 1 1]);
saveas(gcf,fname,'tif');

npeaks = fpoints;

end
function [checked] = checkpoints(ic,points)
   [iLength, iWidth] = size(ic);
   checked = 1;
   for i=1:length(points)
    if(points(i,1)<1) checked = 0; return; end;
    if(points(i,2)<1) checked = 0; return; end;
    if(points(i,1)>iWidth) checked = 0; return; end;
    if(points(i,2)>iLength) checked = 0; return; end;
   end
end   
function [m_ipoints]= adjustLines(w,wd,avx,avy,iim,angles,folder)
        windowSize = 3;
        b = (1/windowSize)*ones(1,windowSize);
        a = 1;
        hw = w*0.5;
        rect = [round(avx-hw) round(avy-hw) w w]; % this needs to be changed relative to the image magnification
        ic = imcrop(iim, rect); % crops the image I. rect is a four-element position vector[xmin ymin width height] 
        ic = histeq(ic);
        
        I = gaussfilt(ic,1.2);
        [PC, or] = phasecongmono(I);
        nm = nonmaxsup(PC, or, 1.5); 
        bw = hysthresh(nm,0.005, 0.2);
        
%         [grad,or]=canny(ic,0);
%         nm = nonmaxsup(grad, or, 3);
%         med= double(median(median(ic)))/255.0;
%         factor_a = 20*med;
%         factor_b = 0.5*med;
%         bw = hysthresh(nm,factor_b, factor_a);
        
       % bw = (bw1+bw2);
        show(bw);

        PR1 = projectionsOneE(bw,angles(1));
        PR1 = filter(b,a,PR1);
        len_pos1= round(length(PR1)/2);
        PR1 = supress_comp(PR1,len_pos1,wd);
        mp1 = nonmaxsup1D(PR1,2,round(wd*0.25));
        
        PR2 = projectionsOneE(bw,angles(2));
        PR2 = filter(b,a,PR2);
        len_pos2= round(length(PR2)/2);
        PR2 = supress_comp(PR2,len_pos2,wd);
        mp2 = nonmaxsup1D(PR2,2,round(wd*0.25));

        mp(1,1)=mp1(1,2);
        mp(1,2)=angles(1);
        mp(2,1)=mp1(2,2);
        mp(2,2)=angles(1);
        mp(3,1)=mp2(1,2);
        mp(3,2)=angles(2);
        mp(4,1)=mp2(2,2);
        mp(4,2)=angles(2);
        
        rlines = tlines(ic,mp); 
        show(ic); 
        hold on;
        for k=1:2:length(rlines)
            p1 = rlines(k,:);
            p2 = rlines(k+1,:);
            line( [p1(1) p2(1)],[p1(2) p2(2)],'LineWidth',2,'Color','red');
        end;
        
        m_ipoints= findIntersectionPoints(rlines);
        % In case that angles are a little bit moved, we need to remove
        % outliers
        if(length(m_ipoints)>4)
            D = m_pdist(m_ipoints(:,1:2));            
            %# find the indices corresponding to each distance
            tmp = ones(size(m_ipoints,1));
            tmp = tril(tmp,-1); %# creates a matrix that has 1's below the diagonal
            %# get the indices of the 1's
            [rowIdx,colIdx ] = find(tmp);
            %# create the output
            out = [D',m_ipoints(rowIdx,1:2),m_ipoints(colIdx,1:2)];
            out = sortrows(out,1);
            m_ipoints= unique([out(1:4,2:3);out(1:4,4:5)],'rows');
        end
        
end

%% 
function [failed]= testSquareFails(m_ipoints,wo,w,avx,avy,iim,distp)
        failed = false;
        [iLength, iWidth] = size(iim);
         if( abs(distp(2,1)-distp(3,1)) > 0.2*wo) % must be a square (more or less) wo*wo
            failed = true;
            return;
         end      
        for i = 2:3
            axp = m_ipoints(i,1);
            ayp = m_ipoints(i,2);
            if((avx-w*0.5)<0)
                dist_test = round(sqrt((w/2-axp)^2 + (w/2-ayp)^2)-abs(avx-w*0.5));
            elseif((avy-w*0.5)<0)
                dist_test = round(sqrt((w/2-axp)^2 + (w/2-ayp)^2)-abs(avy-w*0.5));
            elseif((avx+w*0.5)>iWidth)
                dist_test = round(sqrt((w/2-axp)^2 + (w/2-ayp)^2)-abs(iWidth-(avx+w*0.5)));
            elseif((avy+w*0.5)>iLength)
                dist_test = round(sqrt((w/2-axp)^2 + (w/2-ayp)^2)-abs(iLength-(avy+w*0.5)));
            else
                dist_test = round(sqrt((w/2-axp)^2 + (w/2-ayp)^2));
            end;
            if(distp(i,1)<wo*0.5 ||distp(i,1)>wo*2|| dist_test>wo*1.5) %try again with a bigger distance squared
                failed = true;
              break;
            end
        end
       
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