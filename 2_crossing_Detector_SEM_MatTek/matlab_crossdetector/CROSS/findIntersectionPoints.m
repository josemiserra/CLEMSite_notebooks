function [ipoints]= findIntersectionPoints(goodlines)
    [tl, ln] = size(goodlines);
    tl = tl/2;
    % Find inner square intersection points
    % for each line
    j = 1;
    slope = zeros(tl,1);
    intercept = zeros(tl,1);
    for k = 1:tl
    % take all points related to that line
     p1 = goodlines(j,:);
     p2 = goodlines(j+1,:);
     j= j+2;
     slope(k) = round(atand((p2(2)-p1(2))/(p2(1)-p1(1))));
     if(isnan(slope(k))) slope(k)= 0; end;
     if(slope(k)==90 || slope(k)==-90)
        intercept(k) = intersect(p1,p2);
     else
        intercept(k) = round((p1(2) - tand(slope(k))*p1(1)));     
     end;
         
    end
    % Now I have all my lines
    i=1;
    for k = 1:tl
        for  j = (k+1):tl
            if(slope(k)~=slope(j))
                % now find the intersection
                if(slope(k)==90 || slope(k)==-90)
                    xp = intercept(k);
                    yp = intercept(j);
                else
                    if(slope(j)==90 || slope(j)==-90)
                        xp = intercept(j);
                        yp = intercept(k);
                    else
                        xp =(intercept(j)-intercept(k))/(tand(slope(k))-tand(slope(j)));
                        yp = (tand(slope(k))*xp)+intercept(k);
                    end
                end
             ipoints(i,1) = round(xp);
             ipoints(i,2) = round(yp);
             ipoints(i,3) = k;
             ipoints(i,4) = j;
             i = i+1;
            end  
        end
    end
