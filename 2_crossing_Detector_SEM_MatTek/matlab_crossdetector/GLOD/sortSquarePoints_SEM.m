%% Adapted to mirrored GRID!!!
  function [slist,sortorder] =  sortSquarePoints_SEM(plist,angneg)
        % calculate centroid
        cx = mean(plist(:,1));
        cy = mean(plist(:,2));
        tpoints(:,1) = plist(:,1)-cx; % move x points to the center
        tpoints(:,2) = plist(:,2)-cy; % move y points to the center
        irotmat =[ cosd(angneg) sind(angneg); -sind(angneg) cosd(angneg) ];
        tpoints = irotmat*tpoints(:,1:2)';
        tpoints = tpoints';
        sortorder = zeros(1,4);
        for i = 1:4
            if(tpoints(i,1)<0&&tpoints(i,2)<0) % -- means top left corner
                sortorder(2)=i;
            else if (tpoints(i,1)>=0&&tpoints(i,2)>=0)
                 sortorder(4)=i;
                else if (tpoints(i,1)>=0&&tpoints(i,2)<0)
                  sortorder(1) = i;
                    else
                        sortorder(3) = i;
                    end
                end
            end
        end
        slist = plist(sortorder,:);
        
  end