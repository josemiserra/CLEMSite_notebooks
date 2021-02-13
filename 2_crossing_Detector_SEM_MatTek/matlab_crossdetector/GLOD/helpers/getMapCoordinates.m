function [mapcoordinates] =getMapCoordinates(lnames)

xlabels='0123456789abcdefghijk_';
ylabels='ABCDEFGHIJKLMNOPQRSTUVWXYZ_';
rows = length(lnames);
mapcoordinates = zeros(rows,2);
for k=1:rows
     cletter = lnames(k,:);
        poslminus =  find(xlabels==cletter(1))-1;
        poslplus  =  find(ylabels==cletter(2))-1;
        ext_x = 0;
        ext_y =0;
        if(length(cletter)>2)
            if(str2double(cletter(4))==1)
                ext_x = -1;
                ext_y = -1;
            else
               if((str2double(cletter(4))==2))
                ext_x = +1;
                ext_y = -1; 
               else
                if((str2double(cletter(4))==4))
                ext_x = -1;
                ext_y = +1;  
                else
                  if((str2double(cletter(4))==3))
                  ext_x = +1;
                  ext_y = +1;
                  end
                end
               end
            end
        end
        mapcoordinates(k,1)= poslminus*10+ext_x;
        mapcoordinates(k,2)= poslplus*10+ext_y;
end