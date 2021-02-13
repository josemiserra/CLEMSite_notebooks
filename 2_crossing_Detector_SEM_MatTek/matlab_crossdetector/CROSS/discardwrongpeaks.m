
function [goodpeaks,error] = discardwrongpeaks(R,ipeaks,gridsize)

%%%  Discard wrong peaks  %%%%
% Discard wrong peaks is based in the following facts:
%  - In a grid you will fing positive and negative angles
%  - Make pairs of positive and negative angles. It is a square, it is made
%    of two pairs of lines. We group first positive, then negative
%  - If we don't have enough for a square, we finish at that point.
%  Otherwise,  we check that the pairs sum 90 degrees complementary, so we
%  group by square. For each positive, it must exist a negative that
%  complements, and viceversa.
%  ------------------------------------------------------------------------
% Function ERRORS

% 6 - couldn´t find enough peaks fitting grid conditions
%--  Divide between positive and negative angles 
      goodpeaks = [];
      error = 0;
      angleposind = find(ipeaks(:,2)>0);
      anglenegind = find(ipeaks(:,2)<1);
      
      error = enoughpeaks(angleposind,anglenegind);
      if(error>0)
          if(error==1)
            anglepos = ipeaks(angleposind,:);
            anglepos = sortrows(anglepos,1);  
            [positivePairs] = findCrossSequence(anglepos,gridsize);
            [goodpeaks,error] = predictGrid(positivePairs,R,gridsize);
            return;
          end
          if(error ==2)
             angleneg = ipeaks(anglenegind,:);
             angleneg = sortrows(angleneg,1);
            [negativePairs] = findCrossSequence(angleneg,gridsize);
            [goodpeaks,error] = predictGrid(negativePairs,R,gridsize);
            return;
          end
          return;
      end
      
      
      anglepos = ipeaks(angleposind,:);
      angleneg = ipeaks(anglenegind,:);
      anglepos = sortrows(anglepos,1); 
      angleneg = sortrows(angleneg,1);
      
      %% We have 2 exceptions: 0 and -1, in that case, majority wins and is converted
      %  
        total_0 = length(angleneg(angleneg(:,2)==0,2));
        total_1 = length(anglepos(anglepos(:,2)==1,2));
        total_m1 = length(angleneg(angleneg(:,2)==-1,2));
        total_90 = length(anglepos(anglepos(:,2)==90,2))+length(anglepos(anglepos(:,2)==89,2)); % We consider 90 and 89
        total_m89 = length(angleneg(angleneg(:,2)==-89,2))+length(angleneg(angleneg(:,2)==-88,2)); % We consider -89 and -88
        
        if(total_0 > 0 && total_m1 > 0)
            if(total_90+total_0 > total_m89+total_1)                
                angleneg = [ angleneg; angleneg(angleneg(:,2) == -1,:)]; % move and delete
                anglepos(anglepos(:,2) == -1,:) = [];
                angleneg(angleneg(:,2) == -1,:) = 0;
            else
                angleneg = [ angleneg; angleneg(angleneg(:,2) == 0,:)]; % move and delete
                angleneg(angleneg(:,2) == 0,:) = [];
                angleneg(angleneg(:,2) == 0,:) = -1;
            end
        end
      %  and second the -89/90. 
        if(total_90 > 0 && total_m89 > 0)
           % R(:,1:3) = flipud(R(:,1:3)); % Invert -89 and -88 to be like 90 and repeat...
            
            if(total_90+total_0 >= total_m89+total_1) 
                % We find all negative, we search max          
                anglepos = [ anglepos; angleneg(angleneg(:,2) == -89,:)]; % move and delete
                anglepos = [ anglepos; angleneg(angleneg(:,2) == -88,:)]; % move and delete
                for k = 1:length(anglepos)
                    if(anglepos(k,2)==-89 || anglepos(k,2)==-88)
                        anglepos(k,1) = length(R)- anglepos(k,1);
                    end
                end
                anglepos(anglepos(:,2) == -89,2) = 90;                
                angleneg(angleneg(:,2) == -89,:) = [];     

                anglepos(anglepos(:,2) == -88,2) = 90;
                angleneg(angleneg(:,2) == -88,:) = [];
            else
                angleneg = [ angleneg; anglepos(anglepos(:,2) == 90,:)]; % move and delete
                angleneg = [ angleneg; anglepos(anglepos(:,2) == 89,:)]; % move and delete
                [rows,~]=size(angleneg);
                 for k = 1:rows
                    if(angleneg(k,2)==90 || angleneg(k,2)==89)
                        angleneg(k,1) = length(R)- angleneg(k,1);
                    end
                end
                angleneg(angleneg(:,2) == 90,2) = -89;
                anglepos(anglepos(:,2) == 90,:) = [];
                
                anglepos(anglepos(:,2) == 89,:) = [];
                angleneg(angleneg(:,2) == 90,2) = -89;
            end
            % We choose the sign based on majority
            %
        end
      
            
      %% 
      % now make 90 degrees pairs
      % Take first positive angles and compare with negative if they add
      % 90+/-5
     pos_ang = unique(anglepos(:,2));
     neg_ang = unique(angleneg(:,2));
     good_angles_pos = [];
     good_angles_neg = [];
     [nneg,~]=size(neg_ang);
     [npos,~]=size(pos_ang);
     for i=1:npos  
         for j=1:nneg
             nty = pos_ang(i)+abs(neg_ang(j));
             if(nty>85 && nty<95)
                 % good combination
                 good_angles_pos = union(good_angles_pos,pos_ang(i));
                 good_angles_neg = union(good_angles_neg,neg_ang(j));
             end
         end
     end
     
     good_ind_pos = [];
     for i =1:length(good_angles_pos)
        good_ind_pos = union(good_ind_pos, find(anglepos(:,2)==good_angles_pos(i)));
     end
     anglepos = anglepos(good_ind_pos,:);
     
     good_ind_neg = [];
     for i =1:length(good_angles_neg)
        good_ind_neg = union(good_ind_neg, find(angleneg(:,2)==good_angles_neg(i)));
     end
     angleneg = angleneg(good_ind_neg,:);
 
      error = enoughpeaks(anglepos,angleneg);
      if(error>0)
          return;
      end
      %% 
      %
 
      [positivePairs] = findCrossSequence(anglepos,gridsize);
      [negativePairs] = findCrossSequence(angleneg,gridsize);
      
      error = enoughpeaks(positivePairs,negativePairs);
      if(error>0)
          if(error==1)
            [goodpeaks,error] = predictGrid(positivePairs,R,gridsize);
            return;
          end
          if(error ==2)
            [goodpeaks,error] = predictGrid(negativePairs,R,gridsize);
            return;
          end
          return;
      end
      
      angpos = unique(positivePairs(:,2));
      angneg = unique(negativePairs(:,2));
      
      fangpos = mode(positivePairs(:,2));
      fangneg = mode(negativePairs(:,2));
      oldval = abs((fangpos - fangneg) -90);
      for i=length(angpos)
          for j=length(angneg)
           val = abs((angpos(i) - angneg(j))-90);
           if( val < oldval)
              fangpos = angpos(i);
              fangneg = angneg(j);
              break;
          end
          end
      end
      
      nty = fangpos - fangneg;
      goodpeaks = [ positivePairs; negativePairs];
      [px py] = size(positivePairs);
      for k = 1:px
           goodpeaks(k,2) = fangpos;
      end;
      [nx ny] = size(negativePairs);
      for k = px+1:px+nx
           goodpeaks(k,2) = fangneg;
      end;
       
      fprintf('Angle sum :%d \n',nty);
      if(nty<(85) && nty>(60))
                warning('The angle orientations are not 90 degrees. Adjust properly.');
       end
end
%-----------------------------------------------------------------------------
function [error] = enoughpeaks(pospeaks,negpeaks)
      error = 0;
      if(isempty(pospeaks)&&isempty(negpeaks))
          error = 3;
          return;
      end
      if(isempty(negpeaks))          
          if(length(pospeaks)<2)
            fprintf('WARNING:NO horizontal lines found. ');
            error = 4;
            return;
          else
            error = 1;
          end;
          return;
      end
      if(isempty(pospeaks))
          if(length(negpeaks)<2)
            fprintf('WARNING:NO vertical lines found. ');
            error = 5;
            return 
          else
            error = 2;
          end;
          return;
      end    
     
     if(length(pospeaks)<2 && length(negpeaks)>=2)
             error = 2;
             return;
     end
        
     if(length(pospeaks)>=2 && length(negpeaks)<2)
             error = 1;
             return;
     end

 end
      

