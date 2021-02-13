 function  [finalList,final_ind]=evaluateLists(elist,mlist,clabels)
    % We have the expected list and a list for each recognised letter
    [npic,nx,ny]=size(mlist);
    if(npic==1)
        % If there is only one picture, and the expected list is not far away
        % from this list, we take it. Otherwise we pick the expected list
        % Heuristic, how many labels? AT LEAST 4
        piclist = {mlist(1,:).letter};
        aup = compareLists(piclist,clabels);
        if(aup<5)
             % Less than five, we keep the 
             fprintf('WARNING: Keeping original estimation! \n')
             finalList = elist;
             final_ind = -1;
             return
        else
           finalList = mlist(1,:);
           final_ind = 1;
           return
        end
    else
    % If there is more than one picture, compare all combinations 
    % and vote by similarity. From the voting matrix, we take the list with
    % biggest amount of votes. If there is a tie, then we calculate the
    % closest to the expected list. If the closest is not by a reasonable
    % shift, then the expected list is kept and a warn is given!!
        nset = nchoosek(1:npic,2);
        asame = zeros(length(nset),1);
        
        for i = 1:npic
            sset = sum(nset'==i);
            sset = logical(sset);
            sset = nset(sset,1:2);
            temps = [];
            [nx,ny] = size(sset);
            for j=1:nx
                temps(j)= isequal(mlist(sset(1),:),mlist(sset(2),:));
            end
            asame(i) = sum(temps);    
          %  if(nx==1)
          %      break;
          %  end
        end    
        [val,indg] = max(asame);
        fprintf('Total match of lists: %d from %d \n',val,npic);
        if(val == 0) % all of them are wrong
            % pic the closest to the expected list
            for i = 1:npic
                piclist = {mlist(i,:).letter};
                aup(i) = compareLists(piclist,clabels);
            end
            [val,indb] = max(aup);
            if(val<5)
                fprintf('WARNING: Keeping original estimation!\n')
                finalList = elist;
                final_ind = -1;
                return
            else
                finalList = mlist(indb,:,:);
                final_ind = indb;
                return
            end
        else
           finalList = mlist(indg,:,:);
           final_ind = indg;
           return
        end
     end
        
 end
 
 function [samevalues] = compareLists(list1,list2)
    samevalues = 0;
    for i=1:length(list2)
       letter = [list2{i}];
       samevalues = samevalues + sum(strcmp(list1, letter));
    end;
 
 end   

 
