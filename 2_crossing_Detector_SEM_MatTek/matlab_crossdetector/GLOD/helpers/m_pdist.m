


function D = m_pdist(x)
     [rows,cols] = size(x);
     if(rows<2)
         D = x;
     end
     order = nchoosek(1:rows,2);
     Xi = order(:,1);
     Yi = order(:,2);
     X = x';
     diff = X(:,Xi) - X(:,Yi);
     D = sqrt (diff.^2);
end


