function peaks = detectPeaksNMS(varargin)
%FindPEAKS Identify peaks in SOFT transform.
%   PEAKS = detectPeaksNMS(H,NUMPEAKS) locates peaks in the Hough 
%   transform matrix, H, generated by the HOUGH function. NUMPEAKS 
%   specifies the maximum number of peaks to identify. PEAKS is 
%   a Q-by-2 matrix, where Q can range from 0 to NUMPEAKS. Q holds
%   the row and column coordinates of the peaks. If NUMPEAKS is 
%   omitted, it defaults to 1.
%
%   PEAKS = detectPeaksNMS(...,PARAM1,VAL1,PARAM2,VAL2) sets various 
%   parameters. Parameter names can be abbreviated, and case 
%   does not matter. Each string parameter is followed by a value 
%   as indicated below:
%
%   'Threshold' Nonnegative scalar.
%               Values of H below 'Threshold' will not be considered
%               to be peaks. Threshold can vary from 0 to Inf.
%   
%               Default: 0.5*max(H(:))
%
%   'NHoodSize' Two-element vector of positive odd integers: [M N].
%               'NHoodSize' specifies the size of the suppression
%               neighborhood. This is the neighborhood around each 
%               peak that is set to zero after the peak is identified.
%
%               Default: smallest odd values greater than or equal to
%                        size(H)/50.
%
%   Class Support
%   -------------
%   H is the output of the projections function. NUMPEAKS is a positive
%   integer scalar.
%
%   Example
%   -------


[h, numpeaks, threshold, nhood] = parseInputs(varargin{:});

% initialize the loop variables
done = false;
hnew = h;
nhood_center = (nhood-1)/2;
peaks = [];

while ~done
  [dummy max_idx] = max(hnew(:)); %#ok
  [p, q] = ind2sub(size(hnew), max_idx);
  
  p = p(1); q = q(1);
  if hnew(p, q) >= threshold
    
    if(q == 180 || q == 179)
        hnew(:,1:3) = flipud(hnew(:,1:3)); % Invert -89 and -88 to be like 90
    end
    if(q == 1 || q == 2)
        hnew(:,178:180) = flipud(hnew(:,178:180)); % Invert -89 and -88 to be like 90
    end
    peaks = [peaks; [p q hnew(p,q)]]; %#ok<AGROW> % add the peak to the list
    % Suppress this maximum and its close neighbors.
    p1 = p - nhood_center(1); p2 = p + nhood_center(1);
    q1 = q - nhood_center(2); q2 = q + nhood_center(2);
    % Create a square around the maxima to be supressed
    [qq, pp] = meshgrid(q1:q2, max(p1,1):min(p2,size(h,1)));
    pp = pp(:); qq = qq(:);
    
    % For coordinates that are out of bounds in the theta
    % direction, we want to consider that is circular
    % for theta = +/- 90 degrees.
    theta_too_low = find(qq < 1);
    qq(theta_too_low) = size(h, 2) + qq(theta_too_low);
   % pp(theta_too_low) = size(h, 1) - pp(theta_too_low) + 1;
    theta_too_high = find(qq > size(h, 2));
    qq(theta_too_high) = qq(theta_too_high) - size(h, 2);
 %   pp(theta_too_high) = size(h, 1) - pp(theta_too_high) + 1;
    
    % Convert to linear indices to zero out all the values.
     hnew(sub2ind(size(hnew), pp, qq)) = 0;
     if(q == 180 || q == 179)
        hnew(:,1:3) = flipud(hnew(:,1:3)); % After supress, return the signal to normality
     end
     if(q == 1 || q == 2)
        hnew(:,178:180) = flipud(hnew(:,178:180)); % Invert -89 and -88 to be like 90
     end
    done = size(peaks,1) == numpeaks;
  else
    done = true;
  end
  
end

if(isempty(peaks))
    return
end
peaks(:,2) = peaks(:,2) - 90;
 
%-----------------------------------------------------------------------------
function [h, numpeaks, threshold, nhood] = parseInputs(varargin)

narginchk(1,6);

h = varargin{1};
validateattributes(h, {'double'}, {'real', '2d','nonempty'}, ...
              mfilename, 'H', 1);

numpeaks = 1; % set default value for numpeaks

idx = 2;
if nargin > 1
  if ~ischar(varargin{2})
    numpeaks = varargin{2};
    validateattributes(numpeaks, {'double'}, {'real', 'scalar', 'integer', ...
                        'positive', 'nonempty'}, mfilename, 'NUMPEAKS', 2);
    idx = 3;
  end
end

% Initialize to empty
nhood = [];
threshold = [];

% Process parameter-value pairs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
validStrings = {'Threshold','NHoodSize'};

if nargin > idx-1 % we have parameter/value pairs
  done = false;

  while ~done
    input = varargin{idx};
    inputStr = validatestring(input, validStrings,mfilename,'PARAM',idx);
    
    idx = idx+1; %advance index to point to the VAL portion of the input 
    
    if idx > nargin
      error(message('images:houghpeaks:valForhoughpeaksMissing', inputStr))
    end
    
    switch inputStr
      
     case 'Threshold'
      threshold = varargin{idx};
      validateattributes(threshold, {'double'}, {'real', 'scalar','nonnan' ...
                          'nonnegative'}, mfilename, inputStr, idx);
     
     case 'NHoodSize'
      nhood = varargin{idx};
      validateattributes(nhood, {'double'}, {'real', 'vector', ...
                          'finite','integer','positive','odd'},...
                    mfilename, inputStr, idx);
      
      if (any(size(nhood) ~= [1,2]))
        error(message('images:houghpeaks:invalidNHoodSize', inputStr))
      end
      
       if (any(nhood > size(h)))
        error(message('images:houghpeaks:tooBigNHoodSize', inputStr))
      end     
      
     otherwise
      %should never get here
      error(message('images:houghpeaks:internalError'))
    end
    
    if idx >= nargin
      done = true;
    end
    
    idx=idx+1;
  end
end

% Set the defaults if necessary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(nhood)
  nhood = size(h)/50; 
  nhood = max(2*ceil(nhood/2) + 1, 1); % Make sure the nhood size is odd.
end

if isempty(threshold)
  threshold = 0.5 * max(h(:));
end