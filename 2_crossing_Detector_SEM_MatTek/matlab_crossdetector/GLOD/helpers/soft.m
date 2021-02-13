
function [prepro,bwedge,orientim,reliability,fswt] = soft(varargin)
%   [prepro,bwedge,orientim,reliability,fswt] = SOFT(I,PARAM1,VAL1,PARAM2,VAL2...) sets various parameters.
%   Ex.[P,BWEDGE,ORIENT,REL,SWT] = soft(cdata,'canthresh',0.06,'sigma',1.0,'clahe',false,'dispim',true);
%   Preprocessing needed for the projector,
%   Parameter names can be abbreviated, and case does not matter. Note: +CT means, increases computation
%   time.Each string parameter is followed by a value as indicated below:

%
%   'wiener'       Two-element vector of positive integers: [M N].
%                  [M N] specifies the number of tile rows and
%                  columns.  Both M and N must be at least 2. 
%                  The total number of image tiles is equal to M*N.
%                  If the lines are too thin, it is used to dilate them.
%                  Default: [2 2].
%                  Use 0 to not execute the wiener filter.
%    
%   'strokeWidth'  When the Stroke Width Transform is executed, for each
%                  pixel, rays are created from the pixel to the next
%                  change of gradient. If your stroke is big, use a bigger
%                  width.(strokeWidth big, ++CT)
%                  Default: 20.

%   'canthresh'    Automatic canny thresholding is performed using an
%                  iterative loop. If the percentage of white pixels is bigger than a
%                  threshold,then we are assuming the image is getting more
%                  and more clutter.
%                  Default: 0.075, means a 7.5% of the pixels is white.
%
%   'Sigma'        Preprocessing gaussian filter. Helps with noisy images
%                  of after CLAHE. Values between 0 and 2 are recommended.
%
%                  Default: 0 (not applied).
%   'clahe'        If true, CLAHE (automatic brightness and contrast
%                  balance) is applied.
%                  Default: False.
%   'dispim'         If true, images are shown. If false,no images are shown.
%                  Default: True
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Return values:

%  prepro - image after preprocessing (clahe and gaussian filter)
%  bwedge - image after automatic canny filtering
%  orientim - image with ridge orientations
%  reliability - probabilistic plot of orientations
%  fswt - stroke width transform after cleansing

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[I,strokeWidth,wiener,canthresh,sigma,clahe,dispim] = parseInputs(varargin{:});



if(clahe) 
    % This values can be changed to the default ones without any trouble
    % The only condition is that the number of bins must be small or the
    % image becomes too noisy 
    I = adapthisteq(I,'ClipLimit',0.02,'NumTiles',[32 32],'NBins',64,'Distribution','rayleigh','Alpha',0.7);
    fprintf('-  CLAHE:  TRUE \n');
else
    fprintf('-  CLAHE: FALSE \n'); 
end;

if(sigma>0)
    I = gaussfilt(I,sigma);
end

fprintf('Preprocessing done: \n');
fprintf('-  Gaussian Filter : %f \n',sigma);


if(dispim) 
    show(I);
end;
prepro = I;

%%%% CANNY automatic filtering  %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[grad,or]=canny(I,2);
nm = nonmaxsup(grad, or, 3);
autoedge = autocanny(nm,canthresh);

if(dispim)
    show(autoedge); 
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  bwedge = bwmorph(autoedge, 'spur', 5);
  J = wiener2(bwedge,wiener);  %% Border enhancement
  fprintf('Wiener filter done \n');
  bwedge = J;
  if(dispim) show(J); end;
    
 % Determine ridge orientations
  [orientim, reliability] = ridgeorient(grad, 1, 5, 5);
   % Determine ridge reliability (granularity fine, we want probabilities)
  [~, reliability2] = ridgeorient(grad, 1, 3, 3); % Here was I
  % Remove all the regions when disorder of orientations is bigger
  reliability2(reliability2<0.5)=0;
  tl = J.*reliability2; %Enhance the bw image removing disordered regions
  
  CC = bwconncomp(tl,8);
  numPixels = cellfun(@numel,CC.PixelIdxList);
  [w,h] = size(tl);
  for i = 1:length(numPixels)
            if(numPixels(i)< 0.04*w)
                tl(CC.PixelIdxList{i}) = 0;
            end
  end
  
  
  fprintf('Orientations done \n');
  if(dispim)
    plotridgeorient(orientim, 20,I, 2);
    show(reliability2,6);
  end;
  if(strokeWidth>0)
    fprintf('Starting SWT with strokeWidth of %d. \n',strokeWidth);
    tos = SWTtotal(I,tl,orientim,strokeWidth);
    fprintf('SWT done \n');
    if(dispim)
      show(tos)
     end;
    fprintf('Removing ill components \n');
    % Now we open the image
    fswt = cleanswt2(tos,tl);
    fprintf('Removing done \n');
  end
 
 if(dispim)
      show(fswt);
 end;
end
%-----------------------------------------------------------------------------

function [I,strokeWidth,wiener,canthresh,sigma,clahe,dispim] = parseInputs(varargin)

narginchk(1,13);

I = varargin{1};
validateattributes(I, {'uint8', 'uint16', 'double', 'int16', 'single'}, ...
              {'real', '2d', 'nonsparse', 'nonempty'}, ...
              mfilename, 'I', 1);

% convert int16 to uint16
if isa(I,'int16')
  I = int16touint16(I);
  int16ClassChange = true;
else
  int16ClassChange = false;
end

if any(size(I) < 2)
  error(message('images:softd:inputImageTooSmall'))
end

%Other options
%%%%%%%%%%%%%%

%Set the defaults
        wiener = [2 2];
        canthresh = 0.05; % 0.075 for SEM 
        strokeWidth = 20;
        sigma = 0;
        clahe = false;
        dispim = true;
            
            
validStrings = {'strokeWidth','wiener',...
                'canthresh','sigma','clahe','dispim'};


if nargin > 1
  done = false;

  idx = 2;
  while ~done
    input = varargin{idx};
    inputStr = validatestring(input, validStrings,mfilename,'PARAM',idx);

    idx = idx+1; %advance index to point to the VAL portion of the input 

    if idx > nargin
      error('Missing Value:'+inputStr)
    end
    
    switch inputStr

     case 'wiener'
       wiener = varargin{idx};
       validateattributes(wiener, {'double'}, {'real', 'vector', ...
                           'integer', 'finite','positive'},...
                     mfilename, inputStr, idx);
       if(any(size(wiener) ~= [1,2]))
         error('Invalid input vector for Wiener. Try [2 2]:'+inputStr)
       end
       
     case 'strokeWidth'
        strokeWidth = varargin{idx};      
        validateattributes(strokeWidth, {'double'}, {'scalar','real','integer',...
                          'positive'}, mfilename, 'strokeWidth', idx);
     
        if strokeWidth > 1024
            error('Invalide Stroke Width, too big :'+inputStr)
        end
      
        if strokeWidth < 5
            error('Invalid Stroke Width, too small :'+inputStr)
         end
     case 'canthresh'
      canthresh = varargin{idx};
      validateattributes(canthresh, {'double'}, ...
                    {'scalar','real','nonnegative'},...
                    mfilename, 'canthresh', idx);
      
      if canthresh > 1
        error('Invalid Threshold, cannot be bigger than 1:'+canthresh)
      end
     
     
  
     case 'sigma'
      sigma = varargin{idx};
      validateattributes(sigma, {'double'},{'scalar','real',...
                          'nonnan','nonnegative','finite'},...
                    mfilename, 'sigma',idx);
     
      if sigma > 5
        error('Invalid Sigma Value, bigger than 5.:'+inputStr)
      end
     case 'clahe'
      clahe = varargin{idx};
      validateattributes(dispim, {'logical'},{},...
                    mfilename, 'clahe',idx);
      
     case 'dispim'
      dispim = varargin{idx};
      validateattributes(dispim, {'logical'},{},...
                    mfilename, 'dispim',idx);
      
     otherwise
      error('Bad input string') %should never get here
    end
    
    if idx >= nargin
      done = true;
    end
    
    idx=idx+1;
  end
end

end
        