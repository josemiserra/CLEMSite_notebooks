
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
[I,strokeWidth,wiener,cantresh,sigma,clahe,dispim] = parseInputs(varargin{:});

I = adapthisteq(I,'ClipLimit',0.07,'NumTiles',[32 32],'NBins',128);
I = gaussfilt(I,sigma);
prepro = I;
%%%% CANNY automatic filtering  %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[grad,or]=canny(I,2);
nm = nonmaxsup(grad, or, 1.5);
med= double(median(median(nm(nm>0))));

%  [PC, or] = phasecongmono(cdata);
%   nm = nonmaxsup(PC, or, 1.5); 
%   bw = hysthresh(nm, 0.1, 0.3);

factor_a = 0.9;
factor_b = 0.4;
max_factor = max(max(nm));
lenm = size(nm);

value = 1;
while(value>cantresh) 
    factor_a = factor_a + 0.1;
    if(factor_a*med > max_factor)
        break; 
    end;
    bwedge = hysthresh(nm, factor_b*med,factor_a*med);
    value = sum(sum(bwedge))/(lenm(1)*lenm(2));
end
factor_a = factor_a - 0.15;
value = 1;
while(value>cantresh) 
    factor_a = factor_a + 0.01;
    if(factor_a*med > max_factor)
        break; 
    end;
    bwedge = hysthresh(nm, factor_b*med,factor_a*med);
    value = sum(sum(bwedge))/(lenm(1)*lenm(2));
end

 [PC, or] = phasecongmono(I,'k',6,'sigmaOnf',0.05,'nscales',6);
  or = uint8(or/20)*20;
  nm = nonmaxsup(PC, or, 1.5);
  bw = hysthresh(nm,0.005, 0.1);
  
  se23 = strel('line',4,45);
  se24 = strel('line',4,-45);
  
  se21 = strel('line',7,45);
  bw2 = imdilate(bw,se21);
 
   
  se22 = strel('line',7,-45);
  bw3 = imdilate(bw,se22);
  
  bw2 = imerode(bw2,se23);
  bw3 = imerode(bw3,se24);
  
  bw2 = bwareaopen(bw2, 50);
  bw3 = bwareaopen(bw3, 50);
   
  bwf = bw2 | bw3;
  se = strel('disk',1);
  bwf2 = imopen(bwf,se);
  se = strel('disk',3);
  bwf4 = imclose(bwf2,se);
  bw = imfill(bwf4,'holes');
  bw = bw - (bwareaopen(bw,700));

bwedge = bw | bwedge;
if(dispim)
    show(bwedge); 
end;
  
  J = wiener2(bwedge,wiener);  %% Border enhancement
  fprintf('Wiener filter done \n');
  if(dispim) 
      show(J); 
  end;
  
 % Identify ridge-like regions and normalise image
 % blksze = 30; thresh = 0.08;
 % [normim, mask] = ridgesegment(cdata, blksze, thresh);
 % show(normim,1);
    
 % Determine ridge orientations
  [orientim, reliability] = ridgeorient(grad, 1, 5, 5);
   % Determine ridge reliability (granularity fine, we want probabilities)
  [~, reliability2] = ridgeorient(grad, 1, 3, 3); % Here was I
  % Remove all the regions when disorder of orientations is bigger
  reliability2(reliability2<0.5)=0;
  tl = J.*reliability2; %Enhance the bw image removing disordered regions
  
  fprintf('Orientations done \n');
  if(dispim)
    plotridgeorient(orientim, 20,I, 2);
    show(reliability2);
  end;
  if(strokeWidth>0)
    fprintf('Starting SWT with strokeWidth of %d. \n',strokeWidth);
    tos = SWTtotal(I,tl,orientim,strokeWidth);
    fprintf('SWT done \n');
    if(dispim)
      show(tos)
     end;
    fprintf('Removing ill components \n');
    fswt = cleanswt(tos,tl); 
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
      
        if strokeWidth < 1
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
        