
%	C = M_XCORR2(TEMPLATE, A) computes the cross-correlation of
%   matrices TEMPLATE and A. 
%
%	M_XCORR2 replaces slow matlab XCORR2 that doens't correlate using the
%	fourier domain
%
%
%	Maurits Diephuis, Fokko Beekhof
%
%
%	IN:
%		t:	2D NxM numeric array
%		b:	2D NxM numeric array
%
%
%
%	OUT:
%		c: NxM array with correlation coeficients
%		
%	Version 0.2
%	29-01-2010
%	23-08-2010
%
%
function c = m_xcorr2(varargin)

	t = varargin{1};
%	iptcheckinput(t,  {'logical', 'numeric'}, {'real', 'nonsparse', '2d', 'finite'}, 'm_xcorr2', 't', 1);

	%	Determine padding size in x and y dimension
	size_t 		= size(t);
	outsize 	= 2*size_t - 1;

	%	If two images were passed
	if ( length(varargin) == 2 )
		a = varargin{2};
%	  iptcheckinput(a,  {'logical', 'numeric'}, {'real', 'nonsparse', '2d', 'finite'}, 'm_xcorr2', 'a', 2);
	  
	  %	Determine padding size in x and y dimension
	  size_a 		= size(a);
	  outsize 	= size_t + size_a - 1;
	end

	%	FFT first argument
	Ft = fft2(t, outsize(1), outsize(2));


	%	Fourier transform
	if ( length(varargin) == 2 )
		Fa = fft2(a, outsize(1), outsize(2));
	  c = abs( fftshift( ifft2(Fa .* conj(Ft))) );
	else
	  c = abs( fftshift( ifft2(Ft .* conj(Ft))) );
	end