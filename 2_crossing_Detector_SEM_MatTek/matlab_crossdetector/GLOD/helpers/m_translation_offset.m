%
%	Determines the offset in translation between a misaligned and a target
%	image using the maximum cross correlation peak
%
%	Maurits Diephuis
%
%
%
function [y_offset, x_offset] = m_translation_offset(template, im)


	%	Check types. t and a must be MxN (2D) arrays with real numbers.
%	iptcheckinput(template,  {'logical', 'numeric'}, {'real', 'nonsparse', '2d', 'finite'}, 'm_translation_offset', 'template', 1);
%	iptcheckinput(im,  {'logical', 'numeric'}, {'real', 'nonsparse', '2d', 'finite'}, 'm_translation_offset', 'im', 2);
	
	%	Determin 2D cross correlation
	c = m_xcorr2(template, im);

	%	Find peak location
	[max_c, imax] 	= max(abs(c(:)));
	[ypeak, xpeak] 	= ind2sub(size(c), imax(1));
  
  	assert(ypeak > 0, 'ASSERT failed: ypeak > 0');
  	assert(xpeak > 0, 'ASSERT failed: xpeak > 0');
	
	%	Correct found peak location for image size
	corr_offset = round([(ypeak-(size(c, 1)+1)/2) (xpeak-(size(c, 2)+1)/2)]);
	
	%	Write out offsets
	y_offset = corr_offset(1);
	x_offset = corr_offset(2);
