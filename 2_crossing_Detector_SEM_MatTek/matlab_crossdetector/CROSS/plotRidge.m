function plotRidge(im)

blksze = 50; thresh = 0.1;
[normim, mask] = ridgesegment(im, blksze, thresh);
[orientim, reliability] = ridgeorient(normim, 1, 5, 5);
plotridgeorient(orientim, 20,im , 2)