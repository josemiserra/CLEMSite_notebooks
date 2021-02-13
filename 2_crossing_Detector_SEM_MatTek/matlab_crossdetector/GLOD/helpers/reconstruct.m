function RSWT = reconstruct(FSWT)
    %% Find border points
    %  Check for all points that are added from the edges (have specific
    %  value. Once a pixel is found is examined in the 8 directions.
    