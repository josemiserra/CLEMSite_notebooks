function saveCoordinatesXML(points,mappoints,lnames,fname)

docNode = com.mathworks.xml.XMLUtils.createDocument('GridPointList');
ipl = docNode.getDocumentElement;
ipl.setAttribute('version','1.0');
[rows cols]= size(points);
for k=1:rows
    curr_node = docNode.createElement('GridPoint');
    curr_node.setAttribute('FieldXCoordinate',num2str(points(k,1))); 
    curr_node.setAttribute('FieldYCoordinate',num2str(points(k,2))); 
    curr_node.setAttribute('FieldZCoordinate','0.0'); 
    curr_node.appendChild(docNode.createTextNode(''));
    ipl.appendChild(curr_node);
    curr_node2 = docNode.createElement('GridPointRef');
    cletter = lnames(k,:);
    if(isempty(mappoints))
         curr_node2.setAttribute('FieldXCoordinateRef','0.0'); 
         curr_node2.setAttribute('FieldYCoordinateRef','0.0'); 
         curr_node2.setAttribute('FieldZCoordinateRef','0.0');
         curr_node2.setAttribute('Map',num2str(points(k,3))); 
    else
        curr_node2.setAttribute('FieldXCoordinateRef', num2str(mappoints(k,1))); 
        curr_node2.setAttribute('FieldYCoordinateRef', num2str(mappoints(k,2))); 
        curr_node2.setAttribute('FieldZCoordinateRef','0.0');
        curr_node2.setAttribute('Map',cletter);
    end;
    curr_node2.appendChild(docNode.createTextNode(''));
    ipl.appendChild(curr_node2);
end
xmlwrite(fname,docNode);
type(fname);