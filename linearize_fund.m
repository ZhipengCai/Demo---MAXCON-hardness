
function [A, B] = linearize_fund(x1,x2)
    % Build the constraint matrix
 
    npts = size(x1, 2); 

    
    XX = [x2(1,:)'.*x1(1,:)'   x2(1,:)'.*x1(2,:)'  x2(1,:)' ...
         x2(2,:)'.*x1(1,:)'   x2(2,:)'.*x1(2,:)'  x2(2,:)' ...
         x1(1,:)'             x1(2,:)'            ones(npts,1) ];       

    A = XX; 
    B = - (XX(:, 9));
    A(:, 9) = []; 

end

