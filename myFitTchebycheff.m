%##########################################################################
%% G L O B A L L Y   O P T I M A L   C O N S E N S U S   M A X I M I S A T I O N
%% This package contains the source code which implements optimal Consensus 
% Maximisation proposed in
% T.J. Chin, P. Purkait, A. Eriksson and D. Suter
% Efficient Globally Optimal Consensus Maximisation with Tree Search, 
% In Proceedings of the IEEE Conference on Computer Vision and Pattern 
% Recognition (CVPR), June 2015, Boston
% 
% Copyright (c) 2015 Pulak Purkait (pulak.purkait@adelaide.edu.au.)
% School of Computer Science, The University of Adelaide, Australia
% The Australian Center for Visual Technologies
% http://www.cs.adelaide.edu.au/directory/pulak.purkait
%% Please acknowledge the authors by citing the above paper in any academic 
%  publications that have made use of this package or part of it.
%##########################################################################

% Function to optimize minimax problem using Ascending optimization 
% -----------------------------------------------------------------

function [xn,d,M] = myFitTchebycheff(X, Y, xn)

if nargin < 2
    Y = X(end, :)'; 
    X = X(1:end-1, :)';
end

if nargin < 3
    xn = rand(size(X, 2), 1); 
end

%replace each constraint with an absolute value by two linear constraints
X = [X; -X]; 
Y = [Y; -Y]; 
%number of parameters
n = size(X, 2); 

flag = 0; %for some initialization, there might be too many active constraints, if that happens, we re-initialize the problem until this is not the case

while flag==0
    r = X*xn - Y;
    
    %maximum value of the linear residuals
    d = max(r, [], 1);
    %find the indices (of the linear constraints) with current maximum
    %residuals
    res = abs(r-d);
    
    M = find(res<1e-8);
    
    %set the residual of those points to the maximum value
    r(M) = d;
    
    % if size(ones(n+1, 1),1) ~= size(X(M', :),1)
    %     disp('already degenerate before while loop');
    %     n+1
    %     size(X(M', :))
    % end
    
    %the number of the points with maximum residuals is less than the size of
    %the basis
    while(numel(M) < n+1)
        A = X(M, :);
        c = -(A*A')\ones(numel(M), 1);
        y = A'*c;
        lambda = (d - r)./(X*y+1);
        lambda(isnan(lambda)) = 0;
        lambda(lambda<=0) = Inf;
        [lambda_j, j] = min(lambda);
        xn = xn + lambda_j*y;
        M = [M; j];
        
        r = X*xn - Y;
        d = max(r, [], 1);
        r(M) = d;
    end
    
    if n+1 == size(X(M', :),1)
        flag = 1;
    else
        xn = randn(n,1);
    end
end

C = inv([ones(n+1, 1), X(M', :)]); 
C1 = C(1, :); 
Ch = C1./abs(C1); 

itrn = 50; 
while(sum(Ch) < n) && itrn
    
    [~, p1] = min(C1); %% Error Occuring ingeneral in p1-th position
    y = C(2:end, p1)/C(1, p1); 
    t = (d - r)./(X*y+1); 
    
    t(isnan(t)) = 0; 
    
%     id = find(t > 0); 
%     t = t(id);  
%     [t_j, j] = min(t); 
%     j = id(j); 
    
    t(t<=0) = Inf; 
    [t_j, j] = min(t); 
    
    xn = xn + t_j*y; 
    
    lambda = [1, X(j, :)]*C; 
    
%     [~, beta] = max(lambda./C1); % This DOESN'T work on higher dimension
    beta = p1; % This huristic
    Cb = C(:, beta)/lambda(beta); 
    
    C = C - (ones(n+1, 1)*lambda) .* (Cb*ones(1, n+1));
    C(:, beta) = Cb; 
    
%     C(:, 1:beta-1) = C(:, 1:beta-1) - (ones(n+1, 1)*lambda(1:beta-1)) .* (Cb*ones(1, beta-1)); 
%     C(:, beta+1:end) = C(:, beta+1:end) - (ones(n+1, 1)*lambda(beta+1:end)) .* (Cb*ones(1, n+1-beta)); 
    
    M(beta) = j;          %% In some cases beta replaced by p1 solves
    
    r = X*xn - Y; 
    d = max(r);
    r(M) = d; 
    
    C1 = C(1, :); 
    Ch = C1./abs(C1); 
    itrn = itrn - 1; 
    if itrn < 1
        disp(d); 
    end
end

l = numel(Y)/2; 
id = M > l; 
M(id) = M(id) - l; 

% M = unique(M); 
