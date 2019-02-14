%test for the number of unique basis in MAXCON
clear;
close all;

N = 200;    % Number of Points
NOInst = 50; %Number of problem instances tested for each fixed dimension and NO. outliers

%minimum and maximum number of the parameters, the combinatorial
%dimension (size |B| of a basis) should be d+1
dimMin = 1;
dimMax = 10;

%minimum and maximum number of outliers
outMin = 1;
outMax = 20;

%Unique node number for A*
max_UNN_ASTAR = zeros(dimMax-dimMin+1, outMax-outMin+1); 
mean_UNN_ASTAR = zeros(dimMax-dimMin+1, outMax-outMin+1); 
dev_UNN_ASTAR = zeros(dimMax-dimMin+1, outMax-outMin+1); 


bound = zeros(dimMax-dimMin+1, outMax-outMin+1); %matousek's bound
bound2 = zeros(dimMax-dimMin+1, outMax-outMin+1); %naive bound for tree search

sig = 0.02; % Inlier Varience
osig = 1;   % Outlier Varience
th = 0.02;  % Inlier Threshold


for d = dimMin:dimMax
    for o = outMin:outMax
        idxD = d-dimMin+1;
        idxO = o-outMin+1;
        bound(idxD,idxO) = (1+1/o)^o*(o+1)^(d+1); %matousek's bound
        level = 0:o;
        bound2(idxD,idxO) = sum((d+1).^level); %naive bound for BFS tree search
        
        UNNASTAR{idxD,idxO} = zeros(NOInst,1);
        
        disp(['starting new iteration with current d = ' num2str(d) '; current o = ' num2str(o)]);        
        
        for iter = 1:NOInst
            %GT model
            m = rand(d-1, 1);
            c = randn;
            
            % Generate data
            xo = randn(d-1,N);
            yo = m'*xo + repmat(c,1,N);
            
            % Corrupt data (uniform distribution)
            %x = xo + sig*(rand(d-1,N)-0.5)*2;
            x = xo;
            y = yo + sig*(rand(1,N)-0.5)*2;
            
            
            % Add outliers (Uniform distribution).
            sn = sign(y(1:o) - m'*xo(:, 1:o)-c);
            y(1:o) = yo(1:o)+(sn<0).*sn.*(osig*rand(1, o)+sig);
            y(1:o) = y(1:o)+(sn>0).*sn.*(osig*rand(1, o)+sig);
            
            x = x';
            y = y';
            
            
            %index of outliers
            outIdx = 1:o;
            
            
            %% Rewrite the equation of lines
            x = [x, ones(N, 1)];
            
            x0 = rand(d, 1);

            
            [P5_ASTAR, val5_ASTAR, v5_ASTAR, mcnum5_ASTAR,mxnum5_ASTAR,~,UNNASTAR{idxD,idxO}(iter)] = maxconASTAR(x, y, x0, th);
            
            
        end
        
        
        %results for ASTAR
        mean_UNN_ASTAR(idxD,idxO) = mean(UNNASTAR{idxD,idxO});
        dev_UNN_ASTAR(idxD, idxO) = std(UNNASTAR{idxD,idxO});
        max_UNN_ASTAR(idxD,idxO) = max(UNNASTAR{idxD,idxO});
        
        
        disp(['maximum unique node number for A* = ' num2str(mean_UNN_ASTAR(idxD,idxO))]);
        disp(['Matousek bound = ' num2str(bound(idxD,idxO))]);
    end
    
end

figure;
hold on;
[dd, oo] = meshgrid(dimMin:dimMax, outMin:outMax);
colormap([1 0 0; 0 0 1; 0 1 0; 1 1 0]);
%bound for NO. bases
surf(dd, oo, bound', ones(outMax-outMin+1, dimMax-dimMin+1));
surf(dd, oo, bound2', ones(outMax-outMin+1, dimMax-dimMin+1)+1);
surf(dd, oo, max_UNN_ASTAR', ones(outMax-outMin+1, dimMax-dimMin+1)+2);
surf(dd, oo, mean_UNN_ASTAR', ones(outMax-outMin+1, dimMax-dimMin+1)+3);

set(gca, 'ZScale', 'log');

save('numberOfUniqueBases.mat');