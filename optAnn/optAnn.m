function [b,c] = optAnn(G,a,K,varargin)
% One-shot optimiser for data fidelity with
% annihilation constraint. The function outputs
% the solution with the annihilation filter.
% 
% Solves minimise |a-Gb|^2
%        subject to b*c = 0
%
% INPUT:  Forward transform matrix, G
%         Measurement vector, a
%         Annihilating filter order, K
%
% OUTPUT: Solution vector, b
%         Annihilating filter, c
%
% Author: Abijith J Kamath
% kamath-abhijith@github.io
%
% For more information, see: Pan et al.,
% ieeexplore.ieee.org/abstract/document/7736135
    
    % varargin
    if nargin>=4
        NOISEERROR = varargin{1};
    else
        NOISEERROR = 1e-6;
    end
    if nargin>=5
        MAXITER = varargin{2};
    else
        MAXITER = 50;
    end
    
    % Preprocessing
    [~,n] = size(G);
    
    GtG = G'*G;
    Gta = G'*a;
    beta = (GtG)\Gta;
    
    % Random initialisation and setup
    c0 = (1+1j)*randn(K+1,1);
    c = c0;
    
    Tbeta = Tmtx(beta,K+1);
    R = Rmtx(c,K,n);
    
    rhsCmtx = [zeros(2*n+1,1); 1];
    rhsBmtx = [Gta; zeros(n-K,1)];
    
    row1_minCmtx = [zeros(K+1,K+1) Tbeta' zeros(K+1,n) c0];
    rowL_minCmtx = [c0' zeros(1,2*n-K+1)];
    
    % alt-min iterations
    for iIter = 1:MAXITER
       
       % Update annihilating filter
       minCmtx = [row1_minCmtx;...
                  Tbeta, zeros(n-K,n-K), -R, zeros(n-K,1);...
                  zeros(n,K+1), -R', GtG, zeros(n,1);...
                  rowL_minCmtx];
       c = minCmtx\rhsCmtx;
       c = c(1:K+1);
       
       % Update solution
       R = Rmtx(c,K,n);
       
       minBmtx = [GtG, R';...
                  R, zeros(n-K,n-K)];
       b = minBmtx\rhsBmtx;
       b = b(1:n);
       
       % Check stopping condition
       error = norm(a-G*b)^2;
       if error < NOISEERROR
           break;
       end       
    end
end

function R = Rmtx(data,K,seqLen)
% Constructs the dual convolution matrix
% INPUT:  Data vector, data
%         Data length, K
%         Matrix dimension for zero padding, seqLen
% OUTPUT: Dual conv matrix, R

    col = [data(end); zeros(seqLen-K-1,1)];
    row = [flip(data); zeros(seqLen-K-1,1)];
    R = toeplitz(col,row);
end

function T = Tmtx(data,K)
% Construct the Toeplitz convolution
% matrix of size K
%
% INPUT:  Data vector, data
%         Order, K
% OUTPUT: Convolution matrix, T
    
    col = data(K:end);
    row = data(K:-1:1);
    T = toeplitz(col,row);
end