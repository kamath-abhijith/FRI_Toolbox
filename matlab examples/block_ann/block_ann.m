function [freq_est] = block_ann(x, y, K)
% High-Resolution Spectral Estimation (HRSE) from two
% sequences in Sum of Weighted Complex Exponential (SWCE) form
% with identical frequencies. Weights may differ.
% 
% Frequencies are estimated using the roots of an
% annihilating filter. Solution to the null space is
% computed using Eckart-Young Theorem. Since the frequencies are
% identical, the annihilating filter is common.
%
% INPUT:  Sequences in SWCE form, x,y
%         Model Order, K
% OUTPUT: Estimated frequencies, freq_est
%
% Modified for FRI-FD: Abijith J Kamath
% kamath-abhijith@github.io
%
% For more information, check: 
% Compressive sampling of multiple sparse signals having common support
% using finite-rate-of-innovation principle.

% Preprocessing
y = y(:); x = x(:);
N=length(y);
M=floor(N/2);

% Error checks
if ~mod(N,2)
    error('Number of samples must be greater than 2K and odd');
end

% Construct the individual convolution matrices
A=zeros(2*M-K+1,K+1);
B=zeros(2*M-K+1,K+1);
for i=0:2*M-K
    A(i+1,:)=y((-M+K:-1:-M)+ceil(N/2)+i);
    B(i+1,:)=x((-M+K:-1:-M)+ceil(N/2)+i);
end

% Solution to the null space
[~,~,V]=svd([A;B],0);    
h=V(:,K+1);

% Estimating frequencies using roots of the filter
freq_est = angle(roots(h));
freq_est(freq_est>0) = freq_est(freq_est>0)-2*pi;
end
