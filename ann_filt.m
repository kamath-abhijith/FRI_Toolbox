function [freq_est]=ann_filt(y,K)
% Annhilating filter  method,  based on the paper:
%  T. Blu, P. L. Dragotti, M. Vetterli, P. Marziliano, and L. Coulot, 
% "Sparse sampling of signal innovations", IEEE Signal Process. Mag., vol. 25,
% no. 2, pp. 3140, Mar. 2008.
%
% written by Kfir Gedalyahu (kfir.gedal@gmail.com)

% y - samples
% K - number of exponentials

% wk - estimated frequencies
% xk - estimated amplitudes
y = y(:);
N=length(y);
M=floor(N/2);

if N<2*K || ~mod(N,2)
    error('Number of samples must be greater than 2K and odd');
end

%% total least squares 
A=zeros(2*M-K+1,K+1);

for i=0:2*M-K
    A(i+1,:)=y((-M+K:-1:-M)+ceil(N/2)+i);
end

[~,~,V]=svd(A,0);    

h=V(:,K+1);
freq_est = angle(roots(h));
freq_est(freq_est>0) = freq_est(freq_est>0)-2*pi;
end
