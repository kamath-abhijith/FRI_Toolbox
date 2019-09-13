%%
% Cadzow Denoising
%
% Author: Abijith J Kamath
% Institute: NITK, Spectrum Lab IISc.
% Date: May 2018
% 
% Purpose: For denoising in shape fitting using FRI principle
%
% Averaging the low rank approximation of a particular sequence
%
%%

function x_out = cad_den(x_in, in_rank)
% Cadzow denoising
%  
% Iteratiely denoises a sequences assuming spectral
% separation in the signal content and noise.
%
% INPUT:  Noisy sequence, x_in
%         Assumed rank of signal, in_rank
% OUTPUT: Denoised output, x_out
%
% Written by: Abijith J Kamath
% kamath-abhijith@github.io

% Create the Hankel structure
N = length(x_in);
L = floor(N/2);
cx = x_in(L+1:N);
rx = fliplr(x_in(1:L+1));
XT = toeplitz(cx, rx);

% Global constraints
iterLim = 100;
thresh = .001;

% Cadzow iterations
for iter = 1:iterLim
  % Compute the low rank form of XT
  [U,S,V] = svd(XT,'econ');
  s = diag(S);
  
  % Break condition
  if(real(s(in_rank+1))<thresh*real(s(in_rank)) && iter>iterLim)
    break
  end
  
  % Reduce rank to in_rank and recompute XT
  s(in_rank+1:end) = 0;
  S_lr = diag(s);
  XT_lr = U*S_lr*V'; % Not toeplitz structure
  
  % Average the low rank form to toeplitz
  for j=1:L+1
    r_tp(j)=mean(diag(XT_lr,j-1));
  end

  for i=1:(N-L)
    c_tp(i)=mean(diag(XT_lr,-i+1));
  end
  XT_tp = toeplitz(c_tp, r_tp);
  
  % Setup for next iteration
  XT = XT_tp;
end

% Rearrange output
out1 = XT(1,:);
out2 = XT(:,1).';

x_out = [out1(end:-1:2) out2];

end