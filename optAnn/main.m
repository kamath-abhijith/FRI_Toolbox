clear
clc
close all

%% Support 
dt = 1e-4; t = 0:dt:1; t=t(:);
T0 = max(t); w0 = 2*pi/T0;
Nt = length(t); n = 1:Nt;

tLPF = -5:dt:5;

%% Define FRI signal
% Set weights and delays
al = [0.5; -0.45; 0.3]; taul = [0.21; 0.33; 0.72];
taulIdx = floor(Nt*taul/T0); L = length(al);

% Construct the signal
x = zeros(Nt,1);
x(taulIdx) = al;

%% Sampling
% Sampling kernel
T0SMS = T0; w0SMS = 2*pi/T0SMS; KSMS = L;
smsKernel = genSmsKernel(tLPF,KSMS,T0SMS,0);
y = conv(x,smsKernel,'same');

% Random sampling
Ntn = 14;
tnIdx = sort(floor(Nt*rand(Ntn,1)));
% tnIdx = floor(linspace(1,Nt,Ntn));
tn = t(tnIdx); ytn = y(tnIdx);

%% Parameter estimation
% Recovery of Fourier samples
K = L;
F = exp(1j*w0*tn*(-K:K));

G = F;

% Estimation of delays
[swce,h] = optAnn(G,ytn,L);
estTaul = angle(roots(h));
estTaul(estTaul>0) = estTaul(estTaul>0)-2*pi;
estTaul = sort(-T0*estTaul/(2*pi));

% Estimation of amplitudes
E = exp(-1j*2*pi*(-K:K)'*estTaul'/T0);
estAl = (2*K+1)*real(pinv(E)*swce);

%% Reconstruction
xRec = zeros(Nt,1);
estTaulIdx = ceil(Nt*estTaul/T0);
xRec(estTaulIdx) = estAl;

%% Error Metric
errorDelay = (norm(taul - estTaul)/norm(taul))^2;
errorAmp = (norm(al - estAl)/norm(al))^2;

errorSignal = (x-xRec);
normError = norm(errorSignal)^2/Nt;

%% Plots
figure(1)
subplot(1,2,1)
plot(t,y,'-b',"LineWidth",4)
grid on, hold on
stem(tn,ytn,'-or',"LineWidth",4)
% axis([0 T0 -ub ub])
xlabel('$t$','Interpreter','latex')
ylabel('$y(t)$','Interpreter','latex')
title('Sampling','Interpreter','latex')

subplot(1,2,2)
plot(t,x,'-b',"LineWidth",4)
grid on, hold on
plot(t,xRec,'-r',"LineWidth",4)
% axis([0 T0 -ub ub])
xlabel('$t$','Interpreter','latex')
ylabel('$x(t)$','Interpreter','latex')
title('Reconstruction','Interpreter','latex')