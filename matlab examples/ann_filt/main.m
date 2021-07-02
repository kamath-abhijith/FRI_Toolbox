clear
clc
close all

%% Support 
dt = 1e-3; t = 0:dt:1; T0 = max(t);
Nt = length(t); n = 1:Nt;

%% Define FRI signal
% Set weights and delays
al = [1; -2; -1; 3];
taul = [0.1; 0.3; 0.6; 0.7];
taul_idx = floor(Nt*taul/T0);
L = length(al);

% Construct the signal
x = zeros(Nt,1);
x(taul_idx) = al;

%% Sampling
% Sampling kernel
tlpf = -1:dt:1;
lpf = sinc(100*tlpf);
y = conv(x,lpf,'same');

% Uniform sampling
nn = 1:25:Nt;
yn = y(nn);
Nk = floor(length(yn)/2);

%% Parameter estimation
% Estimation of delays
haty = fft(yn);
wk = ann_filt(haty,L);
est_taul = -T0*wk/(2*pi);

% Estimation of amplitudes
E = exp(1j*(-Nk:Nk)'*wk');
est_al = T0*pinv(E)*haty;
est_al = sign(angle(est_al)).*abs(est_al);

%% Reconstruction
xrec = zeros(Nt,1);
est_taul_idx = floor(Nt*est_taul/T0);
xrec(est_taul_idx) = est_al;

%% Plots
figure, subplot(2,1,1)
plot(t,y,'-g',"LineWidth",2)
grid on, hold on
stem(t(nn),yn,'-b')
axis([0 T0 -max(-min(x),max(x))-0.5 max(-min(x),max(x))+0.5])
xlabel('$t$','Interpreter','latex')
ylabel('$x(t)$','Interpreter','latex')
title('Sampling','Interpreter','latex')

subplot(2,1,2)
stem(t,x,'-g',"LineWidth",2)
grid on, hold on
stem(t,xrec,'-b',"LineWidth",2)
axis([0 T0 -max(-min(x),max(x))-0.5 max(-min(x),max(x))+0.5])
xlabel('$t$','Interpreter','latex')
ylabel('$x(t)$','Interpreter','latex')
title('Reconstruction','Interpreter','latex')