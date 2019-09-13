clear
clc
close all

%% Support
n = 1:49; N = length(n);
dt = 0.1; t = 0:dt:49; Nt = length(t);

%% Weights
spden = 0.1;
x = full(sprandn(1,N,spden));

%% Sinc interpolation // Sampling kernel
[Ts,T] = ndgrid(t,n);
y = sinc(Ts - T)*x';

%% Sampling
nn = 1:4:Nt;
yn = y(nn);

%% HRSE from SWCE form
haty = fft(yn);
L = ceil(spden*N);
wk = ann_filt(haty,ceil(L));

tk = N*wk./(2*pi);

%% Plots
figure
stem(n,x,'-g',"LineWidth",1)
hold on, grid on
plot(t,y,'-b',"LineWidth",2)
stem(t(nn),yn,'-r',"LineWidth",2)
axis([0 max(t) -1.2*max(-min(x),max(x)) 1.2*max(-min(x),max(x))])