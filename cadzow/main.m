clear
clc
close all

%% Support
t = 0:0.01:1;

%% Noisy sequence
x = sin(2*pi*1.5*t);
xn = awgn(x,30);

%% Cadzow denoising
x_clean = cad_den(xn',2);

%% Plots
figure, subplot(1,2,1)
plot(t,x,'-g',"LineWidth",2)
grid on, hold on
axis([0 max(t) -1.2*max(-min(x),max(x)) 1.2*max(-min(x),max(x))])
stem(t,xn,'-b',"LineWidth",2)
title('Original and Noisy')

subplot(1,2,2)
stem(t,x,'-g',"LineWidth",2)
grid on, hold on
axis([0 max(t) -1.2*max(-min(x),max(x)) 1.2*max(-min(x),max(x))])
stem(t,x_clean,'-r',"LineWidth",2)
title('Original and Cleaned')