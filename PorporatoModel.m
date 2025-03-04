%% Setup: initialize parameters
T_total = 243; % number of days to simulate
n = 0.4; % porosity
z = 800; % soil depth [mm]
s_h = 0.02; % hygroscopic point, no evap below
s_w = 0.065; % wilting point, max evap above, no transp below
s_sc = 0.17; % stomato closing point, max transp above
s_fc = 0.3; % field capacity, leakage begins above here
E_max = 1.0; % maximum evap [mm/day] % ET max in Porporato is 4.5 = sum Emax and Tmax (use 1 and 3.5 for E and T)
T_max = 3.5; % maximum transp [mm/day]
K_s = 1100; % saturated hydrualic conductivity [mm/day]
b = 4.05; % pore size distribution index
beta = 2*b + 4;
add_lit = 1.5; % carbon added to litter pool daily [gC/m^2/day] % Porporato had 1.5
k_d = 0.0085; % biomass death rate [1/day] % Porporato had 0.0085
k_l = 0.000065; % constant for rate of decomposition in litter [m^3/day/gC]
k_h = 0.0000025; % constant for rate of decomposition in humus [m^3/day/gN] % Porporato had 0.0000025
k_amm = 1; % constants for partitioning immobilized N [m^3/day/gC]
k_nit = 1; % constants for partitioning immobilized N [m^3/day/gC]
k_n = 0.6; % rate of nitrification [m^3/day/gC] % Porporato had 0.6
a_amm = 0.05; % scale factor for ammonium leaching
a_nit = 1; % scale factor for nitrate leaching
r_r = 0.6; % fraction of decomposed carbon that goes to respiration [-]
CN_add = 58;  % C:N ratio in added litter % Porporato had 58
CN_bio = 11.5; % C:N ratio in biomass
CN_hum = 22; % C:N ratio in humus
dem_amm = 0.2; % plant demand for ammonium [gN/m^3/day]
dem_nit = 0.5; % plant demand for nitrate [gN/m^3/day]
F = 0.1; % diffusion coefficient [m/day]
d = 3; % diffusion exponent for soil moisture dependence

s = zeros(1, T_total); % relative soil moisture [-]
s(1) = 0.15;
c_lit = zeros(1, T_total); % carbon in litter pool [g/m^2]
c_lit(1) = 1300;
c_bio = zeros(1, T_total); % carbon in biomass pool
c_bio(1) = 50;
c_hum = zeros(1, T_total); % carbon in humus pool
c_hum(1) = 8500;
n_lit = zeros(1, T_total); % nitrogen in litter pool
n_lit(1) = c_lit(1)/34;
% n_bio = zeros(1, T_total); % nitrogen in biomass pool
% n_bio(1) = c_bio(1)/11.5;
n_amm = zeros(1, T_total); % nitrogen in litter pool
n_amm(1) = 0.2; % 0.002 for Porporato, but maybe ~4 for IML?
n_nit = zeros(1, T_total); % nitrogen in litter pool
n_nit(1) = 1.00; % 0.3 for Porporato, but maybe 20-25 for IML?

decomp = zeros(1, T_total); %
mineralize = zeros(1, T_total); %
psi_time = zeros(1, T_total); %
phi_time = zeros(1, T_total); %
rh_time = zeros(1, T_total); %

%% Rainfall
lambda = 0.23; % rate of days with rain % 0.23 for Porporato
alpha = 11; % mean rainfall depth on days with rain [mm] % 11 for Porporato
Event = zeros(1,T_total);
Pois = rand(1,T_total);
Event(Pois<lambda) = 1;
PDepth = exprnd(alpha,1,T_total);
Precip = Event .* PDepth ;

%% Loop
for t=2:T_total
    % Soil moisture dynamics
    Avail_s = n*z*(1-s(t-1));
    Infil = min(Precip(t), Avail_s);
    if s(t-1)<s_h
        Evap = 0;
        Transp = 0;
        Leak = 0;
    elseif s(t-1)<s_w
        Evap = E_max *(s(t-1)-s_h)/(s_w-s_h);
        Transp = 0;
        Leak = 0;
    elseif s(t-1)<s_sc
        Evap = E_max;
        Transp = T_max *(s(t-1)-s_w)/(s_sc-s_w);
        Leak = 0;
    elseif s(t-1)<s_fc
        Evap = E_max;
        Transp = T_max;
        Leak = 0;
    else
        Evap = E_max;
        Transp = T_max;
        Leak = K_s *(exp(beta*(s(t-1)-s_fc))-1)/(exp(beta*(1-s_fc))-1);
    end
    % Soil moisture effects on decomposition (d) and nitrification (n)
    if s(t-1)<=s_fc
        fact_d = s(t-1)/s_fc;
        fact_n = s(t-1)/s_fc;
    else
        fact_d = s_fc/s(t-1);
        fact_n = (1-s(t-1))/(1-s_fc);
    end
    % Update CN ratio for litter and r_h parameter
    CN_lit = c_lit(t-1)/n_lit(t-1);
    r_bound = CN_hum/CN_lit;
    r_h = min (0.25, r_bound);
    % Phi, psi - mineralization/immobilization
%     imm_max = (k_amm*n_amm(t-1) + k_nit*n_nit(t-1))*fact_d; % Porporato also has *c_bio(t-1), but that breaks mass balance
    %%%%%%%%%%%%% try new imm_max formulation %%%%%%%%%%%%%%%%%%%%
    imm_max = (k_amm*n_amm(t-1) + k_nit*n_nit(t-1))*fact_d; % Porporato also has *c_bio(t-1), but that breaks mass balance
    phi_bracket = k_h*c_hum(t-1)*(1/CN_hum - (1-r_r)/CN_bio) + k_l*c_lit(t-1) * ...
        (1/CN_lit - r_h/CN_hum - (1-r_h-r_r)/CN_bio);
    if phi_bracket >= 0
        fact_psi = 1;
        miner = fact_psi * fact_d * c_bio(t-1) * phi_bracket;
        imm = 0;
    else
        imm_temp = fact_d * c_bio(t-1) * phi_bracket;
        if imm_temp >= -imm_max
            fact_psi = 1;
        else
%             fact_psi = -(k_amm*n_amm(t-1)+k_nit*n_nit(t-1)) / ...
%                 (c_bio(t-1)*(k_h*c_hum(t-1)*(1/CN_hum - (1-r_r)/CN_bio) + ...
%                 k_l*c_lit(t-1)*(1/CN_lit - r_r/CN_hum - (1-r_h-r_r)/CN_bio))); % add c_bio(t-1) to denom b/c removed from imm_max, see above
            %%%%% try new fact_psi formulation %%%%%
            fact_psi = -imm_max/imm_temp;
        end
        miner = 0;
        imm = -(fact_psi * fact_d * c_bio(t-1) * phi_bracket);
    end
    imm_amm = k_amm*n_amm(t-1)/(k_amm*n_amm(t-1)+k_nit*n_nit(t-1)) * imm;
    imm_nit = k_nit*n_nit(t-1)/(k_amm*n_amm(t-1)+k_nit*n_nit(t-1)) * imm;
    % Carbon decomposition, litter (also for N decomp)
    bd = k_d * c_bio(t-1); % biomass death
    dec_lit = (fact_psi*fact_d*k_l*c_bio(t-1))*c_lit(t-1); % decomposition in litter
    % Carbon decomposition, humus  (also covers redundant N decomp, C:N is constant in humus)
    dec_hum = (fact_psi*fact_d*k_h*c_bio(t-1))*c_hum(t-1);
    % Carbon and nitrogen decomp, biomass (all necessary rates/factors calculated above)
    % Nitrification
    nit = fact_n * k_n * c_bio(t-1) * n_amm(t-1);
    if nit>n_amm(t-1)
        nit = n_amm(t-1);
    end
    % Leaching
    leach_amm = (Leak/(s(t-1)*n*z))*a_amm*n_amm(t-1);
    leach_nit = (Leak/(s(t-1)*n*z))*a_nit*n_nit(t-1);
    % Plant uptake
    k_u_amm = a_amm /(s(t-1)*n*z) * F * s(t-1)^d;
    k_u_nit = a_nit /(s(t-1)*n*z) * F * s(t-1)^d;
    up_pass_amm = (Transp/(s(t-1)*n*z))*a_amm*n_amm(t-1);
    up_pass_nit = (Transp/(s(t-1)*n*z))*a_nit*n_nit(t-1);
    if up_pass_amm > dem_amm
        up_act_amm = 0;
    elseif up_pass_amm > (dem_amm - k_u_amm*n_amm(t-1))
        up_act_amm = dem_amm - up_pass_amm;
    else
        up_act_amm = k_u_amm*n_amm(t-1);
    end
    if up_pass_nit > dem_nit
        up_act_nit = 0;
    elseif up_pass_nit > (dem_nit - k_u_nit*n_nit(t-1))
        up_act_nit = dem_nit - up_pass_nit;
    else
        up_act_nit = k_u_nit*n_nit(t-1);
    end
    up_amm = up_pass_amm + up_act_amm;
    up_nit = up_pass_nit + up_act_nit;
    % Update states
    s(t) = s(t-1) + (1/(n*z))*(Infil - Evap - Transp - Leak);
    if s(t)<0
       Leak = Leak + s(t);
       s(t) = 0;
    end
    c_lit(t) = c_lit(t-1) + add_lit + bd - dec_lit;
    n_lit(t) = n_lit(t-1) + add_lit/CN_add + bd/CN_bio - dec_lit/CN_lit;
    c_hum(t) = c_hum(t-1) + r_h*dec_lit - dec_hum;
    c_bio(t) = c_bio(t-1) + (1-r_h-r_r)*dec_lit + (1-r_r)*dec_hum - bd ;
%     n_bio(t) = n_bio(t-1) + (1-r_h*(CN_lit/CN_hum))*(dec_lit/CN_lit) + ...
%         dec_hum/CN_hum - bd/CN_bio - phi;
    n_amm(t) = n_amm(t-1) + miner - imm_amm - nit - leach_amm - up_amm;
    n_nit(t) = n_nit(t-1) + nit - imm_nit - leach_nit - up_nit;
    if n_amm(t)<0
        n_amm(t)=0;
    end
    if n_nit(t)<0
        n_nit(t)=0;
    end
    decomp(t-1) = dec_lit;
    mineralize(t-1) = miner;
    psi_time(t-1) = fact_psi;
    phi_time(t-1) = phi_bracket;
    rh_time(t-1) = r_h;
    % N Fertilizer
    if t==80 || t==160
       n_nit(t) = n_nit(t) + 0.5;
       n_amm(t) = n_amm(t) + 0.05;
    end
end


%% Plot
% figure
% plot(s)
% hold on
% ylabel('Relative soil moisture [-]', 'fontsize', 26)
% yyaxis right
% bar(Precip)
% hold off
% xlabel('Day of simulation', 'fontsize', 26)
% ylabel('Precipitation [mm]', 'fontsize', 26)
% figure
% plot(c_lit)
% figure
% plot(c_hum)
% figure
% plot(c_bio)
% figure
% plot(n_amm)
% figure
% hold on
% plot(n_nit)
% xlabel('Day of simulation', 'fontsize', 26)
% ylabel('Nitrate-N concentration [gN/m^3]', 'fontsize', 26)
% figure
% plot(n_lit)
% figure
% plot(decomp)
figure
plot(mineralize)
% CN_lit = c_lit ./ n_lit;
% figure
% plot(CN_lit)
% figure
% plot(psi_time)
% figure
% plot(phi_time)
% figure
% plot(rh_time)
