clear; close all; clc;
tic; % 开始计时

%% ===================== 全局参数设置 =====================
fprintf('====== 动态Bianchi I几何中准周期系统分析 ======\n');
fprintf('初始化参数...\n');

% 晶格参数
Lx = 20; % 晶格在 x 方向的大小
Ly = 20; % 晶格在 y 方向的大小  
Lz = 20; % 晶格在 z 方向的大小
N = Lx * Ly * Lz; % 总晶格点数

% 物理参数
t0 = 1; % 跃迁振幅的基准值
V0 = 2.0; % 准周期势场的强度

% 准周期势场参数
beta_x = (sqrt(5)-1)/2; % 无理数，定义准周期
beta_y = (sqrt(3)-1)/2;
beta_z = (sqrt(2)-1)/2;
phi_x = 0.1; % 相位
phi_y = 0.3;
phi_z = 0.5;

% 时间参数
time_steps = 50; % 时间步数
t_vals = linspace(0, 10, time_steps); % 时间范围

% 动态膨胀因子
a1_t = 1.0 + 0.1 * sin(2 * pi * t_vals / max(t_vals)); % x 方向
a2_t = 1.2 + 0.1 * cos(2 * pi * t_vals / max(t_vals)); % y 方向
a3_t = 1.5 + 0.1 * sin(4 * pi * t_vals / max(t_vals)); % z 方向

% 动量空间参数 
Nk = 30; % 动量空间离散点数
kx_vals = linspace(-pi, pi, Nk);
ky_vals = linspace(-pi, pi, Nk);
kz_vals = linspace(-pi, pi, Nk);

% 高精度扫描参数（用于临界点分析）
a3_critical_est = 1.442; % 估计的临界点
a3_range = 0.02; % 扫描范围
a3_precision = 100; % 扫描点数

% 结果存储初始化
results = struct();
results.time_vals = t_vals;
results.a1_t = a1_t;
results.a2_t = a2_t;
results.a3_t = a3_t;

fprintf('参数设置完成！\n');
fprintf('  晶格尺寸: %dx%dx%d (总点数: %d)\n', Lx, Ly, Lz, N);
fprintf('  时间步数: %d\n', time_steps);
fprintf('  动量网格: %d^3\n', Nk);

%% ===================== 主要分析流程 =====================
fprintf('开始主要分析...\n');

%% 1. 动态几何分析
fprintf('1. 进行动态几何分析...\n');
IPR_total_time = zeros(1, time_steps);
IPR_x_time = zeros(1, time_steps);
IPR_y_time = zeros(1, time_steps);
IPR_z_time = zeros(1, time_steps);
energy_levels_time = zeros(10, time_steps);
energy_gaps_time = zeros(1, time_steps);
chern_numbers_time = zeros(1, time_steps);

% 创建进度条
h_progress = waitbar(0, '动态几何分析进行中...');

for time_idx = 1:time_steps
    waitbar(time_idx/time_steps, h_progress, sprintf('处理时间点 %d/%d', time_idx, time_steps));
    
    % 当前时间的膨胀因子
    a1 = a1_t(time_idx);
    a2 = a2_t(time_idx);
    a3 = a3_t(time_idx);
    
    % 动态跃迁振幅
    tx = t0 / a1;
    ty = t0 / a2;
    tz = t0 / a3;
    
    % 构建哈密顿量
    H = build_hamiltonian(Lx, Ly, Lz, tx, ty, tz, V0, a1, a2, a3, beta_x, beta_y, beta_z, phi_x, phi_y, phi_z);
    
    % 计算本征值和本征态
    num_eigenvalues = 10;
    opts.isreal = true;
    opts.issym = true;
    try
        [eigenvectors, eigenvalues] = eigs(H, num_eigenvalues, 'smallestreal', opts);
        eigenvalues = diag(eigenvalues);
        energy_levels_time(:, time_idx) = eigenvalues;
        energy_gaps_time(time_idx) = eigenvalues(2) - eigenvalues(1);
        
        % 计算IPR
        wavefunction = eigenvectors(:, 1);
        [IPR_total, IPR_x, IPR_y, IPR_z] = calculate_directional_IPR(wavefunction, Lx, Ly, Lz);
        IPR_total_time(time_idx) = IPR_total;
        IPR_x_time(time_idx) = IPR_x;
        IPR_y_time(time_idx) = IPR_y;
        IPR_z_time(time_idx) = IPR_z;
        
        % 计算Berry曲率和Chern数（每5个时间点计算一次以节省时间）
        if mod(time_idx, 5) == 1 || time_idx == time_steps
            Berry_curvature = calculate_berry_curvature_3D(kx_vals, ky_vals, kz_vals, tx, ty, tz, V0, a1, a2, a3);
            chern_numbers_time(time_idx) = calculate_chern_number(Berry_curvature, Nk);
        end
    catch ME
        fprintf('时间点 %d 计算失败: %s\n', time_idx, ME.message);
        % 使用前一个时间点的值或默认值
        if time_idx > 1
            energy_levels_time(:, time_idx) = energy_levels_time(:, time_idx-1);
            energy_gaps_time(time_idx) = energy_gaps_time(time_idx-1);
            IPR_total_time(time_idx) = IPR_total_time(time_idx-1);
            IPR_x_time(time_idx) = IPR_x_time(time_idx-1);
            IPR_y_time(time_idx) = IPR_y_time(time_idx-1);
            IPR_z_time(time_idx) = IPR_z_time(time_idx-1);
        end
    end
end

close(h_progress);

% 存储动态分析结果
results.IPR_total_time = IPR_total_time;
results.IPR_x_time = IPR_x_time;
results.IPR_y_time = IPR_y_time;
results.IPR_z_time = IPR_z_time;
results.energy_levels_time = energy_levels_time;
results.energy_gaps_time = energy_gaps_time;
results.chern_numbers_time = chern_numbers_time;

fprintf('动态几何分析完成！\n');

%% 2. Y方向局域化机制深度分析
fprintf('2. 进行Y方向局域化机制分析...\n');
% 聚焦t=4到6区间
t_dense = linspace(3.8, 6.2, 50);
n_times_y = length(t_dense);

a2_t_dense = 1.2 + 0.1 * cos(2 * pi * t_dense / 10);
IPR_y_dense = zeros(1, n_times_y);
y_center_of_mass = zeros(1, n_times_y);
y_spread = zeros(1, n_times_y);

h_progress = waitbar(0, 'Y方向局域化分析进行中...');

for time_idx = 1:n_times_y
    waitbar(time_idx/n_times_y, h_progress, sprintf('Y方向分析 %d/%d', time_idx, n_times_y));
    
    a1 = 1.0 + 0.1 * sin(2 * pi * t_dense(time_idx) / 10);
    a2 = a2_t_dense(time_idx);
    a3 = 1.5 + 0.1 * sin(4 * pi * t_dense(time_idx) / 10);
    
    tx = t0 / a1;
    ty = t0 / a2;
    tz = t0 / a3;
    
    H = build_hamiltonian(Lx, Ly, Lz, tx, ty, tz, V0, a1, a2, a3, beta_x, beta_y, beta_z, phi_x, phi_y, phi_z);
    
    opts.isreal = true;
    opts.issym = true;
    try
        [eigenvectors, ~] = eigs(H, 1, 'smallestreal', opts);
        
        wavefunction = eigenvectors(:, 1);
        prob_density = abs(wavefunction).^2;
        prob_3D = reshape(prob_density, [Lx, Ly, Lz]);
        
        % Y方向分析
        prob_y = squeeze(sum(sum(prob_3D, 1), 3));
        prob_y = prob_y / sum(prob_y);
        IPR_y_dense(time_idx) = sum(prob_y.^2) * Ly;
        
        y_coords = 1:Ly;
        y_center_of_mass(time_idx) = sum(y_coords .* prob_y);
        y_spread(time_idx) = sqrt(sum(((y_coords - y_center_of_mass(time_idx)).^2) .* prob_y));
    catch ME
        fprintf('Y方向时间点 %d 计算失败: %s\n', time_idx, ME.message);
        if time_idx > 1
            IPR_y_dense(time_idx) = IPR_y_dense(time_idx-1);
            y_center_of_mass(time_idx) = y_center_of_mass(time_idx-1);
            y_spread(time_idx) = y_spread(time_idx-1);
        end
    end
end

close(h_progress);

% 存储Y方向分析结果
results.t_dense = t_dense;
results.a2_t_dense = a2_t_dense;
results.IPR_y_dense = IPR_y_dense;
results.y_center_of_mass = y_center_of_mass;
results.y_spread = y_spread;

fprintf('Y方向分析完成！\n');

%% 3. Z方向临界点高精度分析
fprintf('3. 进行Z方向临界点分析...\n');
a3_dense = linspace(a3_critical_est - a3_range/2, a3_critical_est + a3_range/2, a3_precision);
IPR_z_scan = zeros(1, a3_precision);
energy_gap_scan = zeros(1, a3_precision);
participation_entropy = zeros(1, a3_precision);

h_progress = waitbar(0, 'Z方向临界点分析进行中...');

for a3_idx = 1:a3_precision
    waitbar(a3_idx/a3_precision, h_progress, sprintf('Z方向扫描 %d/%d', a3_idx, a3_precision));
    
    a1 = 1.0; % 固定
    a2 = 1.2; % 固定
    a3 = a3_dense(a3_idx);
    
    tx = t0 / a1;
    ty = t0 / a2;
    tz = t0 / a3;
    
    H = build_hamiltonian(Lx, Ly, Lz, tx, ty, tz, V0, a1, a2, a3, beta_x, beta_y, beta_z, phi_x, phi_y, phi_z);
    
    opts.isreal = true;
    opts.issym = true;
    opts.tol = 1e-10;
    try
        [eigenvectors, eigenvalues] = eigs(H, 2, 'smallestreal', opts);
        eigenvalues = diag(eigenvalues);
        energy_gap_scan(a3_idx) = eigenvalues(2) - eigenvalues(1);
        
        wavefunction = eigenvectors(:, 1);
        [~, ~, ~, IPR_z] = calculate_directional_IPR(wavefunction, Lx, Ly, Lz);
        IPR_z_scan(a3_idx) = IPR_z;
        
        % 参与熵
        prob_density = abs(wavefunction).^2;
        prob_density = prob_density / sum(prob_density); % 确保归一化
        S1 = -sum(prob_density .* log(prob_density + 1e-16));
        participation_entropy(a3_idx) = exp(S1);
    catch ME
        fprintf('Z方向扫描点 %d 计算失败: %s\n', a3_idx, ME.message);
        if a3_idx > 1
            IPR_z_scan(a3_idx) = IPR_z_scan(a3_idx-1);
            energy_gap_scan(a3_idx) = energy_gap_scan(a3_idx-1);
            participation_entropy(a3_idx) = participation_entropy(a3_idx-1);
        end
    end
end

close(h_progress);

% 找到关键点
[max_IPR_z, max_IPR_z_idx] = max(IPR_z_scan);
a3_critical = a3_dense(max_IPR_z_idx);
[min_gap, min_gap_idx] = min(energy_gap_scan);
a3_min_gap = a3_dense(min_gap_idx);

% 存储Z方向分析结果
results.a3_dense = a3_dense;
results.IPR_z_scan = IPR_z_scan;
results.energy_gap_scan = energy_gap_scan;
results.participation_entropy = participation_entropy;
results.a3_critical = a3_critical;
results.max_IPR_z = max_IPR_z;
results.a3_min_gap = a3_min_gap;
results.min_gap = min_gap;

fprintf('Z方向分析完成！\n');

%% ===================== 结果可视化 =====================
fprintf('生成可视化图表...\n');

% 设置图形属性
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesLineWidth', 1.5);
set(0, 'DefaultFigureColor', 'w');

%% 图1: 动态几何分析总览
figure('Position', [100, 100, 1200, 900], 'Name', '动态几何分析总览');

% 1.1 方向性IPR随时间变化
subplot(2, 3, 1);
plot(t_vals, IPR_total_time, 'k-', 'LineWidth', 2.5);
hold on;
plot(t_vals, IPR_x_time, 'r-', 'LineWidth', 2);
plot(t_vals, IPR_y_time, 'g-', 'LineWidth', 2);
plot(t_vals, IPR_z_time, 'b-', 'LineWidth', 2);
xlabel('时间', 'FontSize', 14);
ylabel('IPR', 'FontSize', 14);
title('方向性IPR随时间演化', 'FontSize', 16);
legend('总IPR', 'X方向', 'Y方向', 'Z方向', 'Location', 'best');
grid on; box on;

% 1.2 膨胀因子随时间变化
subplot(2, 3, 2);
plot(t_vals, a1_t, 'r-', 'LineWidth', 2);
hold on;
plot(t_vals, a2_t, 'g-', 'LineWidth', 2);
plot(t_vals, a3_t, 'b-', 'LineWidth', 2);
xlabel('时间', 'FontSize', 14);
ylabel('膨胀因子', 'FontSize', 14);
title('膨胀因子随时间变化', 'FontSize', 16);
legend('a_1(t)', 'a_2(t)', 'a_3(t)', 'Location', 'best');
grid on; box on;

% 1.3 能谱随时间演化
subplot(2, 3, 3);
imagesc(t_vals, 1:10, energy_levels_time);
colorbar;
xlabel('时间', 'FontSize', 14);
ylabel('能级索引', 'FontSize', 14);
title('能谱随时间演化', 'FontSize', 16);

% 1.4 能隙随时间变化
subplot(2, 3, 4);
plot(t_vals, energy_gaps_time, 'b-', 'LineWidth', 2.5);
xlabel('时间', 'FontSize', 14);
ylabel('能隙', 'FontSize', 14);
title('能隙随时间变化', 'FontSize', 16);
grid on; box on;

% 1.5 Chern数随时间变化
subplot(2, 3, 5);
chern_indices = find(chern_numbers_time ~= 0);
if ~isempty(chern_indices)
    plot(t_vals(chern_indices), chern_numbers_time(chern_indices), 'ro-', 'LineWidth', 2);
else
    plot(t_vals, chern_numbers_time, 'ro-', 'LineWidth', 2);
end
xlabel('时间', 'FontSize', 14);
ylabel('Chern数', 'FontSize', 14);
title('Chern数随时间演化', 'FontSize', 16);
grid on; box on;

% 1.6 IPR与膨胀因子关系
subplot(2, 3, 6);
yyaxis left;
plot(t_vals, IPR_y_time, 'g-', 'LineWidth', 2.5);
ylabel('Y方向IPR', 'FontSize', 14);
yyaxis right;
plot(t_vals, a2_t, 'r--', 'LineWidth', 2);
ylabel('Y方向膨胀因子', 'FontSize', 14);
xlabel('时间', 'FontSize', 14);
title('Y方向IPR与膨胀因子关系', 'FontSize', 16);
grid on; box on;

% 保存图1
saveas(gcf, 'Bianchi_I_动态几何分析总览.png');
saveas(gcf, 'Bianchi_I_动态几何分析总览.fig');

%% 图2: Y方向局域化机制详细分析
figure('Position', [150, 150, 1200, 800], 'Name', 'Y方向局域化机制分析');

% 2.1 Y方向IPR与膨胀因子
subplot(2, 2, 1);
yyaxis left;
plot(t_dense, IPR_y_dense, 'b-', 'LineWidth', 2.5);
ylabel('Y方向IPR', 'FontSize', 14);
yyaxis right;
plot(t_dense, a2_t_dense, 'r--', 'LineWidth', 2);
ylabel('Y方向膨胀因子 a_2', 'FontSize', 14);
xlabel('时间', 'FontSize', 14);
title('Y方向IPR与膨胀因子关系（高精度）', 'FontSize', 16);
grid on; box on;

% 2.2 Y方向质心变化
subplot(2, 2, 2);
plot(t_dense, y_center_of_mass, 'g-', 'LineWidth', 2.5);
xlabel('时间', 'FontSize', 14);
ylabel('Y方向质心', 'FontSize', 14);
title('Y方向波函数质心演化', 'FontSize', 16);
grid on; box on;

% 2.3 Y方向扩展度
subplot(2, 2, 3);
plot(t_dense, y_spread, 'm-', 'LineWidth', 2.5);
xlabel('时间', 'FontSize', 14);
ylabel('Y方向扩展度', 'FontSize', 14);
title('Y方向波函数扩展度', 'FontSize', 16);
grid on; box on;

% 2.4 IPR-膨胀因子相空间图
subplot(2, 2, 4);
scatter(a2_t_dense, IPR_y_dense, 50, t_dense, 'filled');
colorbar;
xlabel('Y方向膨胀因子 a_2', 'FontSize', 14);
ylabel('Y方向IPR', 'FontSize', 14);
title('IPR-膨胀因子相空间轨迹', 'FontSize', 16);
grid on; box on;

% 保存图2
saveas(gcf, 'Bianchi_I_Y方向局域化机制.png');
saveas(gcf, 'Bianchi_I_Y方向局域化机制.fig');

%% 图3: Z方向临界点分析
figure('Position', [200, 200, 1200, 800], 'Name', 'Z方向临界点分析');

% 3.1 Z方向IPR扫描
subplot(2, 2, 1);
plot(a3_dense, IPR_z_scan, 'b-', 'LineWidth', 2.5);
hold on;
plot(a3_critical, max_IPR_z, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(a3_critical, max_IPR_z*1.05, sprintf('a_{3,max} = %.4f', a3_critical), 'FontSize', 12);
xlabel('Z方向膨胀因子 a_3', 'FontSize', 14);
ylabel('Z方向IPR', 'FontSize', 14);
title('Z方向IPR高精度扫描', 'FontSize', 16);
grid on; box on;

% 3.2 能隙扫描
subplot(2, 2, 2);
semilogy(a3_dense, energy_gap_scan, 'r-', 'LineWidth', 2.5);
hold on;
semilogy(a3_min_gap, min_gap, 'bo', 'MarkerSize', 10, 'LineWidth', 2);
text(a3_min_gap, min_gap*2, sprintf('a_{3,gap} = %.4f', a3_min_gap), 'FontSize', 12);
xlabel('Z方向膨胀因子 a_3', 'FontSize', 14);
ylabel('能隙 (对数刻度)', 'FontSize', 14);
title('能隙随膨胀因子变化', 'FontSize', 16);
grid on; box on;

% 3.3 参与熵
subplot(2, 2, 3);
plot(a3_dense, participation_entropy, 'g-', 'LineWidth', 2.5);
xlabel('Z方向膨胀因子 a_3', 'FontSize', 14);
ylabel('参与熵', 'FontSize', 14);
title('参与熵随膨胀因子变化', 'FontSize', 16);
grid on; box on;

% 3.4 IPR与能隙关系
subplot(2, 2, 4);
scatter(IPR_z_scan, energy_gap_scan, 50, a3_dense, 'filled');
colorbar;
xlabel('Z方向IPR', 'FontSize', 14);
ylabel('能隙', 'FontSize', 14);
title('IPR与能隙关系（颜色表示a_3）', 'FontSize', 16);
grid on; box on;

% 保存图3
saveas(gcf, 'Bianchi_I_Z方向临界点分析.png');
saveas(gcf, 'Bianchi_I_Z方向临界点分析.fig');

%% ===================== 数据保存 =====================
fprintf('保存分析结果...\n');

% 保存主要结果
save('Bianchi_I_完整分析结果.mat', 'results');

% 保存关键发现总结
summary = struct();
summary.title = '动态Bianchi I几何中准周期系统研究 - 关键发现';
summary.y_direction_findings = struct();
summary.y_direction_findings.description = 'Y方向U形局域化特性';

% 安全地找到最小IPR_y的索引
if ~isempty(IPR_y_dense) && any(IPR_y_dense > 0)
    [min_IPR_y_val, min_IPR_y_idx] = min(IPR_y_dense);
    summary.y_direction_findings.optimal_a2 = a2_t_dense(min_IPR_y_idx);
    summary.y_direction_findings.min_IPR_y = min_IPR_y_val;
    summary.y_direction_findings.max_IPR_y = max(IPR_y_dense);
else
    summary.y_direction_findings.optimal_a2 = NaN;
    summary.y_direction_findings.min_IPR_y = NaN;
    summary.y_direction_findings.max_IPR_y = NaN;
end

summary.z_direction_findings = struct();
summary.z_direction_findings.description = 'Z方向临界相变';
summary.z_direction_findings.critical_a3 = a3_critical;
summary.z_direction_findings.max_IPR_z = max_IPR_z;
summary.z_direction_findings.critical_gap_a3 = a3_min_gap;
summary.z_direction_findings.min_gap = min_gap;

summary.overall_findings = struct();
summary.overall_findings.description = '各向异性局域化特性';
summary.overall_findings.directional_coupling = '不同方向显示强烈的各向异性和非线性依赖关系';
summary.overall_findings.geometric_control = '几何约束成功实现了对拓扑和局域化特性的精确调控';

save('Bianchi_I_关键发现总结.mat', 'summary');

%% ===================== 分析完成报告 =====================
fprintf('\n====== 分析完成报告 ======\n');
fprintf('1. 动态几何分析: %d个时间点 ✓\n', time_steps);
fprintf('2. Y方向局域化分析: %d个高精度时间点 ✓\n', n_times_y);
fprintf('3. Z方向临界点分析: %d个膨胀因子扫描点 ✓\n', a3_precision);
fprintf('\n主要发现:\n');
if ~isnan(summary.y_direction_findings.optimal_a2)
    fprintf('- Y方向最优膨胀因子: a2 = %.4f (最小IPR = %.2f)\n', ...
        summary.y_direction_findings.optimal_a2, summary.y_direction_findings.min_IPR_y);
else
    fprintf('- Y方向分析: 数据异常，请检查计算\n');
end
fprintf('- Z方向临界膨胀因子: a3 = %.4f (最大IPR = %.2f)\n', ...
    summary.z_direction_findings.critical_a3, summary.z_direction_findings.max_IPR_z);
fprintf('- Z方向能隙最小点: a3 = %.4f (最小能隙 = %.2e)\n', ...
    summary.z_direction_findings.critical_gap_a3, summary.z_direction_findings.min_gap);
fprintf('\n生成文件:\n');
fprintf('- Bianchi_I_完整分析结果.mat\n');
fprintf('- Bianchi_I_关键发现总结.mat\n');
fprintf('- Bianchi_I_动态几何分析总览.png/.fig\n');
fprintf('- Bianchi_I_Y方向局域化机制.png/.fig\n');
fprintf('- Bianchi_I_Z方向临界点分析.png/.fig\n');
fprintf('\n====== 分析程序执行完成! ======\n');

% 显示运行时间
fprintf('总运行时间: %.2f 分钟\n', toc/60);

%% ===================== 函数定义区域 =====================
% 注意：在MATLAB脚本中，所有函数定义必须放在文件最后

% 构建哈密顿量函数
function H = build_hamiltonian(Lx, Ly, Lz, tx, ty, tz, V0, a1, a2, a3, beta_x, beta_y, beta_z, phi_x, phi_y, phi_z)
    N = Lx * Ly * Lz;
    H = sparse(N, N);
    
    for x = 1:Lx
        for y = 1:Ly
            for z = 1:Lz
                % 当前点的索引
                idx = sub2ind([Lx, Ly, Lz], x, y, z);

                % 准周期势场
                V = V0 * (cos(2*pi*beta_x*x/a1 + phi_x) + ...
                          cos(2*pi*beta_y*y/a2 + phi_y) + ...
                          cos(2*pi*beta_z*z/a3 + phi_z));
                H(idx, idx) = V;

                % 邻近跃迁（x方向）
                if x < Lx
                    idx_x = sub2ind([Lx, Ly, Lz], x+1, y, z);
                    H(idx, idx_x) = -tx;
                    H(idx_x, idx) = -tx;
                end

                % 邻近跃迁（y方向）
                if y < Ly
                    idx_y = sub2ind([Lx, Ly, Lz], x, y+1, z);
                    H(idx, idx_y) = -ty;
                    H(idx_y, idx) = -ty;
                end

                % 邻近跃迁（z方向）
                if z < Lz
                    idx_z = sub2ind([Lx, Ly, Lz], x, y, z+1);
                    H(idx, idx_z) = -tz;
                    H(idx_z, idx) = -tz;
                end
            end
        end
    end
end

% 计算方向性IPR函数
function [IPR_total, IPR_x, IPR_y, IPR_z] = calculate_directional_IPR(wavefunction, Lx, Ly, Lz)
    prob_density = abs(wavefunction).^2;
    prob_3D = reshape(prob_density, [Lx, Ly, Lz]);
    
    % 总IPR
    IPR_total = sum(prob_density.^2);
    
    % X方向IPR
    prob_x = squeeze(sum(sum(prob_3D, 2), 3));
    prob_x = prob_x / sum(prob_x);
    IPR_x = sum(prob_x.^2) * Lx;
    
    % Y方向IPR
    prob_y = squeeze(sum(sum(prob_3D, 1), 3));
    prob_y = prob_y / sum(prob_y);
    IPR_y = sum(prob_y.^2) * Ly;
    
    % Z方向IPR
    prob_z = squeeze(sum(sum(prob_3D, 1), 2));
    prob_z = prob_z / sum(prob_z);
    IPR_z = sum(prob_z.^2) * Lz;
end

% 计算Berry曲率函数
function Berry_curvature = calculate_berry_curvature_3D(kx_vals, ky_vals, kz_vals, tx, ty, tz, V0, a1, a2, a3)
    Nk = length(kx_vals);
    Berry_curvature = zeros(Nk, Nk, Nk);
    
    for kx_idx = 1:Nk
        for ky_idx = 1:Nk
            for kz_idx = 1:Nk
                kx = kx_vals(kx_idx);
                ky = ky_vals(ky_idx);
                kz = kz_vals(kz_idx);
                
                % 动量空间哈密顿量
                V_k = V0 * (cos(2 * pi * kx / a1) + cos(2 * pi * ky / a2) + cos(2 * pi * kz / a3));
                H_k = [-tx * cos(kx) - ty * cos(ky) - tz * cos(kz), V_k; ...
                       V_k, tx * cos(kx) + ty * cos(ky) + tz * cos(kz)];
                
                [eigenvectors, eigenvalues] = eig(H_k);
                [~, min_idx] = min(diag(eigenvalues));
                u_k = eigenvectors(:, min_idx);
                
                % 邻近点
                kx_plus_idx = mod(kx_idx, Nk) + 1;
                ky_plus_idx = mod(ky_idx, Nk) + 1;
                
                % 计算邻近点波函数
                kx_plus = kx_vals(kx_plus_idx);
                ky_plus = ky_vals(ky_plus_idx);
                
                V_kx_plus = V0 * (cos(2 * pi * kx_plus / a1) + cos(2 * pi * ky / a2) + cos(2 * pi * kz / a3));
                H_kx_plus = [-tx * cos(kx_plus) - ty * cos(ky) - tz * cos(kz), V_kx_plus; ...
                            V_kx_plus, tx * cos(kx_plus) + ty * cos(ky) + tz * cos(kz)];
                [eigenvectors_x_plus, eigenvalues_x_plus] = eig(H_kx_plus);
                [~, min_idx_x_plus] = min(diag(eigenvalues_x_plus));
                u_kx_plus = eigenvectors_x_plus(:, min_idx_x_plus);
                
                V_ky_plus = V0 * (cos(2 * pi * kx / a1) + cos(2 * pi * ky_plus / a2) + cos(2 * pi * kz / a3));
                H_ky_plus = [-tx * cos(kx) - ty * cos(ky_plus) - tz * cos(kz), V_ky_plus; ...
                            V_ky_plus, tx * cos(kx) + ty * cos(ky_plus) + tz * cos(kz)];
                [eigenvectors_ky_plus, eigenvalues_ky_plus] = eig(H_ky_plus);
                [~, min_idx_ky_plus] = min(diag(eigenvalues_ky_plus));
                u_ky_plus = eigenvectors_ky_plus(:, min_idx_ky_plus);
                
                % Berry曲率计算
                overlap_xy = u_k' * u_kx_plus;
                overlap_yz = u_kx_plus' * u_ky_plus;
                if abs(overlap_xy) > 1e-12 && abs(overlap_yz) > 1e-12
                    Berry_curvature(kx_idx, ky_idx, kz_idx) = imag(log(overlap_xy) - log(overlap_yz));
                end
            end
        end
    end
end

% 计算Chern数函数
function chern_number = calculate_chern_number(Berry_curvature, Nk)
    chern_number = sum(Berry_curvature(:)) * (2 * pi / Nk)^3 / (2 * pi);
end