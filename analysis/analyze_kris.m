%%  load

root_p = "/Users/nick/source/mattarlab/planning_code/computational_model/results";
save_p = fullfile( fileparts(root_p), 'figs' );

rew_plan_res = load( fullfile(root_p, "rew_and_plan_by_n_8.mat") );
rew_plan_res4 = load( fullfile(root_p, "rew_and_plan_by_n_4.mat") );

rew_plan_t8 = analyse_by_n_to_table( rew_plan_res );
rew_plan_t8.lplan(:) = 8;

rew_plan_t4 = analyse_by_n_to_table( rew_plan_res4 );
rew_plan_t4.lplan(:) = 4;

rew_plan_t = [ rew_plan_t8; rew_plan_t4 ];

[policies, scalars] = load_perf_by_rollout_number( root_p );

%%  epochs at which models reach a target average reward

data_vars = {'rew', 'planfrac'};
t = summarize( rew_plan_t, {'seed'}, data_vars, @mean );

match_vars = setdiff( vnames(t), ['epoch', data_vars] );
I = findeach( t, match_vars );

subi = [];
sub_targ_rews = [];
abs_errs = [];

targ_rews = 1:0.25:6.5;

for ri = 1:numel(targ_rews)
  targ_rew = targ_rews(ri);

  for i = 1:numel(I)
    ind = I{i};
    delta = t.rew(ind) - targ_rew;
    delta(delta < 0) = inf;  %  require rewards >= targ_rew
    if ( all(isinf(delta)) )
      error('No rewards at least: %0.3f', targ_rew); 
    end
    [abs_err, ni] = min( abs(delta) );
    subi(end+1, 1) = ind(ni);
    sub_targ_rews(end+1, 1) = targ_rew;
    abs_errs(end+1, 1) = abs_err;
  end
end

targ_rew_t = t(subi, :);
targ_rew_t.target_rew = sub_targ_rews;
targ_rew_t.target_rew_err = abs_errs;

if ( 1 )
  % reject target rewards where a model is too far from the target
  I = findeach( targ_rew_t, 'target_rew' );
  accept_targ = cellfun( @(x) all(targ_rew_t.target_rew_err(x) < 1), I );
  ind = cat( 1, I{accept_targ} );
  targ_rew_t = targ_rew_t(ind, :);
end

%%  planning probability over learning

%{
  res.planfracs (ihid, iseed, iepoch)
%}

episodes_per_epoch = 40 * 200; %  @TODO: This was fixed during training.

p_plan = squeeze( mean(rew_plan_res.planfracs, 2) ); %  mean across seeds
mu_entropy = squeeze( mean(rew_plan_res.mu_entropies, 2) ); %  mean across seeds
tot_episodes = rew_plan_res.epochs * episodes_per_epoch;

figure(1); clf;
subplot( 1, 2, 1 );
plot_by_model_capacity( tot_episodes, p_plan, rew_plan_res.Nhiddens );
title( 'p(plan)' ); ylim( [0, 1] );
subplot( 1, 2, 2 );
plot_by_model_capacity( tot_episodes, mu_entropy, rew_plan_res.Nhiddens );
title( 'policy entropy' ); ylim( [0, 200] );

%%  policy entropy, organized as time course over planning steps

vn = @vnames;

% Ignore planning action (5)
ps = softmax( policies.policy(:, 1:4), 2 );
entropies = -sum( ps .* log(ps), 2 );

[I, C] = findeach( policies, setdiff(vnames(policies) ...
  , {'policy', 'replication', 'seed'}) );
mu_entropies = rowifun( @nanmean, I, entropies );

% organize as time course over planning steps
[entropy_t, I] = rowify_vars( C, {'ith_plan_step'} );
entropy_t.entropy = cate1( cellfun(@(x) mu_entropies(x)', I, 'un', 0) );

%%  visualize decay in entropy per additional planning step

overlay_fit = false;
do_save = true;
match_mean_rew = false;

sliced_epoch = max( entropy_t.epoch );

if ( match_mean_rew )
  tv = intersect( vnames(entropy_t), vnames(targ_rew_t) );
  max_targ_rew = targ_rew_t.target_rew == max( targ_rew_t.target_rew );
  slice_mask = ...
    ismember( entropy_t(:, tv), targ_rew_t(max_targ_rew, tv) ) ...
    & entropy_t.ith_action_to_goal == 1 ...
    & ~entropy_t.planning_disabled;
  match_str = "match_performance";
else
  slice_mask = ...
      entropy_t.epoch == sliced_epoch ...
    & entropy_t.ith_action_to_goal == 1 ...
    & ~entropy_t.planning_disabled;
  match_str = "match_experience";
end

slice_subset = entropy_t(slice_mask, :);

label_vars = string( {'units', 'epoch', 'lplan'} );
slice_subset = sortrows( slice_subset, label_vars );

figure(1); clf;
h = plot( slice_subset.ith_plan_step', slice_subset.entropy', 'linewidth', 2 );
h_lab = table2str( slice_subset(:, label_vars) );
arrayfun( @(h, s) set(h, 'displayname', s), h, h_lab );
legend( h );
xlabel( 'Rollout #' );
ylabel( 'Entropy' );

h_cs = lplan_units_colors( slice_subset );
for i = 1:numel(h), set(h(i), 'color', h_cs(i, :)); end

if ( overlay_fit )
  for i = 1:size(slice_subset, 1)
    % mdl = @(b,x)b(1).*exp(b(2).*x)+b(3);
    % x0 = [20,.001,0];
  
    % mdl = @(b,x)exp(b(1).*x)+b(2);
    % x0 = [-10,0];
  
    mdl = @(b,x)b(1).*exp(-2e-1*x)+b(2);
    x0 = [1,0];
  
    x = slice_subset.ith_plan_step(i, :);
    f = fitnlm(x, slice_subset.entropy(i, :), mdl, x0);
    b1 = f.Coefficients.Estimate(1);
  
    hold on;
    hf = plot( x, predict(f, x(:)), 'LineWidth', 4, 'LineStyle', '--' );
    set( hf, 'color', get(h(i), 'color') ...
      , 'displayname', compose("beta = %0.3f", b1) );
  end
end

% title( compose("Epoch = %d", sliced_epoch) );
title( "Entropy reduction per rollout" );
ylim( [0, 1] );

if ( do_save )
  fname = compose( "reduction_in_policy_entropy_%s.png", match_str );
  saveas(gcf, fullfile(save_p, fname)); 
end

%%  visualize change in value function error per additional planning step

data_vars = { 'vf_error', 'rew' };
scalar_t = rowify_vars( scalars, [{'ith_plan_step'}, data_vars] );
scalar_t = summarize( scalar_t, {'replication', 'seed'}, data_vars, @mean );

match_mean_rew = true;
do_save = true;

if ( match_mean_rew )
  tv = intersect( vnames(scalar_t), vnames(targ_rew_t) );
  tv = setdiff( tv, data_vars );
  max_targ_rew = targ_rew_t.target_rew == max( targ_rew_t.target_rew );
  slice_mask = ...
    ismember( scalar_t(:, tv), targ_rew_t(max_targ_rew, tv) ) ...
    & ~scalar_t.planning_disabled;
  match_str = "match_performance";
else
  slice_mask = scalar_t.epoch == max(scalar_t.epoch) & ~scalar_t.planning_disabled;
  match_str = "match_experience";
end

label_vars = ["units", "epoch", "lplan"];
slice_subset = scalar_t(slice_mask, :);
slice_subset = sortrows( slice_subset, label_vars );

figure(1); clf;
h = plot( slice_subset.ith_plan_step', slice_subset.vf_error', 'linewidth', 2 );
h_lab = table2str( slice_subset(:, label_vars) );
arrayfun( @(h, s) set(h, 'displayname', s), h, h_lab );
hl = legend( h ); set( hl, 'location', 'southeast' );
xlabel( 'Rollout #' );
ylabel( 'vf error' );
title( 'Error in value function estimates for increasing amounts of deliberation' );
ylim( [3, 6] );

h_cs = lplan_units_colors( slice_subset );
for i = 1:numel(h), set(h(i), 'color', h_cs(i, :)); end

if ( do_save )
  fname = compose( "change_in_v_error_%s.png", match_str );
  saveas(gcf, fullfile(save_p, fname)); 
end

%%  plot slopes generated by an exp. fit over the course of learning

epochs = unique( entropy_t.epoch );
slope_ts = table();

match_mean_rew = true;
do_save = true;

plt_var = "slopes";
% plt_var = "entropy_deltas";

if ( match_mean_rew )
  tv = intersect( vnames(entropy_t), vnames(targ_rew_t) );
  targ_rews = unique( targ_rew_t.target_rew );

  slice_mask = cell( size(targ_rews) );
  xs = targ_rews;
  for i = 1:numel(targ_rews)
    match_targ_rew = targ_rew_t.target_rew == targ_rews(i);
    slice_mask{i} = ...
      ismember( entropy_t(:, tv), targ_rew_t(match_targ_rew, tv) ) ...
      & entropy_t.ith_action_to_goal == 1 ...
      & ~entropy_t.planning_disabled;
  end
  xlab = 'mean reward';
  match_str = "match_performance";
else
  slice_mask = cell( size(epochs) );
  xs = epochs;
  for i = 1:numel(epochs)
    slice_mask{i} = ...
        entropy_t.epoch == epochs(i) ...
      & entropy_t.ith_action_to_goal == 1 ...
      & ~entropy_t.planning_disabled;
  end
  xlab = 'training epoch';
  match_str = "match_experience";
end

ns = 3; %  number of steps over which to average early and late entropies
for si = 1:numel(slice_mask)

slice_subset = entropy_t(slice_mask{si}, :);
slice_subset.slopes(:) = nan;
slice_subset.x(:) = xs(si);
slice_subset.entropy_deltas(:) = nan;

for i = 1:size(slice_subset, 1)
  mdl = @(b,x)b(1).*exp(-2e-1*x)+b(2);
  x0 = [1,0];

  ss = slice_subset(i, :);

  max_step = max( ss.ith_plan_step );
  entropy0 = mean( ss.entropy(ismember(ss.ith_plan_step, 0:ns)) );
  entropy1 = mean( ss.entropy(ismember(ss.ith_plan_step, max_step-ns:max_step)) );
  entropyd = entropy1 - entropy0;

  x = slice_subset.ith_plan_step(i, :);
  f = fitnlm( x, slice_subset.entropy(i, :), mdl, x0 );
  slice_subset.slopes(i) = f.Coefficients.Estimate(1);
  slice_subset.entropy_deltas(i) = -entropyd;
end

slope_ts = [ slope_ts; slice_subset ];

end

label_vars = string( {'units', 'lplan', 'planning_disabled'} );

slope_ts = sortrows( slope_ts, label_vars );
[I, C] = findeach( slope_ts, label_vars );
x = cate1( rowifun(@(x) x', I, slope_ts.x, 'un', 0) );
y = cate1( rowifun(@(x) x', I, slope_ts.(plt_var), 'un', 0) );

figure(1); clf;
h = plot( x', y', 'linewidth', 2 );
cs = lplan_units_colors( C );
s = plots.strip_underscore( table2str(C) );
arrayfun( @(h, s) set(h, 'displayname', s), h, s );
for i = 1:numel(h), set(h(i), 'color', cs(i, :)); end
hl = legend( h ); set( hl, 'location', 'southeast' );
xlabel( xlab );
ylabel( 'Slope of exp. decay in entropy per additional rollout' );
title( 'Rollout sensitivity (reduction in entropy)' );

if ( plt_var == "slopes" )
  ylim( [-0.1, 1] );
elseif ( plt_var == "entropy_deltas" )
  ylim( [-0.1, 1] );
end

if ( do_save )
  % shared_utils.plot.fullscreen( gcf );
  fname = compose( "entropy_sensitivity_%s.png", match_str );
  saveas(gcf, fullfile(save_p, fname)); 
end

%%  plot effects of multiple rollouts over learning

figure(1); clf;
num_rollouts = 0:15;
rollout_subsets = 1:5:numel(num_rollouts);

plt_t = entropy_t;
tv = 'entropy';
plt_each = {'units', 'planning_disabled', 'lplan'};
plt_mask = ismember( entropy_t.ith_action_to_goal, [1] );

plt_t = rowify_vars( scalars, {'ith_plan_step', 'vf_error', 'rew'} );
plt_t = summarize( plt_t, {'replication', 'seed'}, {'vf_error', 'rew'}, @mean );
tv = 'vf_error';
plt_each = {'units', 'planning_disabled', 'lplan'};
plt_mask = rowmask( plt_t );

[I, C] = findeach( plt_t, plt_each, plt_mask );
axs = plots.panels( numel(I) );

for axi = 1:numel(I)

ax = axs(axi);
subset = plt_t(I{axi}, :);
h = plot( ax, subset.epoch, subset.(tv)(:, rollout_subsets), 'LineWidth', 2 );
colors = spring( numel(h) );
names = compose( "Rollout = %d", num_rollouts(rollout_subsets) );
for i = 1:numel(h), set(h(i), 'color', colors(i, :), 'displayname', names(i)); end
legend( h );
xlabel( ax, 'Training epoch' );
ylabel( ax, plots.strip_underscore(tv) );

s = plots.strip_underscore( table2str(C(axi, :)) );
title( ax, s );
% ylim( ax, [0, 2] );

end

plots.onelegend( gcf );

%%

function [policies, scalars] = load_perf_by_rollout_number(root_p)

policies = table();
scalars = table();

for epoch = 0:100:1000
  for lp = [4, 8]
    units = [60, 100];
    for unit = units
      fname = compose( "perf_by_n_epoch_%d_N%d_Lplan%d.mat", epoch, unit, lp );
      rollout_res = load( fullfile(root_p, fname) );

      policy_t = rollout_res_policies_to_table( rollout_res );
      scalar = rollout_res_to_scalars( rollout_res );

      vns = ["policy_t", "scalar"];
      for i = 1:numel(vns)
        eval(compose("%s.units(:) = unit;", vns(i)));
        eval(compose("%s.epoch(:) = epoch;", vns(i)));
        eval(compose("%s.lplan(:) = lp;", vns(i)));
      end

      policies = [ policies; policy_t ];
      scalars = [ scalars; scalar ];
    end
  end
end

end

function plot_by_model_capacity(x, y, nhid)
h = plot( x, y, 'linewidth', 2 );
arrayfun( @(h, x) set(h, 'DisplayName', x), h, compose("Hidden = %d", nhid) );
legend( h );
xlabel( 'Total # training episodes' );
end

function t = analyse_by_n_to_table(rew_plan_res)

mu_rs = [];
mu_ps = [];

nh = [];
seed = [];
epoch = [];

for i = 1:numel(rew_plan_res.Nhiddens)
  for j = 1:numel(rew_plan_res.seeds)
    for k = 1:numel(rew_plan_res.epochs)
      nh(end+1, 1) = rew_plan_res.Nhiddens(i);
      seed(end+1, 1) = rew_plan_res.seeds(j);
      epoch(end+1, 1) = rew_plan_res.epochs(k);

      mu_rs(end+1, 1) = rew_plan_res.meanrews(i, j, k);
      mu_ps(end+1, 1) = rew_plan_res.planfracs(i, j, k);
    end
  end
end

t = table( mu_rs, mu_ps, nh, seed, epoch ...
  , 'va', {'rew', 'planfrac', 'units', 'seed', 'epoch'} );

end

function ts = rollout_res_to_scalars(res)

% (plan enabled/disabled, batch, #rollouts)

fs = fieldnames( res );
vf_fs = fs(contains(fs, 'vf_error'));
rew_fs = fs(contains(fs, 'rew'));
second = @(x) string(x{2});
seeds = cellfun( ...
  @(x) extract(second(strsplit(x, 'seed_')), digitsPattern), vf_fs );
seeds = double( seeds );

pds = logical.empty;
nrs = [];
nps = [];
t_seeds = [];
vf_errs = [];
rews = [];

for i = 1:numel(vf_fs)
  fprintf( '\n\t %d of %d', i, numel(vf_fs) );

  v = res.(vf_fs{i});
  r = res.(rew_fs{i});
  
  nreps = size( v, 2 );
  nplan = 0:15; % @TODO: This was fixed during evaluation
  seed = seeds(i);

  for ic = 1:size(v, 1)
    planning_disabled = ic == 2;
    for nr = 1:nreps
      for np = 1:numel(nplan)
        vf_errs(end+1, :) = squeeze( v(ic, nr, np) );
        rews(end+1, :) = squeeze( r(ic, nr, np) );
        pds(end+1, :) = planning_disabled;
        nrs(end+1, :) = nr;
        nps(end+1, :) = nplan(np);
        t_seeds(end+1, :) = seed;
      end
    end
  end
end

ts = table( vf_errs, rews, pds, nrs, nps, t_seeds ...
            , 'va', {'vf_error', 'rew', 'planning_disabled' ...
            , 'replication', 'ith_plan_step', 'seed'} );


end

function ts = rollout_res_policies_to_table(res)

% policies = zeros(2, nreps, length(nplans), 10, 5) .+ NaN;
% (plan enabled/disabled, batch, #rollouts, #actions to goal, policy logits)
%
% This is different from the scalar case above because the policy is
% separately evaluated for different goal distances (i.e., # actions to goal)

fs = fieldnames( res );
policy_fs = fs(contains(fs, 'policies'));
second = @(x) string(x{2});
seeds = cellfun( ...
  @(x) extract(second(strsplit(x, 'seed_')), digitsPattern), policy_fs );
seeds = double( seeds );

pds = logical.empty;
nrs = [];
nps = [];
t_seeds = [];
policies = [];
nas = [];

for i = 1:numel(policy_fs)
  fprintf( '\n\t %d of %d', i, numel(policy_fs) );

  v = res.(policy_fs{i});
  nreps = size( v, 2 );
  nplan = 0:15; % @TODO: This was fixed during evaluation
  nact_to_goal = size( v, 4 ); % action number to goal
  num_actions = size( v, 5 );
  seed = seeds(i);

  for ic = 1:size(v, 1)
    planning_disabled = ic == 2;
    for nr = 1:nreps
      for np = 1:numel(nplan)
        for na = 1:nact_to_goal
          slice = squeeze( v(ic, nr, np, na, :) )';
          policies(end+1, :) = slice;
          pds(end+1, :) = planning_disabled;
          nrs(end+1, :) = nr;
          nps(end+1, :) = nplan(np);
          nas(end+1, :) = na;
          t_seeds(end+1, :) = seed;
        end
      end
    end
  end
end

ts = table( policies, pds, nrs, nps, nas, t_seeds ...
            , 'va', {'policy', 'planning_disabled' ...
            , 'replication', 'ith_plan_step', 'ith_action_to_goal', 'seed'} );

end

function v = vnames(t)
v = t.Properties.VariableNames;
end

function y = softmax(x, varargin)
y = exp( x ) ./ sum( exp(x), varargin{:} );
end

function s = table2str(t)
s = strings( size(t, 1), 1 );
for idx = 1:size(t, 1)
  s(idx) = row2str( t(idx, :) );
end
function s = row2str(t)
  s = "";
  for i = 1:size(t, 2)
    v = compose( "%s=%s", t.Properties.VariableNames{i}, string(t{:, i}) );
    s = s + v;
    if ( i + 1 <= size(t, 2) )
      s = s + " | ";
    end
  end
end
end

function [t, I] = rowify_vars(T, rest)

rest = string( rest );
[I, t] = findeach( T, setdiff(vnames(T), rest) );

for i = 1:numel(rest)
  t.(rest(i)) = cate1( cellfun(@(x) reshape_var(T.(rest(i)), x), I, 'un', 0) );
end

function dv = reshape_var(v, i)
  clns = colons( ndims(v) - 1 );
  subv = v(i, clns{:});
  dv = reshape( subv, [1, size(subv)] );
end

end

function [t, I] = summarize(T, across, vs, fs)

vs = string( vs );

if ( isa(fs, 'function_handle') )
  fs = repmat( {fs}, size(vs) );
else
  validateattributes( fs, {'cell'}, {'2d'}, mfilename, 'fs' );
end

[I, t] = findeach( T, setdiff(vnames(T), [string(across), string(vs)]) );
for i = 1:numel(vs)
  t.(vs(i)) = cate1( rowifun(fs{i}, I, T.(vs(i)), 'un', 0) );
end

end

function cs = lplan_units_colors(t)

short_plan = t.lplan == 4;
long_plan = t.lplan == 8;
large_net = t.units == 100;
small_net = t.units == 60;

large_long = [ 1, 0, 0 ];
large_short = large_long .* 0.75;

small_long = [ 0, 0, 1 ];
small_short = small_long .* 0.75;

cs = zeros( size(t, 1), 3 );
cs(large_net & long_plan, :) = repmat( large_long, sum(large_net & long_plan), 1 );
cs(large_net & short_plan, :) = repmat( large_short, sum(large_net & short_plan), 1 );

cs(small_net & long_plan, :) = repmat( small_long, sum(small_net & long_plan), 1 );
cs(small_net & short_plan, :) = repmat( small_short, sum(small_net & short_plan), 1 );

end